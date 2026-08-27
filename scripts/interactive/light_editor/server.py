# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Serve the light scene editor and write edits back to the scene JSON.

Usage:
    python server.py --scene <scene_state.json> [--port 8770] [--no-extract]

Runs the extractor first unless --no-extract is passed, then serves web/ at
http://localhost:<port>. Binds to loopback only.
"""

import argparse
import atexit
import base64
import binascii
import copy
import difflib
import hashlib
import io
import json
import os
import queue
import secrets
import shutil
import subprocess
import sys
import traceback
import tempfile
import threading
import socket
import yaml
try:
    # Optional: ruamel round-trips the comments in task configs; PyYAML cannot.
    from ruamel.yaml import YAML, YAMLError as RuamelError
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
except ImportError:
    YAML = None
    CommentedMap = CommentedSeq = None
    # Alias keeps the save path's `except` clause valid without ruamel.
    RuamelError = yaml.YAMLError
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from camera_io import (  # noqa: E402
    CFG_SUBDIR as CAMERA_CFG_SUBDIR,
    apply_camera_edits,
    camera_config_paths,
    camera_config_text,
    camera_export_path,
    load_cameras,
    observation_key_map,
    resolve_camera_config,
    validate_camera_edits,
)
from asset_import import (  # noqa: E402
    classify,
    convert_mesh,
    describe_mesh,
    describe_usd,
    import_usd_file,
    resolve_user_path,
)
from asset_import import slug as import_slug  # noqa: E402
from compose import compose_scene, template_summary, validate_scene_name  # noqa: E402
from scene_catalog import (  # noqa: E402
    RecentScenes,
    discover_scenes,
    resolve_scene_path,
    generated_scene_name,
    scene_roots,
    scene_source,
)
from asset_library import (  # noqa: E402
    default_roots,
    discover_assets,
    import_asset,
    object_spec,
    resolve_key,
    unique_object_name,
)
import export_bundle  # noqa: E402
import robot_cameras  # noqa: E402
import task_bindings  # noqa: E402
import task_propose  # noqa: E402
import task_semantics  # noqa: E402
from scene_io import (  # noqa: E402
    UNCHECKED,
    DEFAULT_DATASET_DIR,
    DEFAULT_ROBOT_ASSET_DIR,
    SceneEditError,
    TargetChanged,
    add_objects,
    apply_authored_state_to_manifest,
    apply_edits,
    atomic_write_text,
    authored_state,
    background_id,
    editable_object_names,
    file_digest,
    guarded_write_text,
    iter_objects,
    joint_facts,
    latest_path,
    background_object_names,
    load_scene,
    physics_facts,
    prepare_scene_document,
    promote_scene_text,
    remove_objects,
    resolve_robot_usd,
    robot_object_names,
    scene_file_mode,
    scene_output_path,
    scene_sha256,
    scene_text,
)
from background_io import (  # noqa: E402
    BACKGROUND_OBJECT_NAME,
    background_roots,
    background_spec,
    default_robot,
    discover_backgrounds,
    estimate_table,
    resolve_background,
    write_table_centre,
)
from background_io import table_centre as background_table_centre  # noqa: E402
from ground_plane import (  # noqa: E402
    apply_ground_plane,
    describe as describe_ground_plane,
    read_ground_plane,
    validate_ground_plane,
)

# Mirrors splat_io's default; not imported because splat_io pulls in OpenUSD.
DEFAULT_SPLAT_BUDGET = 1_000_000

MAX_BODY_BYTES = 8 * 1024 * 1024

# Cap on entries returned per directory listing.
BROWSE_ENTRY_LIMIT = 750

# Uploads are buffered in memory; larger files are imported in place via Browse.
UPLOAD_MAX_BYTES = 256 * 1024 * 1024

# Self-contained formats only: a dropped file cannot carry the textures and
# sublayers a .usd or .obj references by relative path.
UPLOADABLE_SUFFIXES = frozenset({".glb", ".ply", ".stl"})

_upload_staging = None


def upload_staging_dir():
    """A per-process scratch directory for dropped files, cleaned up at exit."""
    global _upload_staging
    if _upload_staging is None:
        _upload_staging = tempfile.mkdtemp(prefix="simfoundry_drop_")
        # Staged files must outlive the request that wrote them.
        atexit.register(shutil.rmtree, _upload_staging, True)
    return Path(_upload_staging)

# Host headers accepted for a loopback bind; anything else is the
# DNS-rebinding shape.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})

# Serialises compare-then-write across all mutation endpoints.
WRITE_LOCK = threading.Lock()

def task_config_root():
    """The only tree "Generate task" may create a file in."""
    return HERE.parents[2] / "scripts" / "cfg" / "task"


# Settling boots Isaac Sim, which can take minutes on a cold cache.
SETTLE_TIMEOUT_S = 900


class HttpRefusal(Exception):
    """An HTTP status and body raised from shared validation."""

    def __init__(self, code, body):
        super().__init__(body.get("error", ""))
        self.code = code
        self.body = body


def _checked_vector(field, value, length, *, positive=False):
    """Validate one transform vector arriving from the browser."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SceneEditError(f"{field} must contain exactly {length} numbers")
    out = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise SceneEditError(f"{field} must contain only numbers")
        number = float(component)
        if number != number or number in (float("inf"), float("-inf")):
            raise SceneEditError(f"{field} must contain only finite numbers")
        if positive and number <= 0.0:
            raise SceneEditError(f"{field} values must be greater than zero")
        out.append(number)
    return out


class SettleRunner:
    """Runs settle.py out of process, after a save, without blocking the response.

    Settling needs OmniGibson, whose Isaac Sim ships its own ``pxr`` and cannot
    share this server's interpreter, so it runs as a subprocess and the browser
    polls for the result. One settle runs at a time; each save takes the next
    generation number, only the newest generation may promote, and queued work
    superseded by a newer save is skipped. Promotion happens here rather than
    via ``--promote`` so it can check the generation is still the newest.
    """

    def __init__(self, python, script, steps, tolerance, on_promote=None,
                 expect_latest=None):
        self.python = python
        self.script = script
        self.steps = steps
        self.tolerance = tolerance
        # Called with (path, digest, document) after a promotion rewrites the
        # source scene, so the session can catch up.
        self.on_promote = on_promote
        # Asked, immediately before the swap, what `_latest` is expected to
        # hold, so an old job cannot clobber work done while physics ran.
        self.expect_latest = expect_latest
        self._lock = threading.Lock()
        self._jobs = {}
        self._next_id = 1
        self._newest = 0
        self._queue = queue.SimpleQueue()
        self._worker = threading.Thread(target=self._serve, daemon=True)
        self._started = False

    @property
    def enabled(self):
        return self.python is not None

    def start(self, scene_path, promote):
        """Queue a settle job and return its id."""
        with self._lock:
            job_id = self._next_id
            self._next_id += 1
            self._newest = job_id
            for other_id, job in self._jobs.items():
                if other_id != job_id and job.get("state") == "queued":
                    job["state"] = "superseded"
                    job["error"] = f"superseded by save {job_id}"
                    print(f"[settle] job {other_id} superseded by {job_id}")
            self._jobs[job_id] = {
                "state": "queued",
                "scene": str(scene_path),
                "generation": job_id,
                "promote_requested": bool(promote),
            }
            if not self._started:
                self._started = True
                self._worker.start()
        self._queue.put((job_id, Path(scene_path), bool(promote)))
        return job_id

    def status(self, job_id):
        with self._lock:
            return copy.deepcopy(self._jobs.get(job_id))

    # Terminal states; the browser polls until it sees one.
    TERMINAL_STATES = frozenset({"done", "failed", "superseded"})

    def _finish(self, job_id, result):
        """Record a job's terminal state; forced terminal so polls end."""
        with self._lock:
            job = self._jobs.setdefault(job_id, {})
            job.update(result)
            if job.get("state") not in self.TERMINAL_STATES:
                job["state"] = "failed" if job.get("error") else "done"

    def _serve(self):
        """Single worker: one OmniGibson process at a time, newest work only."""
        while True:
            job_id, scene_path, promote = self._queue.get()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.get("state") != "queued":
                    # Superseded while it waited.
                    continue
                job["state"] = "running"
            try:
                self._run(job_id, scene_path, promote)
            except Exception as e:  # noqa: BLE001 - the worker must never die
                self._finish(job_id, {"state": "failed", "error": f"{type(e).__name__}: {e}"})
            finally:
                # Safety net: a job left non-terminal would poll forever.
                with self._lock:
                    stuck = self._jobs.get(job_id, {})
                    if stuck.get("state") not in self.TERMINAL_STATES:
                        stuck["state"] = "failed"
                        stuck.setdefault(
                            "error", "settle worker returned without reporting a result")

    def _promote(self, job_id, report):
        """Promote a settled result, if it passed and is still the newest.

        Returns a note explaining a refusal, or None when promotion happened.
        """
        if not report.get("ok"):
            return "settle reported objects beyond tolerance"
        settled = report.get("settled_path")
        if not settled:
            return "settle produced no output file"
        with self._lock:
            newest = self._newest
        if job_id != newest:
            return f"superseded by save {newest}"
        out = Path(settled)
        latest = latest_path(out)
        try:
            text = out.read_text(encoding="utf-8")
            document = json.loads(text)
        except (OSError, json.JSONDecodeError) as e:
            return f"settled output could not be read back: {e}"
        expect = UNCHECKED if self.expect_latest is None else self.expect_latest()
        try:
            digest = promote_scene_text(text, out, expect=expect)
        except TargetChanged as e:
            # Refuse rather than overwrite a file that changed underneath us.
            return str(e)
        except OSError as e:
            return f"could not write {latest.name}: {e}"
        if self.on_promote is not None:
            try:
                self.on_promote(latest, digest, document)
            except Exception as e:  # noqa: BLE001 - bookkeeping must not fail a promotion
                print(f"[settle] WARNING: could not record promotion of {latest.name}: {e}")
        print(f"[settle] job {job_id} promoted {out.name} -> {latest.name}")
        return None

    def _run(self, job_id, scene_path, promote):
        report_fd, report_path = tempfile.mkstemp(suffix=".json", prefix="settle_")
        os.close(report_fd)
        # Deliberately no --promote: see the class docstring.
        cmd = [
            str(self.python), str(self.script),
            "--scene", str(scene_path),
            "--steps", str(self.steps),
            "--tolerance", str(self.tolerance),
            "--report", report_path,
        ]

        print(f"[settle] job {job_id} starting: {scene_path.name}")
        try:
            proc = subprocess.run(
                cmd, cwd=str(HERE), capture_output=True, text=True, timeout=SETTLE_TIMEOUT_S
            )
            report = {}
            try:
                report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

            if not report:
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
                self._finish(job_id, {
                    "state": "failed",
                    "error": "settle produced no report",
                    "detail": "\n".join(tail),
                })
                print(f"[settle] job {job_id} failed: no report")
                return

            report["state"] = "failed" if report.get("error") else "done"
            report["generation"] = job_id
            report["promote_requested"] = bool(promote)
            report["promoted"] = False
            report["promotion_blocked"] = False
            if promote:
                refusal = self._promote(job_id, report)
                if refusal is None:
                    report["promoted"] = True
                else:
                    report["promotion_blocked"] = True
                    report["promotion_note"] = refusal
                    print(f"[settle] job {job_id} NOT promoted: {refusal}")
            self._finish(job_id, report)
            moved = report.get("moved", [])
            print(
                f"[settle] job {job_id} {report['state']}: "
                f"{len(moved)} object(s) beyond tolerance -> {report.get('settled_path')}"
            )
        except subprocess.TimeoutExpired:
            self._finish(job_id, {"state": "failed", "error": f"timed out after {SETTLE_TIMEOUT_S}s"})
            print(f"[settle] job {job_id} timed out")
        except Exception as e:  # noqa: BLE001 - a failed settle must not kill the server
            self._finish(job_id, {"state": "failed", "error": f"{type(e).__name__}: {e}"})
            print(f"[settle] job {job_id} failed: {e}")
        finally:
            try:
                os.unlink(report_path)
            except OSError:
                pass


def local_host_names():
    """Names and addresses this machine can legitimately be reached by.

    Used to validate Host headers on a non-loopback bind.
    """
    names = set()
    try:
        hostname = socket.gethostname()
        names.add(hostname.lower())
        names.add(socket.getfqdn(hostname).lower())
        names.update(addr.lower() for addr in socket.gethostbyname_ex(hostname)[2])
    except OSError:
        pass
    # A UDP connect consults the routing table without sending a packet,
    # finding addresses (e.g. a DHCP LAN one) that gethostbyname_ex misses.
    for probe in ("10.255.255.255", "192.168.255.255", "172.31.255.255"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(0.1)
                sock.connect((probe, 1))
                names.add(sock.getsockname()[0])
        except OSError:
            continue
    return frozenset(n for n in names if n)


def find_settle_python(repo_root, explicit=None):
    """Locate an interpreter whose ``omnigibson`` resolves inside this repo."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    else:
        env_python = os.environ.get("SIMFOUNDRY_PYTHON")
        if env_python:
            candidates.append(Path(env_python))
        conda_root = Path.home() / "miniforge3" / "envs"
        for name in ("simfoundry",):
            candidates.append(conda_root / name / "bin" / "python")

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            proc = subprocess.run(
                [str(candidate), "-c", "import omnigibson; print(omnigibson.__file__)"],
                capture_output=True, text=True, timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        path = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        if proc.returncode == 0 and path.startswith(str(repo_root)):
            return candidate
        if explicit:
            print(
                f"WARNING: {candidate} has omnigibson at {path or '<none>'}, "
                f"which is outside {repo_root}"
            )
    return None


# --- task-config round trip -------------------------------------------------

#: The two indentation shapes the checked-in task configs use. Both are dumped
#: and the one closer to the original is kept, so a save does not reflow files
#: authored in the other shape.
TASK_YAML_INDENTS = ((2, 4, 2), (2, 2, 0))


def task_yaml_round_tripper(indent=TASK_YAML_INDENTS[0]):
    """A ruamel YAML tuned to leave a task config the way it found it."""
    rt = YAML()
    rt.preserve_quotes = True
    rt.width = 4096
    mapping, sequence, offset = indent
    rt.indent(mapping=mapping, sequence=sequence, offset=offset)
    # Keep authored `foo: null` as `null`; ruamel would emit a bare `foo:`.
    rt.representer.add_representer(
        type(None),
        lambda representer, data: representer.represent_scalar(
            "tag:yaml.org,2002:null", "null"),
    )
    return rt


def load_task_yaml(text):
    """Parse a task config for editing, preserving comments where possible."""
    if YAML is None:
        return yaml.safe_load(text)
    return task_yaml_round_tripper().load(text)


def dump_task_yaml(document, original):
    """Serialize a task config, staying as close to *original* as possible.

    Args:
        document: What :func:`load_task_yaml` returned, after editing.
        original (str): The file's current text, used to pick the indentation.

    Returns:
        str: The document's YAML. Without ruamel this is a plain dump that
        keeps no comments -- callers must confirm the loss first.
    """
    if YAML is None:
        return yaml.safe_dump(document, sort_keys=False)
    candidates = [
        _dump_with_indent(document, indent) for indent in TASK_YAML_INDENTS
    ]
    # Ties keep the first, which is the shape most of these files use.
    return min(candidates, key=lambda text: line_churn(original, text))


def _dump_with_indent(document, indent):
    """One serialization at one indentation."""
    buffer = io.StringIO()
    task_yaml_round_tripper(indent).dump(document, buffer)
    return buffer.getvalue()


def line_churn(before, after):
    """How many lines a rewrite adds, drops or changes."""
    return sum(
        1
        for line in difflib.unified_diff(
            before.splitlines(), after.splitlines(), n=0, lineterm="")
        if line[:1] in "+-" and not line.startswith(("---", "+++"))
    )


def _flow_seq(values):
    """A range written inline, the way these configs author one.

    Uses ruamel containers only when ruamel parsed the document;
    ``yaml.safe_dump`` cannot represent a ruamel node.
    """
    if YAML is None:
        return list(values)
    seq = CommentedSeq(values)
    seq.fa.set_flow_style()
    return seq


def _unchanged_number(existing, value):
    """Whether *existing* already holds this number, whatever type it is."""
    if isinstance(existing, bool) or not isinstance(existing, (int, float)):
        return False
    return float(existing) == float(value)


def _set_range(container, group, value):
    """Write one group's range, replacing only numbers that actually differ.

    Lists are edited in place to keep their flow style; rewriting an unchanged
    value would still churn the file (ruamel tracks the authored scalar type).
    """
    existing = container.get(group)
    if not isinstance(value, list):
        if not _unchanged_number(existing, value):
            container[group] = value
        return
    if isinstance(existing, list) and len(existing) == len(value):
        for index, number in enumerate(value):
            if not _unchanged_number(existing[index], number):
                existing[index] = number
        return
    container[group] = _flow_seq(value)


def apply_group_ranges(og_cfg, key, values):
    """Set or clear one randomization entry per group, in place.

    Args:
        og_cfg (dict): The ``og_task_config`` block. Mutated.
        key (str): ``group_xyz_randomization`` or ``group_z_rot_randomization``.
        values (dict): ``{group: value or None}``, None meaning "no entry". A
            group *absent* from this map is left exactly as the file has it.

    The parsed map is mutated rather than rebound: ruamel hangs the file's
    comments and formatting off the objects it parsed.
    """
    was_present = key in og_cfg
    current = og_cfg.get(key)
    if not isinstance(current, dict):
        current = None
    for group, value in values.items():
        if value is None:
            if current is not None:
                current.pop(group, None)
            continue
        if current is None:
            current = {} if YAML is None else CommentedMap()
            og_cfg[key] = current
        _set_range(current, group, value)
    if current is not None and not current:
        # An emptied map becomes `null` (the authored spelling for "none");
        # a key the file never carried stays absent.
        if was_present:
            og_cfg[key] = None
        else:
            og_cfg.pop(key, None)


class EditorHandler(SimpleHTTPRequestHandler):
    """Static file server for web/, plus a save endpoint."""

    scene_json_path = None
    # The compilation origin: the immutable document every save recompiles
    # from. Not the same thing as `current_scene`, which tracks what the
    # scene says now.
    base_scene = None
    # Digest of the file `base_scene` was read from. Identity, not currency:
    # the browser echoes it back to prove it loaded this origin. A promotion
    # moves `expected_source_sha256`, not this.
    base_scene_sha256 = None
    # What the scene says now: the document the most recent accepted mutation
    # produced, or the origin when there has been none. Served to reloads and
    # `/api/scene_state`, and named by `scene_revision`.
    current_scene = None
    # The digest `scene_json_path` is expected to hold right now. Advanced by
    # promotion, the only legitimate way the source file changes while an
    # editor is open.
    expected_source_sha256 = None
    # The same, for `_scene_state_latest.json`. None means "no such file",
    # which tells a first promotion apart from an overwrite.
    expected_latest_sha256 = None
    # Expected digests of the camera configs this session writes, keyed by
    # resolved path. A path in here is one this session wrote or was bound to;
    # any other target gets the existence check in `_save_cameras`.
    camera_digests = {}
    # Bumped on every accepted mutation, under WRITE_LOCK and together with
    # `current_scene`, so a stale tab is refused rather than silently
    # reverting newer work with its own complete snapshot.
    scene_revision = 0
    # Task configs bind scene objects by name or category. Parsed at startup.
    task_configs = ()
    known_scene_names = frozenset()
    editable_names = set()
    robot_names = set()
    background_names = set()
    # Rooms a pending swap has displaced. The only names a background may be
    # removed under: see `_set_background`.
    background_removed = set()
    # Same, for a robot a room's default_robot has displaced.
    robot_removed = set()

    @property
    def posable_names(self):
        """Every editable object, plus the robot and the room.

        Position and orientation are writable for all three (see
        `iter_objects` for the full split). Derived because `editable_names`
        grows on every import.
        """
        return self.editable_names | self.robot_names | self.background_names

    @property
    def scalable_names(self):
        """Every editable object, plus the room -- but never the robot.

        A scanned room is plain geometry and resizes cleanly; a robot carries
        URDF-derived joint frames, collision geometry and actuator limits that
        a mesh scale would leave behind.
        """
        return self.editable_names | self.background_names

    settle_runner = None
    scene_name = ""
    cameras = None
    camera_document = None
    camera_source_path = None
    # The rig config --cameras named. Distinct from camera_source_path,
    # which is the *resumed* placement when one exists.
    camera_template_path = None
    camera_out_path = None
    # Per-process secret required on every mutation; see _read_json_request.
    mutation_token = ""
    bind_host = "127.0.0.1"
    extra_hosts = frozenset()
    # Bumped on every accepted camera write so a stale client is told to reload
    # rather than silently overwriting a placement it never saw.
    camera_revision = 0
    # The scanned room this scene is in (registry row) and its table centre.
    # Both None for a scene with no background.
    background_row = None
    table_estimate = None
    # Where the file picker opens, and the shortcuts it offers.
    browse_default = Path.home()
    browse_shortcuts = ()
    camera_background = None
    camera_resumed = False
    # Which policy observation key each camera fills, worked out from the task
    # configs. Resolved per scene, since the rig is keyed by the room.
    camera_observation = None
    # Cameras that belong to the robot asset rather than to a rig config, e.g.
    # the wrist camera the policy receives as `wrist_image_left`. Read-only
    # and kept out of `cameras`, which is what a save writes back to YAML.
    robot_cameras = ()
    robot_camera_observation = {}

    # --- the scene launcher -------------------------------------------------
    # Session state that outlives a scene switch; everything a switch replaces
    # lives in bind_scene().
    options = None
    # Directories the catalog searches, and the only ones a scene may be
    # opened from -- otherwise "open a scene" would read any JSON on this
    # machine.
    scene_roots = ()
    recents = None
    compose_root = None
    # Bumped on every scene switch, never reset, so a stale tab can tell the
    # scene changed under it.
    session_revision = 0

    # --- imported objects ---------------------------------------------------
    # Objects added this session, keyed by name. Pending: the source scene on
    # disk is never modified; every save recompiles base scene, plus these,
    # minus client removals, plus edits.
    pending_adds = {}
    # Which pending adds a save has written. Saves recompile from the startup
    # document, so pending_adds is not cleared by one; unsaved work is
    # pending_adds minus saved_adds.
    saved_adds = set()
    # The served manifest, kept in memory so an import can append to it.
    manifest = None
    # Where it and the extracted proxies live. Declared here because several
    # methods read it before a scene is bound.
    data_dir = None
    asset_roots = ()
    _assets_cache = None
    _added_counter = 0
    textures = True

    def do_GET(self):
        parts = urlsplit(self.path)
        if parts.path == "/api/session":
            # Also the browser's heartbeat.
            self._json(200, {
                "scene_revision": self.scene_revision,
                "camera_revision": self.camera_revision,
                "editable": sorted(self.editable_names),
                "pending_adds": sorted(self.pending_adds),
                "settle_enabled": bool(self.settle_runner and self.settle_runner.enabled),
                "scene": str(self.scene_json_path),
                # Changes when the editor is pointed at another scene; a tab
                # that missed the switch has to reload.
                "session_revision": self.session_revision,
                "scene_name": self.scene_name,
            })
            return
        if parts.path == "/api/scene_state":
            self._json(200, self.scene_state())
            return
        if parts.path == "/api/scenes":
            self._json(200, self.scene_catalog())
            return
        if parts.path == "/api/scenes/recent":
            # Lighter cousin of /api/scenes: renders on every page load, so it
            # skips the full scene-root walk.
            self._json(200, self.recent_catalog())
            return
        if parts.path == "/api/tasks":
            # Task lookup for an arbitrary scene, not just the loaded one:
            # recent-scene rows ask before the scene is ever opened.
            query = parse_qs(parts.query)
            raw = (query.get("scene") or [None])[0]
            try:
                # Recents-wide, not roots-wide: see scene_lookup_roots.
                scene_path = resolve_scene_path(raw, self.scene_lookup_roots())
            except SceneEditError as e:
                self._json(400, {"error": str(e)})
                return
            try:
                scene_data = load_scene(scene_path)
            except (OSError, ValueError) as e:
                self._json(400, {"error": f"could not read {scene_path}: {e}"})
                return
            name = scene_path.stem.split("_scene_state_")[0]
            self._json(200, {"tasks": self._tasks_for(scene_data, name)})
            return
        if parts.path == "/api/backgrounds":
            # Registered rooms, and which one a given scene is laid out in;
            # answers for scenes that are not the loaded one.
            query = parse_qs(parts.query)
            raw = (query.get("scene") or [None])[0]
            current = None
            scene_path = None
            if raw:
                try:
                    scene_path = resolve_scene_path(raw, self.scene_lookup_roots())
                    current = background_id(load_scene(scene_path), scene_path)
                except (SceneEditError, OSError, ValueError) as e:
                    # Not fatal: an unreadable scene still gets the room list.
                    print(f"[backgrounds] {raw}: {type(e).__name__}: {e}")
            self._json(200, {
                "current": current,
                "rooms": [
                    {"id": room["id"], "label": room["label"],
                     "surface_height": room["surface_height"]}
                    for room in self._known_rooms()
                ],
            })
            return
        if parts.path == "/api/task_cfg":
            query = parse_qs(parts.query)
            raw = (query.get("path") or [None])[0]
            try:
                path = self._resolve_task_cfg_path(raw)
            except SceneEditError as e:
                self._json(400, {"error": str(e)})
                return
            try:
                raw_bytes = path.read_bytes()
                document = yaml.safe_load(raw_bytes.decode("utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
                self._json(400, {"error": f"could not read {path}: {e}"})
                return
            og_cfg = document.get("og_task_config") if isinstance(document, dict) else None
            if not isinstance(og_cfg, dict):
                self._json(400, {"error": f"{path.name} has no og_task_config"})
                return
            self._json(200, {
                "path": str(path),
                # The Hydra group that selects this file, and its instruction,
                # so Review & Export can default to the config this panel edits.
                "group": export_bundle.task_group_name(path, HERE.parents[2]),
                "task_name": document.get("task_name"),
                "instruction": document.get("language_instruction"),
                # Digest of the bytes this answer describes; a save quotes it
                # back so an edit cannot land on a file rewritten since.
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                # Whether a save will keep this file's comments.
                "round_trip": YAML is not None,
                "groups": self._task_cfg_groups(path, og_cfg),
                "workspace_bounds": og_cfg.get("workspace_bounds"),
            })
            return
        if parts.path == "/api/library":
            self._json(200, {
                "assets": self.list_assets(),
                "roots": [str(r) for r in self.asset_roots],
            })
            return
        if parts.path == "/api/table":
            self._json(200, self.table_state())
            return
        if parts.path == "/api/ground_plane":
            self._json(200, self.ground_plane_state())
            return
        if parts.path == "/api/camera_configs":
            self._json(200, self.camera_config_catalog())
            return
        if parts.path == "/api/cameras":
            if not self.cameras:
                self._json(200, {"cameras": [], "source": None})
                return
            self._json(200, {
                # Robot cameras ride along for display; `validate_camera_edits`
                # refuses them by name on write-back.
                "cameras": list(self.cameras) + list(self.robot_cameras),
                "source": str(self.camera_source_path),
                "out_path": str(self.camera_out_path),
                "cfg_name": Path(self.camera_out_path).stem,
                # Poses are relative to this object's frame; the browser
                # parents the cameras to it.
                "parent_object": self.camera_parent_name(),
                "parent_link": self.camera_parent_link(),
                # What the placement is remembered under, and whether these
                # poses are one somebody already authored or the rig template.
                "background": self.camera_background,
                "resumed": self.camera_resumed,
                # Which policy input each camera lands in, read from the task
                # configs; the rig config does not say.
                "observation": self.merged_observation(),
                # Echoed back on save so a client that loaded an older
                # placement is refused rather than silently overwriting.
                "camera_revision": self.camera_revision,
            })
            return
        if parts.path == "/api/settle":
            job_ids = parse_qs(parts.query).get("id", [])
            try:
                job_id = int(job_ids[0])
            except (IndexError, ValueError):
                self._json(400, {"error": "id query parameter required"})
                return
            status = self.settle_runner.status(job_id) if self.settle_runner else None
            if status is None:
                self._json(404, {"error": "unknown settle job"})
                return
            self._json(200, status)
            return
        super().do_GET()

    data_dir = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE / "web"), **kwargs)

    def translate_path(self, path):
        """Serve /data/ from this server's own extraction directory.

        Extractions are per-scene, so two servers cannot serve each other's
        geometry.
        """
        split = urlsplit(path).path
        if self.data_dir is not None and (split == "/data" or split.startswith("/data/")):
            relative = split[len("/data"):].lstrip("/")
            # Reuse the base implementation's traversal defences, then re-root.
            safe = super().translate_path("/" + relative)
            base = str(Path(super().translate_path("/")).resolve())
            resolved = str(Path(safe).resolve())
            if not resolved.startswith(base):
                return safe
            return str(Path(self.data_dir) / Path(resolved).relative_to(base))
        return super().translate_path(path)

    def send_head(self):
        """Serve index.html with the mutation token injected.

        The token rides in the HTML so no other origin can fetch it by script.
        """
        split = urlsplit(self.path).path
        if split in ("/", "/index.html"):
            try:
                html = (HERE / "web" / "index.html").read_text(encoding="utf-8")
            except OSError:
                return super().send_head()
            tag = f'<meta name="editor-token" content="{self.mutation_token}">'
            html = html.replace("</head>", f"  {tag}\n</head>", 1)
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # A cached copy would pin a token from a previous process.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return io.BytesIO(body)
        return super().send_head()

    def log_message(self, fmt, *args):
        # Quiet the per-asset request spam; keep anything that is not a 2xx.
        code = str(args[1]) if len(args) > 1 else ""
        if not code.startswith("2"):
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        # Extraction output is per scene but its URLs are not, so a cached
        # /data/ copy can belong to another scene; editor source gets the same
        # treatment so a reload always picks up current code. Everything here
        # is local disk on localhost -- nothing worth caching.
        path = urlsplit(self.path).path
        if path.startswith("/data/") or path.endswith((".js", ".css", ".html")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @classmethod
    def _camera_prim_segments(cls):
        """Path segments of the first camera's parent prim, or []."""
        for camera in cls.cameras or []:
            path = str(camera.get("relative_prim_path") or "")
            segments = [s for s in path.split("/") if s]
            if len(segments) >= 2:
                return segments
        return []

    @classmethod
    def camera_parent_name(cls):
        """Scene object the cameras hang off, e.g. 'robot0'.

        OmniGibson names a controllable prim ``controllable__<class>__<name>``,
        so the scene-object name is the trailing segment.
        """
        segments = cls._camera_prim_segments()
        if not segments:
            return None
        candidate = segments[0].split("__")[-1]
        return candidate if candidate in (cls.base_scene or {}).get(
            "objects_info", {}
        ).get("init_info", {}) else None

    @classmethod
    def camera_parent_link(cls):
        """Link the poses are relative to, e.g. 'panda_link0'."""
        segments = cls._camera_prim_segments()
        return segments[1] if len(segments) >= 2 else None

    @classmethod
    def merged_observation(cls):
        """Which policy input every camera fills, rig config and robot alike.

        Merges the task-config map with the robot asset's own cameras (e.g.
        ``wrist_image_left``).

        Returns:
            dict or None: ``{"cameras": {...}, "note": str}``, or None when no
                camera config is loaded.
        """
        if not cls.robot_camera_observation:
            return cls.camera_observation
        base = cls.camera_observation or {"cameras": {}, "note": ""}
        return {**base, "cameras": {**base.get("cameras", {}),
                                    **cls.robot_camera_observation}}

    def _host_allowed(self):
        """Reject Host headers this server was not reached under.

        Origin==Host alone is not enough: an attacker-controlled hostname can
        point at loopback and make both agree.
        """
        host = self.headers.get("Host", "")
        name = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
        name = name.strip("[]").lower() if name.startswith("[") else name.lower()
        if self.bind_host in ("127.0.0.1", "localhost", ""):
            return name in LOOPBACK_HOSTS
        # A deliberate non-loopback bind also answers to its own address and to
        # whatever name the operator allowed.
        return name in LOOPBACK_HOSTS | {self.bind_host.lower()} | self.extra_hosts

    def _read_json_request(self):
        """Apply the shared write-endpoint guards and return the parsed body.

        Returns None after having already sent an error response.
        """
        if not self._host_allowed():
            self._json(403, {"error": "unrecognised Host header"})
            return None

        # A page from another origin must not be able to write through this
        # loopback service.
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin:
            try:
                same_origin = urlsplit(origin).netloc == host
            except ValueError:
                same_origin = False
            if not same_origin:
                self._json(403, {"error": "cross-origin save rejected"})
                return None

        # The token is injected into index.html, which no other origin can
        # read; requiring it covers the case where Origin and Host agree.
        supplied = self.headers.get("X-Editor-Token", "")
        if not secrets.compare_digest(supplied, self.mutation_token or ""):
            self._json(403, {"error": "missing or invalid editor token; reload the page"})
            return None

        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Content-Type must be application/json"})
            return None

        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("bad content length")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            return payload
        except Exception as e:
            self._json(400, {"error": f"bad request: {e}"})
            return None

    @classmethod
    def list_assets(cls):
        """Discovered assets, walked once and cached for the process."""
        if cls._assets_cache is None:
            cls._assets_cache = discover_assets(
                cls.asset_roots, Path(cls.scene_json_path).parent
            )
        return cls._assets_cache

    @classmethod
    def _scene_hash_accepted(cls, digest):
        """Was this client's manifest built from the origin we compile against?

        Identity only. File currency is `expected_source_sha256`, checked in
        `_compile_scene`; promotion moves the file but not the origin.
        """
        return digest == cls.base_scene_sha256

    @classmethod
    def adopt_written_scene(cls, document, *, promoted_digest=None,
                            promoted_path=None, announce=print):
        """Take a document this session just wrote as the current scene state.

        Call under WRITE_LOCK. Moves `current_scene` and bumps its revision,
        advances the file digests when the write promoted, and rewrites the
        served manifest so a reload shows the written poses.

        Args:
            document (dict): The scene document that was written.
            promoted_digest (str or None): The digest now in ``_latest``, when
                this write promoted. None means nothing on disk moved.
            promoted_path (Path or None): Where it was promoted to.
            announce (callable): Where a manifest-refresh failure is reported.

        Returns:
            int: The new revision.
        """
        cls.current_scene = copy.deepcopy(document)
        if promoted_digest is not None:
            cls.expected_latest_sha256 = promoted_digest
            if (promoted_path is not None and cls.scene_json_path is not None
                    and Path(promoted_path).resolve()
                    == Path(cls.scene_json_path).resolve()):
                cls.expected_source_sha256 = promoted_digest
        revision = cls.bump_revision()
        cls.refresh_served_manifest(announce=announce)
        return revision

    @classmethod
    def scene_state(cls):
        """What the scene currently says, and the revision that says it.

        The revision and the content are read together under the write lock;
        a client must apply both or neither.

        Returns:
            dict: ``scene_revision``, ``session_revision``, ``ground_plane``
            and one ``objects`` entry per posable name -- including objects
            the scene no longer has, marked ``present: false``.
        """
        with WRITE_LOCK:
            scene = cls.current_scene or cls.base_scene or {}
            origin = cls.base_scene or {}
            revision, session = cls.scene_revision, cls.session_revision
            manifest_entries = {
                entry["name"]: entry
                for entry in ((cls.manifest or {}).get("objects") or [])
            }
            current_init = scene.get("objects_info", {}).get("init_info", {}) or {}
            current_registry = (scene.get("state", {}).get("registry", {})
                                .get("object_registry", {}) or {})
            origin_init = origin.get("objects_info", {}).get("init_info", {}) or {}
            origin_registry = (origin.get("state", {}).get("registry", {})
                               .get("object_registry", {}) or {})

            objects = {}
            # The instance property `posable_names`, spelled out for a classmethod.
            for name in (cls.editable_names | cls.robot_names | cls.background_names):
                entry = manifest_entries.get(name) or {}
                physics = entry.get("physics") or {}
                joints = entry.get("joints") or {}
                present = name in current_init
                init = current_init if present else origin_init
                registry = current_registry if present else origin_registry
                state = authored_state(
                    (init.get(name) or {}).get("args") or {},
                    registry.get(name) or {},
                    link=physics.get("link"), joints=joints.get("joints"),
                )
                objects[name] = {**state, "present": present}

            return {
                "scene_revision": revision,
                "session_revision": session,
                "scene": str(cls.scene_json_path),
                "ground_plane": read_ground_plane(scene),
                "objects": objects,
                "pending_adds": sorted(cls.pending_adds),
            }

    @classmethod
    def adopt_scene_binding(cls, scene_json, base_scene):
        """Point the class at one scene document and reset everything derived.

        Sets the compilation origin, its identity digest, the current snapshot
        and its revision, and the two file digests every publish compares
        against -- all of which must move together.
        """
        scene_json = Path(scene_json)
        cls.scene_json_path = scene_json
        cls.base_scene = base_scene
        cls.base_scene_sha256 = scene_sha256(scene_json)
        cls.current_scene = copy.deepcopy(base_scene)
        cls.expected_source_sha256 = cls.base_scene_sha256
        cls.expected_latest_sha256 = file_digest(latest_path(scene_json))
        cls.scene_revision = 0

    @classmethod
    def note_settle_promotion(cls, promoted_path, digest, document):
        """A settle job promoted its output; catch the session up.

        Runs on the settle worker thread, so it takes the write lock like any
        request handler.
        """
        with WRITE_LOCK:
            cls.adopt_written_scene(document, promoted_digest=digest,
                                    promoted_path=promoted_path)

    @classmethod
    def refresh_served_manifest(cls, announce=print):
        """Rewrite ``data/manifest.json`` to describe `current_scene`.

        Re-reads only the authored half -- poses, scales, mass, friction,
        joints, and which objects the scene has; a save cannot change a mesh.
        """
        manifest, scene = cls.manifest, cls.current_scene
        if not manifest or not scene or not cls.data_dir:
            return
        try:
            apply_authored_state_to_manifest(manifest, scene)
            if cls.scene_json_path is not None:
                # The cache digest: which file revision this extraction is good
                # for, so --no-extract survives a promotion.
                manifest["source_scene_sha256"] = cls.expected_source_sha256
            from extract import write_manifest  # noqa: PLC0415 - pulls in pxr
            write_manifest(manifest, cls.data_dir)
        except Exception as e:  # noqa: BLE001 - a served page beats a dead one
            announce(f"[manifest] WARNING: could not refresh: {type(e).__name__}: {e}")

    @classmethod
    def bump_revision(cls):
        """Invalidate other clients' snapshots after an accepted mutation."""
        cls.scene_revision += 1
        return cls.scene_revision

    @classmethod
    def refresh_task_configs(cls):
        """Re-read the task configs from disk after this editor writes one."""
        try:
            # Written to EditorHandler by name so a subclass does not take a
            # shadowing copy.
            EditorHandler.task_configs = tuple(
                task_bindings.discover_task_configs(HERE.parents[2]))
        except Exception as e:  # noqa: BLE001 - a stale list beats a failed write
            print(f"[tasks] WARNING: rediscovery failed: {type(e).__name__}: {e}")

    @classmethod
    def asset_facts(cls):
        """Link and joint names per object, taken off the served manifest.

        The only admissible source for a link or joint name in an edit: a name
        echoed by the browser proves nothing, and a link OmniGibson cannot
        find raises on load.
        """
        facts = {}
        for entry in (cls.manifest or {}).get("objects", []):
            physics = entry.get("physics") or {}
            joints = entry.get("joints") or {}
            facts[entry.get("name")] = {
                "links": list(physics.get("links") or []),
                "joints": list(joints.get("joints") or []),
            }
        return facts

    @staticmethod
    def wire_warning(warning):
        """One warning in the shape the browser reads: ``code``/``severity``/``message``.

        ``code`` is `check_bindings`' own ``kind``, renamed at the wire.
        ``where`` points into the YAML (``goal_predicates_all[0].state``);
        ``path`` is the file the config lives in.
        """
        payload = {k: warning.get(k) for k in
                   ("task", "group", "keys", "severity", "effect", "message",
                    "detail", "path", "where")}
        payload["code"] = warning.get("kind")
        return payload

    @classmethod
    def binding_warnings(cls, removed=(), added=None):
        """Task bindings a proposed edit would leave unsatisfied.

        Warnings only, never a refusal: a task config is a text file anyone
        can fix afterwards.
        """
        if not cls.task_configs:
            return []
        try:
            return task_bindings.check_bindings(
                cls.base_scene, list(cls.task_configs),
                removed=removed, added=added,
                scene_name=cls.scene_name,
                known_scenes=set(cls.known_scene_names),
            )
        except Exception as e:  # noqa: BLE001 - a warning path must not fail a save
            print(f"[tasks] WARNING: binding check failed: {type(e).__name__}: {e}")
            return []

    @classmethod
    def session_categories(cls):
        """``{object name: category}`` for the scene as this session holds it.

        `base_scene` plus the pending imports, so binding readouts see objects
        imported this session. A spec without a category falls back to the
        default so the object stays matchable by name.
        """
        return task_bindings.apply_edit(
            task_bindings.scene_object_categories(cls.base_scene or {}),
            added={name: spec.get("category") or task_bindings.DEFAULT_CATEGORY
                   for name, spec in cls.pending_adds.items()},
        )

    @classmethod
    def taken_names(cls):
        """Every object name that would collide, including pending imports."""
        base = cls.base_scene or {}
        return (
            set(base.get("objects_info", {}).get("init_info", {}))
            | set(base.get("state", {}).get("registry", {}).get("object_registry", {}))
            | set(cls.pending_adds)
        )

    def _set_background(self):
        """Attach a scanned room to this session, replacing any it already has.

        Registered as a *pending* add, like an imported prop: nothing here
        touches the scene file. The room being replaced is the one case where
        a background may be removed at all -- `remove_objects` refuses a bare
        removal, so the permission is granted here, per-name, once a
        replacement is in hand.
        """
        payload = self._read_json_request()
        if payload is None:
            return
        wanted = payload.get("id")
        if not isinstance(wanted, str) or not wanted.strip():
            self._json(400, {"error": "which room? send an id from /api/backgrounds"})
            return

        with WRITE_LOCK:
            try:
                roots = background_roots(self.scene_json_path, self.options.repo_root)
                room = resolve_background(wanted.strip(), roots)
                EditorHandler.background_row = room
                EditorHandler.table_estimate = None
                # Camera rigs are keyed by room id; re-resolve so a save/import
                # now targets this room's placement file, not the startup one.
                for attribute, value in _resolve_cameras(
                        self.scene_name, room["id"], self.options, lambda *_: None).items():
                    setattr(EditorHandler, attribute, value)
            except SceneEditError as e:
                self._json(400, {"error": str(e)})
                return

            # Two kinds of room to displace, collected separately.
            replaced = sorted(background_object_names(self.base_scene, self.scene_json_path))
            dropped = sorted(
                n for n, spec in self.pending_adds.items()
                if spec.get("category") == "mesh_background"
            )
            # A replacement gets a fresh name: a swap that reused the displaced
            # name would cancel itself out in `_compile_scene` and write
            # neither room. Nothing keys off the name.
            name = BACKGROUND_OBJECT_NAME
            if name in self.taken_names():
                name = unique_object_name(self.taken_names(), "mesh_background")

            try:
                from extract import build_proxy, write_manifest

                spec = background_spec(room, self.scene_json_path, name=name)
                EditorHandler._added_counter += 1
                glb_name = f"added_{EditorHandler._added_counter:04d}.glb"
                proxy = build_proxy(room["usd"], self.data_dir, glb_name, self.textures)
                if proxy["glb"] is None and proxy.get("splat") is None:
                    self._json(422, {"error": f"{room['id']}: {proxy['error']}"})
                    return
            except SceneEditError as e:
                self._json(400, {"error": str(e)})
                return
            except Exception as e:  # noqa: BLE001 - a bad room must not kill the server
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return

            entry = {
                "name": name,
                "category": "mesh_background",
                "kind": "background",
                # Permissions by hand -- this entry never goes through
                # iter_objects. A room is posable and scalable, never editable.
                "editable": False,
                "posable": True,
                "scalable": True,
                "physics": None,
                "joints": None,
                "position": spec["registry"]["root_link"]["pos"],
                "orientation": spec["registry"]["root_link"]["ori"],
                "scale": [1.0, 1.0, 1.0],
                "sourceUsd": str(room["usd"]),
                "added": True,
                "assetId": room["id"],
                "backgroundId": room["id"],
                **proxy,
            }

            # A superseded pending room is withdrawn, not removed: it never
            # reached the file.
            for stale in dropped:
                EditorHandler.pending_adds.pop(stale, None)
            EditorHandler.pending_adds[name] = {
                "name": name,
                "init_info": spec["init_info"],
                "registry": spec["registry"],
                "category": "mesh_background",
            }
            # Not `editable_names`: an added room accepts a pose and a scale,
            # nothing else.
            EditorHandler.background_names = (
                (set(self.background_names) - set(dropped)) | {name})
            EditorHandler.background_removed = set(self.background_removed) | set(replaced)
            if self.manifest is not None:
                self.manifest["objects"] = [
                    e for e in self.manifest["objects"] if e.get("name") not in dropped
                ]
                self.manifest["objects"].append(entry)
                write_manifest(self.manifest, self.data_dir)

            # A room can prescribe its robot(s) (e.g. the YAM workstation wants
            # two Yams, not whatever single Franka the scene opened with).
            # Same swap pattern as the room itself: fresh names, old ones
            # displaced -- built all-or-nothing so a bimanual room never ends
            # up with only one arm swapped in.
            robot_rows = default_robot(room)
            robot_entries = robot_replaced = robot_dropped = None
            if robot_rows:
                robot_replaced = sorted(robot_object_names(self.base_scene, self.scene_json_path))
                robot_dropped = sorted(
                    n for n, spec in self.pending_adds.items() if spec.get("category") == "robot"
                )
                built = []
                taken = set(self.taken_names())
                try:
                    for row in robot_rows:
                        robot_name = unique_object_name(taken, "robot")
                        taken.add(robot_name)
                        robot_init = copy.deepcopy(row["init_info"])
                        robot_init.setdefault("args", {})["name"] = robot_name
                        robot_registry = copy.deepcopy(row["registry"])
                        robot_usd = resolve_robot_usd(robot_init, self.options.robot_asset_root)
                        if robot_usd is None:
                            raise SceneEditError(f"no asset mapped for {robot_init.get('class_name')}")
                        EditorHandler._added_counter += 1
                        robot_glb = f"added_{EditorHandler._added_counter:04d}.glb"
                        # Robots are context, not editing subjects; skip their
                        # textures, matching the startup extraction (extract.py).
                        robot_proxy = build_proxy(robot_usd, self.data_dir, robot_glb, False)
                        if robot_proxy["glb"] is None:
                            raise SceneEditError(robot_proxy["error"])
                        built.append((robot_name, robot_init, robot_registry, robot_usd, robot_proxy))
                except Exception as e:  # noqa: BLE001 - a bad robot spec must not sink the room swap
                    print(f"[background] robot swap skipped: {type(e).__name__}: {e}")
                    robot_replaced = robot_dropped = None
                else:
                    for stale in robot_dropped:
                        EditorHandler.pending_adds.pop(stale, None)
                    robot_entries = []
                    new_names = set()
                    for robot_name, robot_init, robot_registry, robot_usd, robot_proxy in built:
                        new_names.add(robot_name)
                        EditorHandler.pending_adds[robot_name] = {
                            "name": robot_name, "init_info": robot_init,
                            "registry": robot_registry, "category": "robot",
                        }
                        robot_entries.append({
                            "name": robot_name,
                            "category": robot_init.get("class_name", "robot").lower(),
                            "kind": "robot",
                            "editable": False, "posable": True, "scalable": False,
                            "physics": None, "joints": None,
                            "position": robot_registry["root_link"]["pos"],
                            "orientation": robot_registry["root_link"]["ori"],
                            "scale": [1.0, 1.0, 1.0],
                            "sourceUsd": str(robot_usd), "added": True,
                            "assetId": robot_init.get("class_name"),
                            **robot_proxy,
                        })
                    EditorHandler.robot_names = (
                        (set(self.robot_names) - set(robot_dropped)) | new_names)
                    EditorHandler.robot_removed = set(self.robot_removed) | set(robot_replaced)
                    # The wrist/onboard camera list and observation-key map are
                    # read off the robot's own USD; re-derive them for the new
                    # robot(s) so they don't keep describing the displaced one.
                    robot_scene = copy.deepcopy(self.base_scene)
                    remove_objects(robot_scene, robot_replaced, removable_names=set(robot_replaced))
                    add_objects(robot_scene, [EditorHandler.pending_adds[n] for n in new_names])
                    try:
                        robot_cams, robot_obs = robot_cameras.scene_robot_cameras(
                            robot_scene, self.scene_json_path, self.options.robot_asset_root,
                            lambda *_: None)
                    except Exception as e:  # noqa: BLE001 - the wrist-camera preview is optional
                        print(f"[background] robot camera resolve skipped: {type(e).__name__}: {e}")
                        robot_cams, robot_obs = [], {}
                    EditorHandler.robot_cameras = tuple(robot_cams)
                    EditorHandler.robot_camera_observation = robot_obs
                    if self.manifest is not None:
                        self.manifest["objects"] = [
                            e for e in self.manifest["objects"] if e.get("name") not in robot_dropped
                        ]
                        self.manifest["objects"].extend(robot_entries)
                        write_manifest(self.manifest, self.data_dir)

        print(f"[background] {name} <- {room['id']}"
              + (f" (replacing {', '.join(replaced)})" if replaced else "")
              + (f" (withdrawing {', '.join(dropped)})" if dropped else ""))
        response = {"entry": entry, "replaced": replaced, "dropped": dropped,
                    "room": room["id"], "label": room["label"]}
        if robot_entries:
            response.update({"robot_entries": robot_entries, "robot_replaced": robot_replaced,
                              "robot_dropped": robot_dropped})
        self._json(200, response)

    def _register_import(self, usd_relative, usd_absolute, category, asset_id,
                         transform, extra=None):
        """Turn an imported asset into a pending object, and describe it.

        Shared by both import routes -- the library and an arbitrary path.

        Returns the manifest entry, or None after having sent an error response.
        """
        try:
            name = unique_object_name(self.taken_names(), category)

            from extract import build_proxy, write_manifest

            EditorHandler._added_counter += 1
            glb_name = f"added_{EditorHandler._added_counter:04d}.glb"
            proxy = build_proxy(usd_absolute, self.data_dir, glb_name, self.textures)
            if proxy["glb"] is None:
                # Refuse rather than add an invisible object.
                self._json(422, {"error": f"{asset_id}: {proxy['error']}"})
                return None

            spec = object_spec(
                name, usd_relative, usd_absolute, category,
                position=transform["position"],
                orientation=transform["orientation"],
                scale=transform["scale"],
            )
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return None
        except Exception as e:  # noqa: BLE001 - a bad asset must not kill the server
            self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return None

        entry = {
            "name": name,
            "category": category,
            "kind": "object",
            "editable": True,
            # Permissions stated by hand: this entry never goes through
            # iter_objects, and the browser treats absent flags as false.
            "posable": True,
            "scalable": True,
            # Physics and joint facts read from the USD so the panels work
            # before the first save; a fresh import starts at the asset's own
            # zero joint configuration.
            "physics": physics_facts(spec["init_info"]["args"], usd_absolute),
            "joints": joint_facts(spec["init_info"]["args"], usd_absolute,
                                  spec["registry"]),
            "position": spec["registry"]["root_link"]["pos"],
            "orientation": spec["registry"]["root_link"]["ori"],
            "scale": spec["init_info"]["args"]["scale"],
            "sourceUsd": str(usd_absolute),
            "added": True,
            "assetId": asset_id,
            **proxy,
            **(extra or {}),
        }

        EditorHandler.pending_adds[name] = spec
        EditorHandler.editable_names = set(self.editable_names) | {name}
        if self.manifest is not None:
            self.manifest["objects"].append(entry)
            write_manifest(self.manifest, self.data_dir)

        print(f"[add] {name} <- {usd_relative}  ({proxy['verts']} verts, "
              f"{'textured' if proxy['textured'] else 'flat'})")
        return entry

    def _requested_transform(self, payload):
        """Validate the pose an import asked to be placed at.

        Checked before any asset is copied or converted: a rejected import
        should leave nothing on disk.
        """
        return {
            "position": _checked_vector(
                "position", payload.get("position", [0.0, 0.0, 0.0]), 3),
            "orientation": _checked_vector(
                "orientation", payload.get("orientation", [0.0, 0.0, 0.0, 1.0]), 4),
            "scale": _checked_vector(
                "scale", payload.get("scale", [1.0, 1.0, 1.0]), 3, positive=True),
        }

    def _add_object(self):
        """Import a library asset into the session and return its manifest entry.

        The asset is copied into the scene directory and converted to a glTF
        proxy synchronously; the scene JSON is untouched until a save compiles
        the addition in.
        """
        payload = self._read_json_request()
        if payload is None:
            return

        key = payload.get("key")
        if not isinstance(key, str):
            self._json(400, {"error": "key must be a string"})
            return

        with WRITE_LOCK:
            # The browser only ever sends a key this server listed, so an import
            # cannot be talked into reading a path of the caller's choosing.
            asset = resolve_key(key, self.list_assets())
            if asset is None:
                self._json(404, {"error": "unknown asset key; reload the library"})
                return
            try:
                transform = self._requested_transform(payload)
                report = {}
                if asset.get("kind") == "mesh":
                    # Raw meshes in a library root need converting; OmniGibson
                    # cannot load a .glb directly.
                    bundle = self._free_bundle_dir(
                        asset["category"], import_slug(asset["asset_id"]))
                    usd_absolute, report = convert_mesh(
                        asset["usd"], bundle, import_slug(asset["asset_id"]),
                        scale=payload.get("mesh_scale", 1.0),
                        up_axis=payload.get("up_axis", "auto"),
                        collision=payload.get("collision", "convexHull"),
                        mass=payload.get("mass"),
                    )
                    usd_relative = Path(usd_absolute).resolve().relative_to(
                        Path(self.scene_json_path).parent).as_posix()
                else:
                    usd_relative, usd_absolute = import_asset(
                        asset, self.scene_json_path)
            except SceneEditError as e:
                self._json(400, {"error": str(e)})
                return
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
                return

            entry = self._register_import(
                usd_relative, usd_absolute, asset["category"], asset["asset_id"],
                transform, extra={"importReport": report} if report else None,
            )
            if entry is None:
                return

        self._json(200, {"object": entry, "report": report,
                         "scene_revision": self.bump_revision()})

    def _read_binary_request(self):
        """Guards for a binary upload, mirroring ``_read_json_request``.

        Returns ``(filename, data)``, or None after sending an error.
        """
        if not self._host_allowed():
            self._json(403, {"error": "unrecognised Host header"})
            return None

        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin:
            try:
                same_origin = urlsplit(origin).netloc == host
            except ValueError:
                same_origin = False
            if not same_origin:
                self._json(403, {"error": "cross-origin upload rejected"})
                return None

        supplied = self.headers.get("X-Editor-Token", "")
        if not secrets.compare_digest(supplied, self.mutation_token or ""):
            self._json(403, {"error": "missing or invalid editor token; reload the page"})
            return None

        # Only the basename is honoured: the name decides an extension and a
        # slug, never a location on disk.
        raw_name = self.headers.get("X-Filename", "")
        filename = Path(unquote(raw_name)).name
        if not filename or filename in (".", ".."):
            self._json(400, {"error": "X-Filename header required"})
            return None

        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._json(400, {"error": "bad Content-Length"})
            return None
        if length <= 0:
            self._json(400, {"error": "empty upload"})
            return None
        if length > UPLOAD_MAX_BYTES:
            self._json(413, {
                "error": f"file is larger than the {UPLOAD_MAX_BYTES // (1024 * 1024)} MB "
                         "upload limit; use Browse… to import it in place instead"
            })
            return None

        data = self.rfile.read(length)
        if len(data) != length:
            self._json(400, {"error": "upload truncated"})
            return None
        return filename, data

    def _upload(self):
        """Stage a dropped file on disk and hand back its path.

        The client then calls /api/import_path exactly as the picker does.
        """
        result = self._read_binary_request()
        if result is None:
            return
        filename, data = result

        suffix = Path(filename).suffix.lower()
        if suffix not in UPLOADABLE_SUFFIXES:
            self._json(400, {
                "error": f"{filename}: drag-and-drop takes a self-contained mesh "
                         f"({', '.join(sorted(UPLOADABLE_SUFFIXES))}). A .usd usually "
                         "references textures beside it, which a dropped file cannot "
                         "carry — use Browse… for those."
            })
            return

        staging = upload_staging_dir()
        # A fresh subdirectory per upload: two drops of "model.glb" must not
        # overwrite each other while the first is still being imported.
        slot = Path(tempfile.mkdtemp(prefix="drop_", dir=staging))
        target = slot / import_slug(Path(filename).stem)
        target = target.with_suffix(suffix)
        try:
            target.write_bytes(data)
        except OSError as e:
            self._json(500, {"error": f"could not stage {filename}: {e}"})
            return

        print(f"[upload] {filename} ({len(data)} bytes) -> {target}")
        self._json(200, {"path": str(target), "name": filename, "bytes": len(data)})

    def _browse(self):
        """List one directory so the browser can offer a file picker.

        The picker needs absolute paths on the *server's* machine, which a
        native file input never reveals. A POST, despite only reading, so it
        inherits the write-endpoint guards; it exposes no authority beyond
        ``/api/import_path``.
        """
        payload = self._read_json_request()
        if payload is None:
            return

        # The same dialog serves the task-yaml Browse button, with a filter.
        yaml_mode = payload.get("filter") == "yaml"

        raw = payload.get("path")
        try:
            if raw in (None, ""):
                target = self.browse_default
            else:
                target = Path(str(raw)).expanduser()
                if not target.is_absolute():
                    raise SceneEditError("path must be absolute")
                target = target.resolve(strict=True)
        except (OSError, RuntimeError):
            self._json(400, {"error": f"no such directory: {raw}"})
            return
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return

        if not target.is_dir():
            self._json(400, {"error": f"not a directory: {target}"})
            return

        try:
            children = sorted(
                target.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            self._json(403, {"error": f"permission denied: {target}"})
            return
        except OSError as e:
            self._json(400, {"error": f"cannot read {target}: {e}"})
            return

        entries, truncated = [], False
        for child in children:
            # Skip dot-entries.
            if child.name.startswith("."):
                continue
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            if is_dir:
                kind = "directory"
            elif yaml_mode:
                kind = "yaml" if child.suffix.lower() in (".yaml", ".yml") else None
            else:
                kind = classify(child)
            # List only directories and importable files.
            if kind is None:
                continue
            entry = {"name": child.name, "path": str(child), "kind": kind}
            if not is_dir:
                try:
                    entry["size"] = child.stat().st_size
                except OSError:
                    entry["size"] = None
            entries.append(entry)
            if len(entries) >= BROWSE_ENTRY_LIMIT:
                truncated = True
                break

        self._json(200, {
            "path": str(target),
            # None at the filesystem root, which is where "up" has to stop.
            "parent": str(target.parent) if target.parent != target else None,
            "entries": entries,
            "truncated": truncated,
            "shortcuts": [
                {"label": label, "path": str(path)}
                for label, path in self.browse_shortcuts
                if Path(path).is_dir()
            ],
        })

    def _import_path(self):
        """Import an asset from anywhere on this machine.

        Takes a path rather than a library key, guarded like every other
        mutation. Reads only what the user names, and only files with an
        asset extension. A directory is added to the library instead of
        imported.
        """
        payload = self._read_json_request()
        if payload is None:
            return

        try:
            source = resolve_user_path(payload.get("path"))
            kind = classify(source)
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return

        if kind == "directory":
            with WRITE_LOCK:
                if source in self.asset_roots:
                    self._json(200, {
                        "kind": "directory", "added_root": False,
                        "assets": self.list_assets(),
                        "roots": [str(r) for r in self.asset_roots],
                        "message": f"{source} is already in the library",
                    })
                    return
                EditorHandler.asset_roots = tuple(self.asset_roots) + (source,)
                EditorHandler._assets_cache = None
                assets = self.list_assets()
            found = sum(1 for a in assets if a["usd"].startswith(str(source)))
            print(f"[library] added root {source} ({found} asset(s))")
            self._json(200, {
                "kind": "directory", "added_root": True, "assets": assets,
                "roots": [str(r) for r in EditorHandler.asset_roots],
                "message": f"{found} asset(s) found in {source.name}",
            })
            return

        category = payload.get("category") or import_slug(source.stem)
        if not isinstance(category, str):
            self._json(400, {"error": "category must be a string"})
            return
        category = import_slug(category)

        with WRITE_LOCK:
            try:
                transform = self._requested_transform(payload)
                # Land in the same place a library import would, so the scene
                # directory has one layout however an asset arrived.
                asset_id = import_slug(source.stem)
                bundle = self._free_bundle_dir(category, asset_id)
                report = {}
                if kind == "usd":
                    usd_absolute, notes = import_usd_file(source, bundle)
                    report = {"notes": notes}
                else:
                    usd_absolute, report = convert_mesh(
                        source, bundle, asset_id,
                        scale=payload.get("mesh_scale", 1.0),
                        up_axis=payload.get("up_axis", "auto"),
                        collision=payload.get("collision", "convexHull"),
                        mass=payload.get("mass"),
                    )
                usd_relative = Path(usd_absolute).resolve().relative_to(
                    Path(self.scene_json_path).parent).as_posix()
            except SceneEditError as e:
                self._json(400, {"error": str(e)})
                return
            except Exception as e:  # noqa: BLE001 - a malformed file must not kill the server
                traceback.print_exc()
                self._json(422, {"error": f"could not import {source.name}: "
                                          f"{type(e).__name__}: {e}"})
                return

            entry = self._register_import(
                usd_relative, usd_absolute, category, asset_id, transform,
                extra={"importedFrom": str(source), "importReport": report},
            )
            if entry is None:
                return
            # Invalidate the cache so the import shows up in the library.
            EditorHandler._assets_cache = None

        self._json(200, {"kind": kind, "object": entry, "report": report,
                         "source": str(source),
                         "scene_revision": self.bump_revision()})

    def _free_bundle_dir(self, category, asset_id):
        """An unused ``objects/<category>/<asset_id>`` under the scene.

        Never overwrite: re-importing an edited file under the same name must
        not silently change the geometry of an object already placed from the
        previous import.
        """
        base = Path(self.scene_json_path).parent / "objects" / category
        candidate = base / asset_id
        suffix = 1
        while candidate.exists():
            candidate = base / f"{asset_id}_{suffix}"
            suffix += 1
        return candidate

    # --- the scene launcher -------------------------------------------------
    # Opening a scene rebinds the whole editor and ends with the page
    # reloading; a switch either completes or changes nothing.

    @classmethod
    def repo_root(cls):
        """The checkout this editor is part of; a method so tests can move it."""
        return HERE.parents[2]

    @classmethod
    def asset_lookup_roots(cls):
        """The dataset and robot roots a scene's assets are resolved against.

        A ``DatasetObject`` and a robot name their assets by class rather than
        by path; without these roots a scene with missing props reads as
        complete.
        """
        if cls.options is None:
            return {"dataset_dir": None, "robot_asset_dir": None}
        return {"dataset_dir": cls.options.dataset_root,
                "robot_asset_dir": cls.options.robot_asset_root}

    @classmethod
    def scene_catalog(cls):
        """Recent scenes and everything the scene roots hold."""
        recents = cls.recents.entries(**cls.asset_lookup_roots()) if cls.recents else []
        current = str(cls.scene_json_path) if cls.scene_json_path else None
        # The open scene decides its own directory's row so the launcher can
        # mark it.
        catalog = discover_scenes(cls.scene_roots, open_scene=cls.scene_json_path,
                                  **cls.asset_lookup_roots())
        # Label each scene's provenance (shipped vs reconstructed), derived
        # from the path.
        repo_root = HERE.parents[2]
        for row in catalog:
            cls._label_source(row, row.get("dir"), repo_root)
        for row in recents:
            cls._label_source(row, row.get("path"), repo_root)
        return {
            "current": current,
            "current_name": cls.scene_name,
            "recent": [r for r in recents if r["path"] != current],
            "scenes": catalog,
            "roots": [str(r) for r in cls.scene_roots],
            "compose_root": str(cls.compose_root) if cls.compose_root else None,
            "session_revision": cls.session_revision,
        }

    @staticmethod
    def _label_source(row, path, repo_root):
        """Label where a scene came from, and rename generated scenes."""
        row["source"] = scene_source(path, repo_root)
        if row["source"] == "generated":
            better = generated_scene_name(row.get("path") or path)
            if better:
                row["name"] = better

    @classmethod
    def recent_catalog(cls):
        """Just the recent-scenes strip, without walking the scene roots."""
        recents = cls.recents.entries(**cls.asset_lookup_roots()) if cls.recents else []
        current = str(cls.scene_json_path) if cls.scene_json_path else None
        repo_root = HERE.parents[2]
        for row in recents:
            cls._label_source(row, row.get("path"), repo_root)
        return {"recent": [r for r in recents if r["path"] != current]}

    @classmethod
    def scene_lookup_roots(cls):
        """Roots an endpoint may answer questions *about* a scene from.

        ``scene_roots`` plus the directories the recents list names, which
        routinely span checkouts. Grants no new authority: a recents entry is
        a path this server itself recorded.
        """
        roots = list(cls.scene_roots)
        seen = {str(r) for r in roots}
        for entry in (cls.recents.entries(describe=False) if cls.recents else []):
            parent = Path(entry["path"]).parent
            if str(parent) not in seen:
                seen.add(str(parent))
                roots.append(parent)
        return tuple(roots)

    def _pending_draft(self):
        """What the *server* knows is unsaved: imports no save has written.

        Transform edits live in the browser and are guarded there; imports
        live here and would be discarded silently by a scene switch.
        """
        return sorted(set(self.pending_adds) - self.saved_adds)

    def _open_scene(self):
        payload = self._read_json_request()
        if payload is None:
            return
        pending = self._pending_draft()
        if pending and payload.get("discard_pending") is not True:
            self._json(409, {
                "error": "this session is holding imported object(s) that no save has "
                         "written yet: " + ", ".join(pending),
                "pending_adds": pending,
            })
            return
        try:
            target = resolve_scene_path(payload.get("path"), self.scene_roots)
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return

        # One switch at a time, and none while a save is compiling: bind_scene
        # replaces the base document every save recompiles against.
        with WRITE_LOCK:
            if target == self.scene_json_path and payload.get("reload") is not True:
                self._json(200, {"scene": str(target), "unchanged": True,
                                 "session_revision": self.session_revision})
                return
            try:
                summary = bind_scene(target, self.options, reuse_cache="prefer")
            except (SceneEditError, OSError, ValueError, json.JSONDecodeError) as e:
                # bind_scene leaves the previous scene bound on failure, so the
                # session is still usable and the browser need not reload.
                self._json(400, {"error": str(e), "scene": str(self.scene_json_path)})
                return
            if self.recents:
                self.recents.record(target)
        print(f"[open] {target}")
        self._json(200, summary)

    def _scene_template(self):
        """Describe what a new composition would inherit from a template."""
        payload = self._read_json_request()
        if payload is None:
            return
        try:
            template = resolve_scene_path(payload.get("path"), self.scene_roots)
            self._json(200, template_summary(template, **self.asset_lookup_roots()))
        except SceneEditError as e:
            self._json(400, {"error": str(e)})

    def _compose_scene(self):
        """Create a new scene from a template, and open it."""
        payload = self._read_json_request()
        if payload is None:
            return
        keep = payload.get("keep", [])
        if not isinstance(keep, list) or not all(isinstance(n, str) for n in keep):
            self._json(400, {"error": "keep must be a list of object names"})
            return
        pending = self._pending_draft()
        if pending and payload.get("discard_pending") is not True:
            self._json(409, {
                "error": "this session is holding imported object(s) that no save has "
                         "written yet: " + ", ".join(pending),
                "pending_adds": pending,
            })
            return
        try:
            template = resolve_scene_path(payload.get("template"), self.scene_roots)
            name = validate_scene_name(payload.get("name"))
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return

        with WRITE_LOCK:
            try:
                result = compose_scene(template, name, keep, dest_root=self.compose_root,
                                       **self.asset_lookup_roots())
            except (SceneEditError, OSError, ValueError) as e:
                self._json(400, {"error": str(e)})
                return
            print(f"[compose] {result['path']} from {Path(template).name} "
                  f"({len(result['kept'])} kept, {len(result['dropped'])} dropped)")
            try:
                summary = bind_scene(Path(result["path"]), self.options, reuse_cache="never")
            except (SceneEditError, OSError, ValueError, json.JSONDecodeError) as e:
                # The scene is on disk and passed compose's checks, so a
                # failure here is about extraction; keep the file and say
                # where it is.
                self._json(500, {
                    "error": (f"created {result['path']} but could not open it: {e}. "
                              "The scene is on disk and will appear in the launcher; "
                              "opening it again will retry the extraction."),
                    "created": result,
                    "kept": True,
                })
                return
            if self.recents:
                self.recents.record(result["path"])
        self._json(200, {**summary, "created": result})

    def do_POST(self):
        if self.path == "/api/open":
            self._open_scene()
            return
        if self.path == "/api/template":
            self._scene_template()
            return
        if self.path == "/api/compose":
            self._compose_scene()
            return
        if self.path == "/api/save_table":
            self._save_table()
            return
        if self.path == "/api/load_cameras":
            self._load_cameras()
            return
        if self.path == "/api/save_cameras":
            self._save_cameras()
            return
        if self.path == "/api/add":
            self._add_object()
            return
        if self.path == "/api/background":
            self._set_background()
            return
        if self.path == "/api/import_path":
            self._import_path()
            return
        if self.path == "/api/check_bindings":
            self._check_bindings()
            return
        if self.path == "/api/inspect":
            self._inspect()
            return
        if self.path == "/api/export":
            self._export()
            return
        if self.path == "/api/open_folder":
            self._open_folder()
            return
        if self.path == "/api/browse":
            self._browse()
            return
        if self.path == "/api/task_cfg":
            self._save_task_cfg()
            return
        if self.path == "/api/task_propose":
            self._task_propose()
            return
        if self.path == "/api/task_create":
            self._task_create()
            return
        if self.path == "/api/upload":
            self._upload()
            return
        if self.path != "/api/save":
            self._json(404, {"error": "not found"})
            return

        payload = self._read_json_request()
        if payload is None:
            return

        runner = self.settle_runner
        settling = bool(runner and runner.enabled)

        try:
            # The revision check, compile, write and revision bump are one
            # critical section; validating outside the lock would let two
            # saves carrying the same revision both pass.
            with WRITE_LOCK:
                plan = self._plan_scene_write(payload)
                promote = plan["promote"]
                compiled = self._compile_scene(plan)
                added, deleted, changed = (
                    compiled["added"], compiled["deleted"], compiled["changed"])
                # When settling, promotion is deferred to the settled output —
                # promoting the hand-placed file first would leave _latest
                # holding a pose that physics has not confirmed.
                promoted_now = promote and not settling
                out, promoted_digest = self._publish_scene(
                    compiled["scene"], promote=promoted_now)
                revision = self.adopt_written_scene(
                    compiled["scene"],
                    promoted_digest=promoted_digest,
                    promoted_path=latest_path(self.scene_json_path) if promoted_now else None,
                )
                # These imports are now on disk; a switch no longer discards
                # them.
                EditorHandler.saved_adds = set(added)
        except HttpRefusal as refusal:
            # A refusal carries its own status; do not fall through to 500.
            self._json(refusal.code, refusal.body)
            return
        except TargetChanged as e:
            # A cross-process collision, not a bug: something outside this
            # editor replaced the file between the digest we hold and now.
            self._json(409, {"error": str(e), "reason": "target_changed"})
            return
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return
        except Exception as e:
            self._json(500, {"error": str(e)})
            return

        summary = f"[save] {len(changed)} moved"
        if added:
            summary += f", {len(added)} added"
        if deleted:
            summary += f", {len(deleted)} removed"
        if compiled["ground_plane"] != "unchanged":
            summary += f", {compiled['ground_plane']} {describe_ground_plane(plan['ground'])}"
        print(f"{summary} -> {out}"
              + ("  (promoted to _latest)" if promoted_now else ""))

        settle = {"state": "disabled"}
        if settling:
            settle = {"state": "running", "id": runner.start(out, promote)}

        self._json(
            200,
            {
                "changed": changed,
                "added": added,
                "removed": deleted,
                # What the write did to the ground plane, and what it now
                # says; the browser re-baselines "unsaved" on it.
                "ground_plane": compiled["ground_plane"],
                "ground_plane_info": plan["ground"],
                "path": str(out),
                "promoted": promoted_now,
                # Why _latest was left alone.
                "promotion_deferred": bool(promote and settling),
                "settle": settle,
                "base_scene_sha256": self.base_scene_sha256,
                "scene_revision": revision,
                # What this edit does to the task configs that bind these
                # objects. An empty group produces episodes that vacuously
                # succeed, which is indistinguishable from a policy result.
                "task_warnings": [
                    self.wire_warning(w)
                    for w in self.binding_warnings(
                        removed=deleted,
                        added={n: self.pending_adds[n]["category"]
                               for n in added if n in self.pending_adds},
                    )
                ],
            },
        )

    def _check_bindings(self):
        """Task-binding warnings for an edit the user has not saved yet.

        Same answer the save response carries, asked before committing.
        """
        payload = self._read_json_request()
        if payload is None:
            return
        removed = payload.get("remove", [])
        if not isinstance(removed, list) or not all(isinstance(n, str) for n in removed):
            self._json(400, {"error": "remove must be a list of object names"})
            return
        keep = set(payload.get("keep") or [])
        added = {
            name: spec["category"]
            for name, spec in self.pending_adds.items()
            if name in keep
        }
        warnings = self.binding_warnings(removed=removed, added=added)
        self._json(200, {
            "warnings": [self.wire_warning(w) for w in warnings],
            "configs": len(self.task_configs),
        })

    def _associated_tasks(self):
        """Task configs that plausibly belong to this scene, best match first.

        Answered against `current_scene`, which differs from the compilation
        origin after a promotion.
        """
        return self._tasks_for(self.current_scene or self.base_scene, self.scene_name)

    def _known_rooms(self):
        """Every registered room this checkout can see, sorted by label.

        A room is a USD with a ``.background.json`` sidecar carrying its
        registered pose.
        """
        try:
            roots = background_roots(self.scene_json_path, self.options.repo_root)
            return sorted(discover_backgrounds(roots), key=lambda r: r["label"].lower())
        except Exception as e:  # noqa: BLE001 - a bad sidecar must not 500 the list
            print(f"[backgrounds] WARNING: {type(e).__name__}: {e}")
            return []

    def _tasks_for(self, scene_data, scene_name):
        """Same lookup as `_associated_tasks`, against any parsed scene."""
        if not self.task_configs:
            return []
        try:
            records = task_bindings.bindings_for_scene(
                scene_data, list(self.task_configs),
                scene_name=scene_name,
                known_scenes=set(self.known_scene_names),
            )
        except Exception as e:  # noqa: BLE001 - advisory, never fatal
            print(f"[export] WARNING: task lookup failed: {type(e).__name__}: {e}")
            return []
        order = {"certain": 0, "likely": 1, "possible": 2}
        records.sort(key=lambda r: order.get(
            (r.get("association") or {}).get("confidence"), 9))
        return [
            {
                "name": r.get("task_name") or r.get("task"),
                "path": str(r.get("path")),
                "group": export_bundle.task_group_name(r.get("path"), HERE.parents[2]),
                "instruction": r.get("instruction"),
                "confidence": (r.get("association") or {}).get("confidence"),
                "evidence": (r.get("association") or {}).get("evidence"),
            }
            for r in records
        ]

    def _resolve_task_cfg_path(self, raw):
        """Check a task-yaml path the browser asked to read or write.

        Sandboxed to the repo: Browse can pick a config anywhere in the
        checkout, but not any file the server process can read.
        """
        if not isinstance(raw, str) or not raw.strip():
            raise SceneEditError("a task config path is required")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise SceneEditError(f"task config path must be absolute: {raw}")
        try:
            path = path.resolve(strict=True)
        except OSError as e:
            raise SceneEditError(f"no such file: {raw} ({e.strerror})") from None
        if not path.is_file() or path.suffix.lower() not in (".yaml", ".yml"):
            raise SceneEditError(f"not a yaml file: {path}")
        root = HERE.parents[2].resolve()
        if not (path == root or root in path.parents):
            raise SceneEditError(f"{path} is outside the repository")
        return path

    def _task_cfg_groups(self, path, og_cfg):
        """Every group in a task config, and what the panel needs to draw it.

        The union of every place a group can be named: a group with no
        randomization entry yet is exactly the one this panel exists to give
        one to.
        """
        mapping = og_cfg.get("semantic_group_mapping")
        mapping = mapping if isinstance(mapping, dict) else {}
        xyz = og_cfg.get("group_xyz_randomization")
        xyz = xyz if isinstance(xyz, dict) else {}
        z_rot = og_cfg.get("group_z_rot_randomization")
        z_rot = z_rot if isinstance(z_rot, dict) else {}
        placed = og_cfg.get("group_predicate_placement")
        placed = placed if isinstance(placed, dict) else {}
        bound = self._task_cfg_objects(path)

        def keys_of(group):
            keys = mapping.get(group)
            if isinstance(keys, str):
                return [keys]
            return [str(key) for key in keys] if isinstance(keys, list) else []

        return {
            group: {
                "xyz": xyz.get(group),
                "z_rot": z_rot.get(group),
                "keys": keys_of(group),
                "objects": bound.get(group, []),
                # `place_with_predicate` runs after group_xyz_randomization and
                # overwrites X and Y, so those ranges are dead for these
                # groups; the panel greys them out.
                "predicate_placed": group in placed,
            }
            for group in sorted(set(mapping) | set(xyz) | set(z_rot))
        }

    def _task_cfg_objects(self, path):
        """Which objects in the open scene each of this task's groups binds.

        Resolved through `task_bindings.groups_for_scene`, the same rule the
        runtime uses (name or category in the group's keys), so this panel and
        the save-time warnings cannot disagree.

        Advisory: a config unrelated to the open scene binds nothing, which is
        a legitimate state and not an error.
        """
        if not self.base_scene:
            return {}
        config = next((c for c in self.task_configs if Path(c["path"]) == path), None)
        if config is None:
            # Browsed from outside `scripts/cfg/task`, or written since startup.
            config = task_bindings.read_task_config(path)
        if config is None:
            return {}
        try:
            categories = self.session_categories()
            return {
                group: list(state["objects"])
                for group, state in task_bindings.groups_for_scene(
                    config, categories).items()
            }
        except Exception as e:  # noqa: BLE001 - advisory, never fatal
            print(f"[tasks] WARNING: binding lookup failed: {type(e).__name__}: {e}")
            return {}

    def _save_task_cfg(self):
        """Write edited per-group randomization ranges back into a task yaml.

        The existing maps are mutated in place and ruamel round-trips the
        rest, so comments and groups the panel did not send back survive.
        """
        payload = self._read_json_request()
        if payload is None:
            return
        try:
            path = self._resolve_task_cfg_path(payload.get("path"))
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return

        expected = payload.get("sha256")
        if not isinstance(expected, str) or not expected.strip():
            self._json(400, {"error": "sha256 from the last read of this file is required"})
            return

        groups = payload.get("groups")
        if not isinstance(groups, dict):
            self._json(400, {"error": "groups must be an object"})
            return
        # None here means "no entry", which is what deletes the group's key --
        # distinct from a group being absent from the payload altogether, which
        # leaves it alone.
        xyz_out, z_rot_out = {}, {}
        for group, spec in groups.items():
            if not isinstance(spec, dict):
                self._json(400, {"error": f"{group}: expected an object"})
                return
            xyz = spec.get("xyz")
            if xyz is not None:
                if (not isinstance(xyz, list) or len(xyz) != 3
                        or not all(isinstance(v, (int, float))
                                   and not isinstance(v, bool) and v >= 0 for v in xyz)):
                    self._json(400, {"error": f"{group}: xyz must be 3 non-negative numbers"})
                    return
                xyz = [float(v) for v in xyz]
            z_rot = spec.get("z_rot")
            if z_rot is not None:
                if (not isinstance(z_rot, (int, float)) or isinstance(z_rot, bool)
                        or z_rot < 0):
                    self._json(400, {"error": f"{group}: z_rot must be a non-negative number"})
                    return
                z_rot = float(z_rot)
            xyz_out[group], z_rot_out[group] = xyz, z_rot

        note = None
        if YAML is None:
            if not payload.get("accept_comment_loss"):
                self._json(409, {
                    "error": (f"ruamel.yaml is not installed here, so saving {path.name} "
                              "would strip its comments and reflow it. Install it (see "
                              "requirements.txt), or save again accepting the loss."),
                    "reason": "comment_loss",
                })
                return
            note = ("saved without ruamel.yaml installed — comments and formatting "
                    "in this file were not preserved")

        # Compare-then-write is one transaction, as for a scene save.
        with WRITE_LOCK:
            try:
                raw_bytes = path.read_bytes()
            except OSError as e:
                self._json(400, {"error": f"could not read {path}: {e}"})
                return
            if hashlib.sha256(raw_bytes).hexdigest() != expected:
                self._json(409, {
                    "error": (f"{path.name} changed on disk since this panel read it — "
                              "reload the panel to see the current ranges before saving."),
                    "reason": "stale",
                })
                return
            try:
                original = raw_bytes.decode("utf-8")
                document = load_task_yaml(original)
            except (UnicodeDecodeError, yaml.YAMLError, RuamelError) as e:
                # A hand-broken config is the user's file to fix, not a crash.
                self._json(400, {"error": f"could not parse {path}: {e}"})
                return
            og_cfg = document.get("og_task_config") if isinstance(document, dict) else None
            if not isinstance(og_cfg, dict):
                self._json(400, {"error": f"{path.name} has no og_task_config"})
                return
            apply_group_ranges(og_cfg, "group_xyz_randomization", xyz_out)
            apply_group_ranges(og_cfg, "group_z_rot_randomization", z_rot_out)
            try:
                text = dump_task_yaml(document, original)
            except (yaml.YAMLError, RuamelError) as e:
                self._json(500, {"error": f"could not serialize {path}: {e}"})
                return
            try:
                # Re-checked at the instant of publication: WRITE_LOCK does
                # not cover other processes.
                guarded_write_text(path, text, expect=expected)
            except TargetChanged as e:
                self._json(409, {"error": str(e), "reason": "stale",
                                 "sha256": file_digest(path)})
                return
            except (OSError, TimeoutError) as e:
                self._json(500, {"error": f"could not write {path}: {e}"})
                return

        self._json(200, {
            "path": str(path),
            # The new revision, so the panel can save twice without re-reading.
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "note": note,
        })

    def _task_propose(self):
        """Generate one task yaml from a prompt, for the "Generate task" panel.

        Scoped to the scene this server process has open.
        """
        payload = self._read_json_request()
        if payload is None:
            return
        prompt = payload.get("prompt")
        objects = payload.get("objects")
        image_data_url = payload.get("image")
        robot_type = payload.get("robot_type") or "franka"
        if not isinstance(objects, list):
            self._json(400, {"error": "objects must be a list"})
            return
        if not isinstance(image_data_url, str) or not image_data_url:
            self._json(400, {"error": "image is required"})
            return

        header, _, encoded = image_data_url.partition(",")
        if "base64" not in header:
            self._json(400, {"error": "image must be a base64 data URL"})
            return
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as e:
            self._json(400, {"error": f"could not decode image: {e}"})
            return

        try:
            result = task_propose.propose_task(
                prompt, objects, image_bytes, self.scene_name, robot_type=robot_type,
            )
        except task_propose.TaskProposeError as e:
            self._json(400, {"error": str(e)})
            return
        except Exception as e:  # noqa: BLE001 - a Gemini/network failure, not a bug here
            self._json(502, {"error": f"Gemini call failed: {e}"})
            return

        default_dir = HERE.parents[2] / "scripts" / "cfg" / "task"
        self._json(200, {**result, "default_dir": str(default_dir)})

    def _resolve_task_dir(self, raw):
        """Check a directory the browser wants to save a *new* task yaml into.

        Narrower than `_resolve_task_cfg_path`: generated configs may only
        land under `scripts/cfg/task/`. Browsing stays repo-wide; this
        refuses at save time.
        """
        if not isinstance(raw, str) or not raw.strip():
            raise SceneEditError("an output folder is required")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise SceneEditError(f"output folder must be absolute: {raw}")
        try:
            path = path.resolve(strict=True)
        except OSError as e:
            raise SceneEditError(f"no such folder: {raw} ({e.strerror})") from None
        if not path.is_dir():
            raise SceneEditError(f"not a folder: {path}")
        # Resolved on both sides so a symlink cannot point out of the tree.
        root = task_config_root().resolve()
        if not (path == root or root in path.parents):
            raise SceneEditError(
                f"generated task configs are written under {root} — {path} is outside it"
            )
        return path

    def _task_create(self):
        """Write a newly generated task yaml to a user-chosen name and folder."""
        payload = self._read_json_request()
        if payload is None:
            return
        try:
            directory = self._resolve_task_dir(payload.get("dir"))
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return

        filename = payload.get("filename")
        if (not isinstance(filename, str) or not filename.strip()
                or "/" in filename or "\\" in filename or filename in (".", "..")
                or Path(filename).suffix.lower() not in (".yaml", ".yml")):
            self._json(400, {"error": "filename must be a bare name ending in .yaml or .yml"})
            return

        yaml_text = payload.get("yaml_text")
        if not isinstance(yaml_text, str) or not yaml_text.strip():
            self._json(400, {"error": "yaml_text is required"})
            return
        try:
            document = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            self._json(400, {"error": f"yaml_text is not valid YAML: {e}"})
            return

        # `task_semantics` checks the config on its own terms;
        # `_scene_side_problems` checks it against the open scene.
        problems = task_semantics.validate_task(document)
        # Only the scene-independent half can refuse: a config may target a
        # scene that is not the one open.
        warnings = problems + self._scene_side_problems(yaml_text)

        # A config that cannot run may still be saved deliberately as a draft;
        # the refusal names what is wrong and the caller may insist.
        allow_invalid = payload.get("allow_invalid", False)
        if not isinstance(allow_invalid, bool):
            self._json(400, {"error": "allow_invalid must be a JSON boolean"})
            return
        blocking = task_semantics.blocking(problems)
        if blocking and not allow_invalid:
            self._json(422, {
                "error": (f"{filename} would not run: "
                          + task_semantics.summarize(blocking)),
                "reason": "invalid_task",
                "problems": [self.wire_warning(w) for w in warnings],
                "path": str(directory / filename),
            })
            return

        target = directory / filename
        # Existence check, hash comparison, write and rediscovery are one
        # transaction under the lock. Overwriting is authorised by a hash,
        # never a boolean: "I saw exactly this" survives the file changing
        # while the dialog is open.
        confirmed = payload.get("overwrite_sha256")
        if confirmed is not None and not isinstance(confirmed, str):
            self._json(400, {"error": "overwrite_sha256 must be a hex digest string"})
            return
        with WRITE_LOCK:
            try:
                current = hashlib.sha256(target.read_bytes()).hexdigest()
            except FileNotFoundError:
                current = None
            except OSError as e:
                self._json(500, {"error": f"could not read {target}: {e}"})
                return

            if current is not None and confirmed != current:
                # `exists`: the name is taken on a first attempt. `stale`: a
                # confirmed overwrite whose hash no longer matches. The current
                # digest goes back either way so a second confirmation can
                # quote it.
                stale = confirmed is not None
                self._json(409, {
                    "error": (
                        f"{filename} changed on disk after this dialog opened — "
                        "nothing was written. Re-read it before overwriting."
                        if stale else
                        f"{filename} already exists in {directory}"
                    ),
                    "reason": "stale" if stale else "exists",
                    "sha256": current,
                    "path": str(target),
                })
                return

            try:
                # expect=None asks for an exclusive create; a digest asks for
                # a replacement of exactly those bytes.
                guarded_write_text(target, yaml_text, expect=current)
            except TargetChanged as e:
                # Caught again at the instant of the write: `expected is None`
                # means a create whose name has since been taken; anything
                # else is a confirmed overwrite whose bytes moved.
                self._json(409, {
                    "error": str(e),
                    "reason": "exists" if e.expected is None else "stale",
                    "sha256": file_digest(target),
                    "path": str(target),
                })
                return
            except OSError as e:
                self._json(500, {"error": f"could not write {target}: {e}"})
                return

            written = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()
            # Re-discover so the new task is visible to the export flow and
            # task pickers; inside the lock so the list never describes a
            # half-finished write.
            self.refresh_task_configs()
        self._json(200, {
            "path": str(target),
            "warnings": [self.wire_warning(w) for w in warnings],
            # Whether what was written is runnable or a draft somebody
            # insisted on.
            "runnable": not blocking,
            "sha256": written,
        })

    def _task_file_problems(self, path):
        """`task_semantics.validate_task` for a config that is already on disk.

        Args:
            path (str or Path or None): The config, or None for "no task
                chosen", which has nothing to report.

        Returns:
            list[dict]: Problems, each carrying ``task`` (the file stem) and
            ``path`` alongside `task_semantics`' own fields.
        """
        if not path:
            return []
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as e:
            # Unreadable is a hard problem, not a silent pass.
            return [{"kind": "unreadable", "effect": "load_error",
                     "severity": "breaks", "where": "", "path": str(path),
                     "task": Path(path).stem,
                     "message": f"could not read {path}: {e.strerror}"}]
        return [{**problem, "task": Path(path).stem, "path": str(path)}
                for problem in task_semantics.validate_task_yaml(text)]

    def _scene_side_problems(self, yaml_text):
        """What a proposed task config's groups resolve to in *this* scene.

        Resolved with the same pending-import-aware object index the save-time
        warnings use, so the two cannot disagree. Reports cardinality, not
        just emptiness: a group binding nothing succeeds vacuously, and one
        binding two objects can assert at reset. The association heuristics
        are skipped: this config was generated for the open scene.

        Returns:
            list[dict]: Warnings in `task_semantics`' shape.
        """
        if not self.base_scene:
            return []
        try:
            with tempfile.TemporaryDirectory() as staging:
                probe = Path(staging) / "task.yaml"
                probe.write_text(yaml_text, encoding="utf-8")
                config = task_bindings.read_task_config(probe)
            if config is None:
                # No mapping worth resolving; task_semantics has already said so.
                return []
            categories = self.session_categories()
            warnings = []
            for group, state in task_bindings.groups_for_scene(
                    config, categories).items():
                keys = state["keys"]
                if state["effect"] is None:
                    continue
                severity = task_bindings.severity_of(state["effect"])
                # Reported even for `no_effect`: a group matching nothing is
                # the tell that a model invented an object name.
                named = ", ".join(keys) if keys else "<no keys>"
                if state["count"] == 0:
                    text = (f"{group}: nothing in {self.scene_name} is named or "
                            f"categorized {named}")
                else:
                    text = (f"{group}: {named} matches {state['count']} objects in "
                            f"{self.scene_name} ({', '.join(state['objects'])})")
                # The consequence comes from task_bindings' own effect table.
                consequence = task_bindings._EFFECT_DETAIL.get(state["effect"], "")
                if consequence:
                    text += f" — {consequence}"
                warnings.append({
                    # `kind` is renamed to `code` by wire_warning.
                    "kind": "empty_group" if state["count"] == 0 else "ambiguous_group",
                    "effect": state["effect"],
                    "severity": severity,
                    "group": group,
                    "keys": list(keys),
                    "where": "og_task_config.semantic_group_mapping." + str(group),
                    "message": text,
                    "detail": text,
                })
            return warnings
        except Exception as e:  # noqa: BLE001 - advisory, never fatal
            print(f"[tasks] WARNING: new-task binding check failed: {type(e).__name__}: {e}")
            return []

    def _camera_target(self, will_write):
        """Which external_sensors config the evaluation should be told to use.

        Until something is written, the config that exists is the one loaded;
        naming an unwritten ``camera_out_path`` would fail at run time.
        """
        if not self.cameras:
            return None
        path = self.camera_out_path if will_write else self.camera_source_path
        return {
            "cfg_name": Path(path).stem,
            "path": str(path),
            "background": self.camera_background,
            "resumed": self.camera_resumed,
            "is_template": (not will_write) and self.camera_source_path == self.camera_template_path,
        }

    def _task_group_entry(self, group):
        """The task an export named, whether or not the heuristic offered it.

        The Task panel lets somebody browse to a config the association guess
        would never produce; the export must accept it too.

        Returns:
            dict or None: The task record, or None for "no task chosen".

        Raises:
            HttpRefusal: If *group* names no config under the task group root.
        """
        if not group:
            return None
        known = {t["group"]: t for t in self._associated_tasks() if t["group"]}
        if group in known:
            return dict(known[group])
        repo_root = self.repo_root()
        candidate = (repo_root / export_bundle.TASK_GROUP_ROOT / f"{group}.yaml").resolve()
        # Still sandboxed: a group is a path fragment, and "any yaml under the
        # task group" is the widest this may ever be.
        root = (repo_root / export_bundle.TASK_GROUP_ROOT).resolve()
        if not (root in candidate.parents and candidate.is_file()):
            raise HttpRefusal(400, {
                "error": f"unknown task config {group!r}",
                "available": sorted(known),
            })
        document = {}
        try:
            document = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            # The semantic check below reads the file properly and reports what
            # is wrong with it; this lookup only needs a name and an instruction.
            document = {}
        return {
            "name": document.get("task_name") or candidate.stem,
            "path": str(candidate),
            "group": group,
            "instruction": document.get("language_instruction"),
            "confidence": "chosen",
            "evidence": "picked in the Task panel rather than inferred from the scene",
        }

    def _camera_snapshot_text(self, camera_edits, dry_run):
        """The camera config this export would pin, as text, written nowhere yet.

        Returns:
            tuple: ``(text, changed)`` -- the YAML the snapshot will hold and the
            camera names an edit moved. ``(None, [])`` when no rig is loaded.

        Raises:
            SceneEditError: If an edit is not valid against the loaded rig.
        """
        if not self.cameras:
            return None, []
        if not camera_edits:
            # No edits: pin the placement on disk, which is what a run loads.
            source = Path(self.camera_source_path)
            try:
                return source.read_text(encoding="utf-8"), []
            except OSError as e:
                raise SceneEditError(f"could not read {source}: {e}") from None
        validated = validate_camera_edits(self.cameras, camera_edits)
        document = copy.deepcopy(self.camera_document)
        changed = apply_camera_edits(document, validated)
        if dry_run:
            return None, list(changed)
        return camera_config_text(
            document, self.camera_source_path, self.scene_name,
            background=self.camera_background,
        ), list(changed)

    def _export(self):
        """Review and export one evaluation: scene, cameras, task, command.

        ``dry_run`` returns exactly what a real export would do without
        writing; the browser shows that and then calls again.

        Staged, then published: every artifact's bytes are built before any is
        written, the writes happen together under one lock, and a failure
        rolls back whatever landed.

        Snapshotted: the shared camera and task configs are copied into
        ``exports/`` under export-unique names with their digests recorded, so
        the manifest keeps describing the same evaluation.
        """
        payload = self._read_json_request()
        if payload is None:
            return
        dry_run = payload.get("dry_run", False)
        if not isinstance(dry_run, bool):
            self._json(400, {"error": "dry_run must be a JSON boolean"})
            return
        allow_invalid = payload.get("allow_invalid", False)
        if not isinstance(allow_invalid, bool):
            self._json(400, {"error": "allow_invalid must be a JSON boolean"})
            return
        # Snapshotting makes an export reproducible, so it is the default; the
        # manifest records when it is off.
        snapshot = payload.get("snapshot_configs", True)
        if not isinstance(snapshot, bool):
            self._json(400, {"error": "snapshot_configs must be a JSON boolean"})
            return

        task_group = payload.get("task")
        if task_group is not None and not isinstance(task_group, str):
            self._json(400, {"error": "task must be a string or null"})
            return
        camera_edits = payload.get("camera_edits") or {}
        if not isinstance(camera_edits, dict):
            self._json(400, {"error": "camera_edits must be an object"})
            return
        if camera_edits and not self.cameras:
            self._json(400, {"error": "camera edits sent but no camera config is loaded"})
            return

        repo_root = self.repo_root()
        staged = []
        review = None
        try:
            # One lock over the whole decision and publication; revision
            # checks inside it, as for a save.
            with WRITE_LOCK:
                task = self._task_group_entry(task_group)
                if camera_edits and payload.get("camera_revision") != self.camera_revision:
                    raise HttpRefusal(409, {
                        "error": "cameras changed since this page loaded; reload to continue",
                        "camera_revision": self.camera_revision,
                    })

                # A task that cannot run is refused before anything is
                # written, with the same allow_invalid escape
                # /api/task_create offers.
                task_problems = self._task_file_problems(task and task["path"])
                task_blocking = task_semantics.blocking(task_problems)
                if task_blocking and not dry_run and not allow_invalid:
                    raise HttpRefusal(422, {
                        "error": (f"{task_group} would not run: "
                                  + task_semantics.summarize(task_blocking)),
                        "reason": "invalid_task",
                        "task": task_group,
                        "problems": [self.wire_warning(w) for w in task_problems],
                    })

                plan = self._plan_scene_write(payload)
                promote = plan["promote"]
                settling = bool(self.settle_runner and self.settle_runner.enabled)
                compiled = self._compile_scene(plan)

                # --- stage: every byte decided, nothing written -------------
                camera_text, camera_changed = self._camera_snapshot_text(
                    camera_edits, dry_run)
                task_text = None
                if task and not dry_run:
                    try:
                        task_text = Path(task["path"]).read_text(encoding="utf-8")
                    except OSError as e:
                        raise SceneEditError(
                            f"could not read {task['path']}: {e}") from None

                scene_path = (
                    scene_output_path(self.scene_json_path) if not dry_run
                    else Path(self.scene_json_path).with_name(
                        f"{self.scene_name}_scene_state_light_edit_"
                        "<assigned on export>.json"))
                bundle = self._export_targets(
                    repo_root, scene_path, task, snapshot=snapshot and not dry_run)

                cameras = self._camera_target(will_write=bool(camera_edits))
                if cameras is not None:
                    cameras["written"] = bool(camera_edits) and not dry_run
                    cameras["changed"] = list(camera_changed)
                    if bundle["camera_cfg"]:
                        cameras["cfg_name"] = bundle["camera_cfg"]
                        cameras["snapshot"] = str(bundle["camera_snapshot"])

                command = export_bundle.eval_command(
                    repo_root=repo_root,
                    scene_name=self.scene_name,
                    scene_json=scene_path,
                    cameras_cfg=(bundle["camera_cfg"]
                                 or (cameras and cameras["cfg_name"])),
                    task_group=bundle["task_group"] or task_group,
                    # Naming the task is not enough: the stage prefers
                    # `s15_eval.prompt`, which the run configs set, so the
                    # instruction has to be pinned alongside the config.
                    prompt=task and task.get("instruction"),
                )

                warnings = {
                    # Config-side and scene-side task warnings, in one list.
                    "task": [
                        {k: w.get(k) for k in
                         ("task", "group", "severity", "effect", "message", "where")}
                        for w in (task_problems
                                  + self._shadowing_warnings(repo_root, task, bundle)
                                  + self.binding_warnings(
                                      removed=compiled["deleted"],
                                      added={n: self.pending_adds[n]["category"]
                                             for n in compiled["added"]
                                             if n in self.pending_adds}))
                    ],
                    # The browser owns the geometric checks (it has the
                    # meshes) and sends its verdict along.
                    "layout": payload.get("layout_warnings") or [],
                }

                artifacts = [{
                    "role": "scene", "path": str(scene_path),
                    "sha256": export_bundle.sha256_text(
                        scene_text(prepare_scene_document(compiled["scene"]))),
                }]
                if camera_text is not None:
                    artifacts.append({
                        "role": "cameras",
                        "path": str(bundle["camera_snapshot"] or
                                    (cameras and cameras["path"])),
                        "sha256": export_bundle.sha256_text(camera_text),
                    })
                if task_text is not None:
                    artifacts.append({
                        "role": "task",
                        "path": str(bundle["task_snapshot"] or task["path"]),
                        "sha256": export_bundle.sha256_text(task_text),
                    })

                manifest = export_bundle.build_manifest(
                    scene={
                        "path": str(scene_path),
                        "source": str(self.scene_json_path),
                        "scene_name": self.scene_name,
                        "moved": compiled["changed"],
                        "added": compiled["added"],
                        "removed": compiled["deleted"],
                        "promoted": bool(promote and not settling and not dry_run),
                        "promotion_deferred": bool(promote and settling),
                        # Recorded because it decides where props come to
                        # rest; under a Gaussian room it is the only
                        # collision geometry.
                        "ground_plane": plan["ground"],
                        "ground_plane_change": compiled["ground_plane"],
                    },
                    cameras=cameras or {},
                    task=task,
                    command=command,
                    warnings=warnings,
                    artifacts=artifacts,
                    export={
                        "id": export_bundle.export_id(scene_path),
                        # What the export will do, not what this call did; a
                        # review has no snapshot names of its own.
                        "configs": "snapshot" if snapshot else "referenced",
                        "dry_run": dry_run,
                    },
                )

                if dry_run:
                    # Built here, sent outside the lock: a stalled reader must
                    # not hold the write lock.
                    review = self._export_response(
                        manifest, None, scene_path, dry_run=True,
                        settle={"state": "disabled"})
                else:
                    # --- publish: all of it, or none of it -----------------
                    out, promoted_digest = self._publish_scene(
                        compiled["scene"], promote=promote and not settling,
                        staged=staged, out=scene_path)
                    if camera_text is not None:
                        self._publish_cameras(camera_text, bundle,
                                              write_live=bool(camera_edits),
                                              staged=staged)
                    if task_text is not None and bundle["task_snapshot"] is not None:
                        self._stage_write(bundle["task_snapshot"], task_text,
                                          expect=None, staged=staged)
                    manifest_file = export_bundle.manifest_path(out)
                    self._stage_write(manifest_file,
                                      json.dumps(manifest, indent=2, allow_nan=False),
                                      expect=None, staged=staged)

                    revision = self.adopt_written_scene(
                        compiled["scene"],
                        promoted_digest=promoted_digest,
                        promoted_path=(latest_path(self.scene_json_path)
                                       if promoted_digest is not None else None),
                    )
                    # These imports are on disk now; the switch guard must
                    # stop offering to discard them.
                    EditorHandler.saved_adds = set(compiled["added"])
                    if camera_edits:
                        EditorHandler.camera_revision += 1
                        self._adopt_written_cameras()

                    settle = {"state": "disabled"}
                    if settling:
                        settle = {"state": "running",
                                  "id": self.settle_runner.start(out, promote)}
        except HttpRefusal as refusal:
            self._rollback(staged)
            self._json(refusal.code, refusal.body)
            return
        except TargetChanged as e:
            recovered = self._rollback(staged)
            self._json(409, {"error": str(e), "reason": "target_changed",
                             "recovered": recovered})
            return
        except SceneEditError as e:
            recovered = self._rollback(staged)
            self._json(400, {"error": str(e), "recovered": recovered})
            return
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            recovered = self._rollback(staged)
            # Report a partial write out loud.
            self._json(500, {
                "error": f"{type(e).__name__}: {e}",
                "recovered": recovered,
                "partial_write": bool(staged) and not recovered.get("complete", True),
            })
            return

        if review is not None:
            self._json(200, review)
            return

        print(f"[export] {out.name}")
        for line in export_bundle.summarise(manifest):
            print(f"  {line}")
        self._json(200, self._export_response(
            manifest, manifest_file, out, dry_run=False, settle=settle,
            revision=revision))

    def _export_response(self, manifest, manifest_file, scene_path, *, dry_run,
                         settle, revision=None):
        """The body both halves of `_export` return, built once."""
        return {
            "dry_run": dry_run,
            # Only an exported command names a file that exists. The review is
            # for reading, not for running.
            "command_runnable": not dry_run,
            "manifest": manifest,
            "manifest_path": str(manifest_file) if manifest_file else None,
            "output_dir": str(Path(scene_path).parent),
            "tasks": self._associated_tasks(),
            "settle": settle,
            "scene_revision": self.scene_revision if revision is None else revision,
            "camera_revision": self.camera_revision,
        }

    def _export_targets(self, repo_root, scene_path, task, *, snapshot):
        """Where this export's config snapshots go, and what selects them.

        Returns:
            dict: ``camera_snapshot``/``camera_cfg`` and
            ``task_snapshot``/``task_group`` -- paths and the Hydra overrides
            that name them -- plus ``snapshotting``. All None when snapshotting
            is off, in which case the command names the shared configs and the
            manifest says so.
        """
        empty = {"camera_snapshot": None, "camera_cfg": None,
                 "task_snapshot": None, "task_group": None, "snapshotting": False}
        if not snapshot:
            return empty
        export = export_bundle.export_id(scene_path)
        result = dict(empty, snapshotting=True)
        if self.cameras:
            stem = Path(self.camera_source_path).stem
            path, group = export_bundle.snapshot_path(
                repo_root, export_bundle.CAMERA_GROUP_ROOT, f"{stem}_{export}")
            result["camera_snapshot"], result["camera_cfg"] = path, group
        if task:
            stem = Path(task["path"]).stem
            path, group = export_bundle.snapshot_path(
                repo_root, export_bundle.TASK_GROUP_ROOT, f"{stem}_{export}")
            result["task_snapshot"], result["task_group"] = path, group
        return result

    def _shadowing_warnings(self, repo_root, task, bundle):
        """Warn when the command's ``task=`` is not the config that would load."""
        if not task:
            return []
        group = bundle["task_group"] or task["group"]
        shadow = export_bundle.shadowing_task_config(repo_root, task.get("name"), group)
        if shadow is None:
            return []
        return [{
            "kind": "task_config_shadowed",
            "effect": "changes_task",
            "severity": "changes_task",
            "task": task.get("name"),
            "group": group,
            "where": "task",
            "message": (
                f"the run will load {shadow} rather than {group}: the eval stage "
                f"resolves task/<task_name>.yaml before the group it was given, "
                f"and this config's task_name is {task.get('name')!r}"),
            "detail": "the evaluation runs a different task config from the one "
                      "this export names",
        }]

    def _stage_write(self, path, text, *, expect, staged):
        """Publish one file and remember how to take it back."""
        previous = None
        target = Path(path)
        if target.exists():
            previous = target.read_bytes()
        guarded_write_text(target, text, expect=expect)
        staged.append((target, previous))

    def _publish_cameras(self, text, bundle, *, write_live, staged):
        """Write the export's camera snapshot, and the live config when edited."""
        if bundle["camera_snapshot"] is not None:
            self._stage_write(bundle["camera_snapshot"], text, expect=None, staged=staged)
        if not write_live:
            return
        target = Path(self.camera_out_path)
        previous = target.read_bytes() if target.exists() else None
        guarded_write_text(
            target, text, expect=self.camera_digests.get(str(target.resolve())))
        staged.append((target, previous))
        # The new digest is not recorded here: a later artifact can still fail
        # and roll this file back. `_adopt_written_cameras` reads it off the
        # file once the whole export has landed.

    def _adopt_written_cameras(self):
        """Re-read the camera config an export wrote as the served placement."""
        try:
            (EditorHandler.cameras,
             EditorHandler.camera_document) = load_cameras(Path(self.camera_out_path))
            EditorHandler.camera_source_path = Path(self.camera_out_path)
            EditorHandler.camera_resumed = True
        except SceneEditError as e:
            print(f"[export] WARNING: could not re-read {self.camera_out_path}: {e}")
        # Read off the file so a rollback leaves the expectation describing
        # what is actually there.
        target = Path(self.camera_out_path)
        EditorHandler.camera_digests[str(target.resolve())] = file_digest(target)

    @staticmethod
    def _rollback(staged):
        """Undo the files an export published before it failed.

        Returns:
            dict: ``removed``, ``restored``, ``failed`` and ``complete``.
        """
        removed, restored, failed = [], [], []
        for path, previous in reversed(staged):
            try:
                if previous is None:
                    Path(path).unlink(missing_ok=True)
                    removed.append(str(path))
                else:
                    atomic_write_text(path, previous.decode("utf-8"))
                    restored.append(str(path))
            except (OSError, UnicodeDecodeError) as e:
                failed.append(f"{path}: {e}")
        staged.clear()
        return {"removed": removed, "restored": restored, "failed": failed,
                "complete": not failed}

    # Directories this server will hand to the desktop file manager: an
    # allowlist of the places this editor actually writes.
    def _openable_dirs(self):
        candidates = [Path(self.scene_json_path).parent]
        if self.camera_out_path:
            candidates.append(Path(self.camera_out_path).parent)
        if self.data_dir:
            candidates.append(Path(self.data_dir))
        return {str(Path(c).resolve()): Path(c).resolve() for c in candidates}

    def _open_folder(self):
        """Show an output directory in the desktop file manager.

        Only directories this editor writes to, never a file, and never an
        argument that reaches a shell. A non-loopback bind refuses: the
        window would open on the server's desktop.
        """
        payload = self._read_json_request()
        if payload is None:
            return
        if self.bind_host not in ("127.0.0.1", "localhost", ""):
            self._json(400, {
                "error": "the server is not bound to loopback, so this would open a "
                         "window on the machine running it rather than on yours",
            })
            return

        wanted = payload.get("path")
        if not isinstance(wanted, str):
            self._json(400, {"error": "path must be a string"})
            return
        allowed = self._openable_dirs()
        target = allowed.get(str(Path(wanted).resolve()) if wanted else "")
        if target is None:
            self._json(403, {
                "error": "that directory is not one this editor writes to",
                "openable": sorted(allowed),
            })
            return
        if not target.is_dir():
            self._json(404, {"error": f"no such directory: {target}"})
            return

        opener = shutil.which("xdg-open") or shutil.which("open")
        if opener is None:
            self._json(501, {
                "error": "no xdg-open on this machine; copy the path instead",
                "path": str(target),
            })
            return
        try:
            # No shell, one fixed argument, output discarded so a chatty file
            # manager cannot wedge on a full pipe.
            subprocess.Popen(
                [opener, str(target)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as e:
            self._json(500, {"error": f"could not open {target}: {e}"})
            return
        print(f"[open] {target}")
        self._json(200, {"opened": str(target)})

    # Preview proxies are named apart from imported ones so they can never be
    # mistaken for scene objects.
    PREVIEW_PREFIX = "preview_"

    # Asset-derived facts per preview .glb, keyed by filename.
    _preview_facts = {}

    def _inspect(self):
        """Describe an asset without importing it.

        Nothing here touches the scene document, ``pending_adds`` or the
        served manifest; the proxy it builds is a preview.
        """
        payload = self._read_json_request()
        if payload is None:
            return

        key = payload.get("key")
        raw_path = payload.get("path")
        try:
            if isinstance(key, str) and key:
                asset = resolve_key(key, self.list_assets())
                if asset is None:
                    self._json(404, {"error": "unknown asset key; reload the library"})
                    return
                source = Path(asset["usd"])
                kind = asset.get("kind") or classify(source)
                label = asset["asset_id"]
                category = asset["category"]
            elif isinstance(raw_path, str) and raw_path:
                source = resolve_user_path(raw_path)
                kind = classify(source)
                if kind == "directory":
                    self._json(400, {"error": "that is a directory, not an asset"})
                    return
                label = source.stem
                category = import_slug(source.stem)
            else:
                self._json(400, {"error": "send either an asset key or a path"})
                return

            options = {
                "scale": payload.get("mesh_scale", 1.0),
                "up_axis": payload.get("up_axis", "auto"),
                "collision": payload.get("collision", "convexHull"),
                "mass": payload.get("mass"),
            }
            facts = (describe_mesh(source, **options) if kind == "mesh"
                     else describe_usd(source))

            from extract import build_proxy

            # Keyed by the source and the options that change its geometry.
            digest = hashlib.sha256(
                f"{source}|{options['scale']}|{options['up_axis']}".encode("utf-8")
            ).hexdigest()[:16]
            glb_name = f"{self.PREVIEW_PREFIX}{digest}.glb"
            if not (Path(self.data_dir) / glb_name).exists():
                if kind == "mesh":
                    # Convert into a throwaway bundle so the preview matches
                    # what an import would write; the scene directory is
                    # untouched.
                    with tempfile.TemporaryDirectory(prefix="preview_") as staging:
                        converted, _ = convert_mesh(
                            source, Path(staging) / "b", import_slug(label), **options)
                        proxy = build_proxy(
                            converted, self.data_dir, glb_name, self.textures)
                else:
                    proxy = build_proxy(source, self.data_dir, glb_name, self.textures)
                if proxy["glb"] is None:
                    self._json(422, {"error": f"{label}: {proxy['error']}"})
                    return
                # Remembered beside the .glb so a warm open quotes the same
                # size as a cold one.
                EditorHandler._preview_facts[glb_name] = {
                    "nativeSize": proxy["nativeSize"],
                    "scaleFidelity": proxy["scaleFidelity"],
                }
            else:
                proxy = {"glb": glb_name, **EditorHandler._preview_facts.get(glb_name, {})}
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return
        except Exception as e:  # noqa: BLE001 - a malformed asset must not kill the server
            traceback.print_exc()
            self._json(422, {"error": f"could not read {Path(str(source)).name}: "
                                      f"{type(e).__name__}: {e}"})
            return

        self._json(200, {
            "kind": kind,
            "asset_id": label,
            "category": category,
            "source": str(source),
            "glb": proxy["glb"],
            # The measurement the size row will show once placed, not the
            # USD's authored bound (a stale `extent` disagrees).
            "native_size": proxy.get("nativeSize"),
            "scale_fidelity": proxy.get("scaleFidelity"),
            "facts": facts,
        })

    def _plan_scene_write(self, payload):
        """Validate a scene-write request without writing anything.

        Shared by ``/api/save`` and ``/api/export`` so the two cannot disagree
        about what a valid edit is.

        Raises:
            HttpRefusal: With the status and body to return.
        """
        edits = payload.get("edits", {})
        promote = payload.get("promote_latest", False)
        # Not bool(): the JSON string "false" is truthy, and this flag decides
        # whether _latest gets overwritten.
        if not isinstance(promote, bool):
            raise HttpRefusal(400, {"error": "promote_latest must be a JSON boolean"})
        if not isinstance(edits, dict):
            raise HttpRefusal(400, {"error": "bad request: edits must be an object"})
        removed = payload.get("remove", [])
        if not isinstance(removed, list) or not all(isinstance(n, str) for n in removed):
            raise HttpRefusal(400, {"error": "remove must be a list of object names"})
        removed = set(removed)

        # Absent means "leave the ground plane as the source scene has it";
        # an explicit null means remove the block.
        if "ground_plane" in payload:
            try:
                ground = validate_ground_plane(payload["ground_plane"])
            except SceneEditError as e:
                raise HttpRefusal(400, {"error": str(e)}) from e
        else:
            ground = read_ground_plane(self.base_scene or {})

        if not self._scene_hash_accepted(payload.get("base_scene_sha256")):
            raise HttpRefusal(
                409, {"error": "scene changed since extraction; reload the editor"})
        # A stale tab must not overwrite a newer save; clients that send no
        # revision are refused rather than trusted.
        if payload.get("scene_revision") != self.scene_revision:
            raise HttpRefusal(409, {
                "error": "the scene changed in another tab since this page loaded; "
                         "reload to continue",
                "scene_revision": self.scene_revision,
            })
        if payload.get("complete_snapshot") is not True:
            raise HttpRefusal(
                400, {"error": "save requires a complete editable-object snapshot"})
        # Every posable object must be either kept-with-a-pose or explicitly
        # removed, so a client bug cannot delete by omission. Robot removal
        # is refused later, by remove_objects's own editable_names check.
        overlap = removed & set(edits)
        missing = self.posable_names - set(edits) - removed
        extra = (set(edits) | removed) - self.posable_names
        if missing or extra or overlap:
            details = []
            if missing:
                details.append(f"missing: {', '.join(sorted(missing))}")
            if extra:
                details.append(f"locked/unknown: {', '.join(sorted(extra))}")
            if overlap:
                details.append(f"both kept and removed: {', '.join(sorted(overlap))}")
            raise HttpRefusal(
                400, {"error": "snapshot object mismatch (" + "; ".join(details) + ")"})
        return {"edits": edits, "removed": removed, "promote": promote, "ground": ground}

    def _compile_scene(self, plan):
        """Build the document a plan describes. Call under WRITE_LOCK.

        Raises:
            HttpRefusal: If the source scene changed underneath this session.
        """
        # The origin check in `_plan_scene_write` says nothing about the file
        # on disk; that is this separate comparison.
        try:
            disk_sha = scene_sha256(self.scene_json_path)
        except OSError as e:
            raise HttpRefusal(500, {"error": f"source scene unreadable: {e}"}) from e
        if disk_sha != self.expected_source_sha256:
            raise HttpRefusal(409, {
                "error": "the source scene changed on disk since this editor "
                         "started; reload to pick up the new revision",
                "disk_sha256": disk_sha,
                "expected_sha256": self.expected_source_sha256,
            })

        removed = plan["removed"]
        # Compile every export from the immutable startup scene. The browser
        # sends all editable transforms, so a later save cannot drop an edit
        # made during an earlier save.
        scene = copy.deepcopy(self.base_scene)
        # Imports first: an added object must exist before apply_edits can
        # pose it; a removed pending add is simply never added.
        added = add_objects(
            scene,
            [spec for name, spec in self.pending_adds.items() if name not in removed],
        )
        deleted = remove_objects(
            scene, sorted(removed - set(self.pending_adds)),
            # A room (or its default robot) is removable only once a
            # replacement is pending; remove_objects refuses a bare removal.
            removable_names=self.editable_names | self.background_removed | self.robot_removed,
        )
        changed = apply_edits(
            scene, plan["edits"], asset_facts=self.asset_facts(),
            editable_names=self.editable_names, posable_names=self.posable_names,
            scalable_names=self.scalable_names,
        )
        ground = apply_ground_plane(scene, plan["ground"])
        return {
            "scene": scene, "added": added, "deleted": deleted, "changed": changed,
            "ground_plane": ground,
        }

    def _publish_scene(self, scene, *, promote, staged=None, out=None):
        """Write a compiled scene, and optionally promote it. Call under WRITE_LOCK.

        Both files carry byte-identical text, and the digest handed back is
        the digest of exactly those bytes.

        Args:
            scene (dict): The compiled document.
            promote (bool): Also overwrite ``_scene_state_latest.json``.
            staged (list or None): Collects ``(path, previous_bytes_or_None)``
                for every file written, for rollback.
            out (Path or None): The name to write. An export chooses it up
                front so its manifest can name the file before anything is
                published.

        Returns:
            tuple[Path, str or None]: The timestamped file, and the digest now
            in ``_latest`` when this promoted (None when it did not).

        Raises:
            TargetChanged: If ``_latest`` no longer holds what this session
                last put there. The timestamped file is written either way,
                so no work is lost.
        """
        source = Path(self.scene_json_path)
        document = prepare_scene_document(scene)
        text = scene_text(document)
        mode = scene_file_mode(source)
        out = Path(out) if out is not None else scene_output_path(source)
        guarded_write_text(out, text, expect=None, mode=mode)
        if staged is not None:
            staged.append((out, None))
        if not promote:
            return out, None
        target = latest_path(source)
        previous = target.read_bytes() if target.exists() else None
        digest = promote_scene_text(
            text, source, expect=self.expected_latest_sha256, mode=mode)
        if staged is not None:
            staged.append((target, previous))
        return out, digest

    @classmethod
    def table_state(cls):
        """Where this room's table centre is, and what the scan guessed.

        ``centre`` is human-placed and authoritative. ``estimate`` is the
        scan's guess, offered only to seed the marker; it reports its own
        extent so it can be disbelieved.
        """
        if cls.background_row is None:
            return {"room": None, "centre": None, "estimate": None}
        return {
            "room": cls.background_row["id"],
            "label": cls.background_row["label"],
            "sidecar": str(cls.background_row["sidecar"]),
            "centre": background_table_centre(cls.background_row),
            "estimate": cls.table_estimate,
        }

    @classmethod
    def background_kind(cls):
        """What kind of room this scene has: ``splat``, ``mesh`` or None.

        Read off the manifest: only opening the USD tells the two apart, and
        the extractor has already done that.
        """
        for entry in (cls.manifest or {}).get("objects", []):
            if entry.get("kind") != "background":
                continue
            return "splat" if entry.get("splat") else "mesh"
        return None

    @classmethod
    def ground_plane_state(cls):
        """The ground plane as the scene on disk has it, plus what to suggest.

        ``plane`` is the saved baseline the browser measures edits against;
        ``table_height`` comes from the room's registered table centre when
        there is one.
        """
        table = background_table_centre(cls.background_row) if cls.background_row else None
        return {
            "plane": read_ground_plane(cls.base_scene or {}),
            "background": cls.background_kind(),
            "table_height": None if table is None else float(table[2]),
        }

    def _save_table(self):
        """Record the table centre for this room, in the room's own sidecar.

        Per room rather than per scene: every scene shot here wants the same
        point.
        """
        if self.background_row is None:
            self._json(404, {"error": "this scene has no scanned room to place a table in"})
            return

        payload = self._read_json_request()
        if payload is None:
            return

        try:
            with WRITE_LOCK:
                path = write_table_centre(
                    self.background_row,
                    payload.get("centre"),
                    estimate=self.table_estimate,
                )
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return
        except OSError as e:
            self._json(500, {"error": f"could not write {type(e).__name__}: {e}"})
            return

        print(f"[table] centre for {self.background_row['id']} -> {path}")
        self._json(200, self.table_state())

    @classmethod
    def camera_config_catalog(cls):
        """Every rig config on disk, so one can be imported without a restart.

        Each row carries its sensor names: the filename says which room a
        placement was aimed in, not which cameras it aims.
        """
        directory = Path(HERE.parents[2]) / CAMERA_CFG_SUBDIR
        rows = []
        for path in sorted(directory.glob("*.yaml")) if directory.is_dir() else []:
            row = {"name": path.stem, "path": str(path), "sensors": [], "error": None}
            try:
                cameras, _ = load_cameras(path)
                row["sensors"] = [c["name"] for c in cameras]
            except (SceneEditError, OSError, ValueError) as e:
                # Listed anyway, with the reason: a config that cannot load is
                # exactly the one somebody needs to be told about.
                row["error"] = str(e)
            rows.append(row)
        current = cls.camera_source_path
        return {
            "configs": rows,
            "directory": str(directory),
            "current": Path(current).stem if current else None,
            "out_name": Path(cls.camera_out_path).stem if cls.camera_out_path else None,
        }

    def _load_cameras(self):
        """Import a different rig config, replacing the one in the browser.

        Refused while camera edits are pending: importing discards the poses
        in the viewport.
        """
        payload = self._read_json_request()
        if payload is None:
            return
        if payload.get("dirty") and not payload.get("discard"):
            self._json(409, {
                "error": "the cameras have unsaved changes; save them or confirm discard",
                "needs_confirm": True,
            })
            return

        repo_root = Path(HERE.parents[2])
        try:
            with WRITE_LOCK:
                template = resolve_camera_config(payload.get("name") or "", repo_root)
                cameras, document = load_cameras(template)
                out_path, source = camera_config_paths(
                    repo_root, template,
                    background=self.camera_background,
                    scene_name=self.scene_name,
                    # An explicit import means *this* file, not whatever
                    # placement happens to be remembered for the room.
                    explicit_out=Path(template).stem,
                )
                EditorHandler.cameras = cameras
                EditorHandler.camera_document = document
                EditorHandler.camera_source_path = Path(template)
                EditorHandler.camera_template_path = Path(template)
                EditorHandler.camera_out_path = out_path
                EditorHandler.camera_digests = {
                    str(Path(out_path).resolve()): file_digest(out_path)}
                EditorHandler.camera_resumed = False
                EditorHandler.camera_observation = observation_key_map(
                    [c["name"] for c in cameras], repo_root,
                    sensors_cfg=Path(template).stem,
                    task_cfg=None,
                )
                # A stale tab holding the old rig must be refused, not merged.
                EditorHandler.camera_revision += 1
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return
        except (OSError, ValueError) as e:
            self._json(400, {"error": f"{type(e).__name__}: {e}"})
            return

        print(f"[cameras] imported {Path(template).name} "
              f"({len(cameras)} sensor(s)) -> saves to {out_path.name}")
        self._json(200, {"ok": True, "name": Path(template).stem,
                         "sensors": [c["name"] for c in cameras],
                         "camera_revision": self.camera_revision})

    def _save_cameras(self):
        """Write edited camera poses to a new per-scene external_sensors config.

        The source config is never modified: cameras are a robot-rig property
        shared by every scene, so an edit authored against one scene must not
        silently retarget the others.
        """
        if not self.cameras:
            self._json(404, {"error": "no camera config loaded (start with --cameras)"})
            return

        payload = self._read_json_request()
        if payload is None:
            return

        try:
            with WRITE_LOCK:
                # Revision compared inside the lock; clients that send none
                # are refused rather than trusted.
                if payload.get("camera_revision") != self.camera_revision:
                    raise HttpRefusal(409, {
                        "error": "cameras changed since this page loaded; reload to continue",
                        "camera_revision": self.camera_revision,
                    })
                edits = validate_camera_edits(self.cameras, payload.get("edits", {}))
                document = copy.deepcopy(self.camera_document)
                changed = apply_camera_edits(document, edits)
                # An explicit name exports beside the room-keyed default, not
                # instead of it.
                out_path = Path(self.camera_out_path)
                requested = payload.get("out_name")
                if requested:
                    out_path = camera_export_path(Path(HERE.parents[2]), requested)
                key = str(out_path.resolve())
                if key in self.camera_digests:
                    # A file this session wrote or was bound to; the digest
                    # only catches outside changes.
                    expect = self.camera_digests[key]
                else:
                    expect = file_digest(out_path)
                    if expect is not None and payload.get("overwrite_sha256") != expect:
                        raise HttpRefusal(409, {
                            "error": f"{out_path.name} already exists and was not written "
                                     "by this session; confirm the overwrite or choose "
                                     "another name",
                            "reason": "exists",
                            "sha256": expect,
                            "path": str(out_path),
                        })
                text = camera_config_text(
                    document, self.camera_source_path, self.scene_name,
                    background=self.camera_background, cfg_name=out_path.stem,
                )
                digest = guarded_write_text(out_path, text, expect=expect)
                out = out_path
                EditorHandler.camera_revision += 1
                EditorHandler.camera_digests[key] = digest
                # Adopt what was written as the served state, so a reload
                # shows it.
                try:
                    (EditorHandler.cameras,
                     EditorHandler.camera_document) = load_cameras(out)
                    # Only for the room-keyed target: a named export must not
                    # retarget the placement this session resumes from.
                    if out == Path(self.camera_out_path):
                        EditorHandler.camera_source_path = out
                        EditorHandler.camera_resumed = True
                except SceneEditError as e:
                    # The write itself succeeded, so this is a warning.
                    print(f"[cameras] WARNING: could not re-read {out.name}: {e}")
        except HttpRefusal as refusal:
            self._json(refusal.code, refusal.body)
            return
        except TargetChanged as e:
            self._json(409, {"error": str(e), "reason": "target_changed",
                             "sha256": file_digest(e.path), "path": str(e.path)})
            return
        except SceneEditError as e:
            self._json(400, {"error": str(e)})
            return
        except Exception as e:
            self._json(500, {"error": str(e)})
            return

        print(f"[cameras] {len(changed)} camera(s) -> {out}")
        self._json(200, {
            "changed": changed,
            "path": str(out),
            "cfg_name": Path(out).stem,
            "background": self.camera_background,
            "camera_revision": self.camera_revision,
        })


def scene_data_dir(scene_json):
    """Extraction directory for one scene.

    Per-scene so two servers cannot overwrite each other's geometry. The stem
    alone is not unique across scene directories, so a digest of the resolved
    path is appended.
    """
    resolved = Path(scene_json).resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return HERE / "web" / "data" / f"{resolved.stem}-{digest}"


def validate_cached_manifest(scene_json, data_dir, splat_budget=DEFAULT_SPLAT_BUDGET):
    """Refuse --no-extract when the cache belongs to another scene revision."""
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        raise SceneEditError(f"cached manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cached_scene = Path(manifest.get("scene_json", "")).resolve()
    if cached_scene != scene_json:
        raise SceneEditError(
            f"cached manifest is for {cached_scene}, not {scene_json}; remove --no-extract"
        )
    # `source_scene_sha256` is the currency digest; `base_scene_sha256` is
    # the fallback for older manifests.
    cached_digest = manifest.get("source_scene_sha256") or manifest.get("base_scene_sha256")
    if cached_digest != scene_sha256(scene_json):
        raise SceneEditError("cached manifest is stale; remove --no-extract")

    # A cache written by an older extractor must not be reused. Imported
    # lazily: extract pulls in pxr and trimesh.
    from extract import extractor_version  # noqa: PLC0415

    if manifest.get("extractor_version") != extractor_version():
        raise SceneEditError(
            "cached extraction was written by a different version of extract.py; "
            "remove --no-extract to rebuild it"
        )

    # A changed --splat-budget invalidates the cache too.
    for entry in manifest.get("objects", []):
        if entry.get("splat") and entry.get("splatBudget") != (splat_budget or None):
            raise SceneEditError(
                "cached extraction used a different --splat-budget; remove --no-extract"
            )

    # Assets edited since extraction would be served from stale proxies.
    for entry in manifest.get("objects", []):
        source = entry.get("sourceUsd")
        recorded = entry.get("sourceMtimeNs")
        if not source or recorded is None:
            continue
        try:
            current = Path(source).stat().st_mtime_ns
        except OSError:
            raise SceneEditError(
                f"cached proxy references a missing asset ({source}); remove --no-extract"
            ) from None
        if current != recorded:
            raise SceneEditError(
                f"{Path(source).name} changed since extraction; remove --no-extract"
            )
    return manifest


class EditorOptions:
    """Startup choices that outlive the scene they were first applied to.

    Gathered once and handed to :func:`bind_scene` on every switch.
    """

    def __init__(self, args, repo_root):
        self.repo_root = Path(repo_root)
        self.textures = not args.no_textures
        self.allow_incomplete = args.allow_incomplete
        self.splat_budget = getattr(args, "splat_budget", DEFAULT_SPLAT_BUDGET)
        self.dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
        self.cameras = args.cameras
        self.cameras_out = args.cameras_out
        self.task_cfg = args.task_cfg
        # An explicit --asset-root pins the library for the whole session; left
        # unset, each scene gets the roots that make sense for it.
        self.asset_root = list(args.asset_root) if args.asset_root else None
        self.extra_scene_roots = list(args.scene_root or [])

    @property
    def dataset_root(self):
        """OmniGibson's ``gm.DATA_PATH``, defaulted.

        ``dataset_dir`` stays None when the user did not pass one, so callers
        can tell "not given" from "given the default".
        """
        return self.dataset_dir or self.repo_root / DEFAULT_DATASET_DIR

    @property
    def robot_asset_root(self):
        """Root of ``omnigibson-robot-assets``."""
        return self.repo_root / DEFAULT_ROBOT_ASSET_DIR


def bind_scene(scene_json, options, *, reuse_cache="prefer", announce=print):
    """Point the running editor at *scene_json*.

    Everything that can fail happens before a single class attribute moves,
    so a failed switch leaves the editor exactly where it was.

    Args:
        scene_json (Path): Scene state document to open.
        options (EditorOptions): Session-wide choices.
        reuse_cache (str): ``require`` (``--no-extract``: fail if the cached
            extraction does not match), ``prefer`` (reuse it when it is valid,
            otherwise extract) or ``never``.
        announce (callable): Where progress lines go.

    Returns:
        dict: Summary of what was bound.

    Raises:
        SceneEditError: If the scene cannot be read, extracted, or is missing
            visual proxies without ``--allow-incomplete``.
    """
    from extract import extract, write_manifest

    scene_json = Path(scene_json).resolve()
    if not scene_json.is_file():
        raise SceneEditError(f"scene JSON not found: {scene_json}")

    # Read first: the cached manifest is pruned against this document below.
    base_scene = load_scene(scene_json)
    known_objects = set(base_scene.get("objects_info", {}).get("init_info", {}) or {})

    data_dir = scene_data_dir(scene_json)
    manifest = None
    if reuse_cache in ("require", "prefer"):
        try:
            manifest = validate_cached_manifest(
                scene_json, data_dir,
                getattr(options, "splat_budget", DEFAULT_SPLAT_BUDGET))
        except (SceneEditError, OSError, ValueError, json.JSONDecodeError) as e:
            if reuse_cache == "require":
                raise SceneEditError(str(e)) from None
            manifest = None
        if manifest is not None:
            # Drop entries the scene document does not have (a previous run's
            # imports, or objects a promotion removed): the browser must not
            # show objects the save endpoint would reject as unknown.
            stale = [entry for entry in manifest["objects"]
                     if entry["name"] not in known_objects]
            if stale:
                manifest["objects"] = [e for e in manifest["objects"]
                                       if e["name"] in known_objects]
                announce(f"Dropped {len(stale)} object(s) the scene no longer has from "
                         "the cached manifest: " + ", ".join(e["name"] for e in stale))
            else:
                announce(f"Reusing the cached extraction of {scene_json.name}")
            # Re-read the authored half from the file being bound; the cached
            # manifest describes what the previous session last served.
            apply_authored_state_to_manifest(manifest, base_scene)
            # The two digests are the same value only at the moment of
            # binding: `base_scene_sha256` is the origin identity (never
            # moves), `source_scene_sha256` the file revision (moves on
            # promotion).
            digest = scene_sha256(scene_json)
            manifest["base_scene_sha256"] = digest
            manifest["source_scene_sha256"] = digest
            write_manifest(manifest, data_dir)

    if manifest is None:
        announce(f"Extracting {scene_json.name}")
        manifest = extract(
            scene_json,
            data_dir,
            options.robot_asset_root,
            textures=options.textures,
            dataset_dir=options.dataset_root,
            splat_budget=getattr(options, "splat_budget", DEFAULT_SPLAT_BUDGET),
        )

    # Placing an object whose proxy is missing means authoring blind, so this
    # stops rather than warning and carrying on.
    degraded = [
        entry.get("name", "?")
        for entry in manifest.get("objects", [])
        if entry.get("status") not in (None, "ok")
    ]
    if degraded and not manifest.get("complete", True):
        summary = ", ".join(degraded[:8]) + (" ..." if len(degraded) > 8 else "")
        if not options.allow_incomplete:
            raise SceneEditError(
                f"{len(degraded)} object(s) have no usable visual proxy: {summary}. "
                "Editing a scene you cannot fully see risks placing objects blind — "
                "fix the asset paths, or restart with --allow-incomplete."
            )
        announce(f"WARNING: {len(degraded)} object(s) lack a usable proxy: {summary}")
        announce("         --allow-incomplete is set; these show as error rows.")

    scene_name = scene_json.stem.split("_scene_state_")[0]
    background = background_id(base_scene, scene_json)

    # Cameras are keyed by the scanned room, so a switch re-resolves them.
    camera_state = _resolve_cameras(scene_name, background, options, announce)

    # Robot-carried cameras; a failure here costs only the wrist preview.
    try:
        robot_cams, robot_obs = robot_cameras.scene_robot_cameras(
            base_scene, scene_json, options.robot_asset_root, announce)
    except Exception as e:  # noqa: BLE001
        announce(f"Robot cameras: unavailable ({type(e).__name__}: {e})")
        robot_cams, robot_obs = [], {}
    for camera in robot_cams:
        seen = robot_obs.get(camera["name"], {})
        flag = "" if seen.get("certain", True) else "  <-- more than one; order unread"
        announce(f"Cameras:   {camera['name']} -> {seen.get('key', '?')} "
                 f"(from the robot asset, read-only){flag}")

    asset_roots = tuple(
        Path(r).resolve() for r in (options.asset_root or default_roots(scene_json))
    )
    shortcuts, seen = [], set()
    for label, path in ([("scene folder", scene_json.parent)]
                        + [(r.name or str(r), r) for r in asset_roots]
                        + [("home", Path.home())]):
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        shortcuts.append((label, resolved))

    # --- everything above can fail; nothing below does ---------------------
    EditorHandler.data_dir = data_dir
    EditorHandler.manifest = manifest
    EditorHandler.textures = options.textures
    EditorHandler.scene_name = scene_name
    # Origin document, digests and revision, reset together.
    EditorHandler.adopt_scene_binding(scene_json, base_scene)
    EditorHandler.editable_names = editable_object_names(base_scene, scene_json)
    # Fixed after binding, unlike editable_names, which imports extend.
    EditorHandler.robot_names = robot_object_names(base_scene, scene_json)
    EditorHandler.background_names = background_object_names(base_scene, scene_json)
    EditorHandler.asset_roots = asset_roots
    EditorHandler.browse_default = scene_json.parent
    EditorHandler.browse_shortcuts = tuple(shortcuts)
    # Per-scene session state, reset so a switch cannot carry the previous
    # scene's pending imports across.
    EditorHandler.pending_adds = {}
    # A pending room (or default-robot) swap belongs to the scene it was made in.
    EditorHandler.background_removed = set()
    EditorHandler.robot_removed = set()
    EditorHandler.saved_adds = set()
    EditorHandler._assets_cache = None
    EditorHandler._added_counter = 0

    # A mis-parse degrades to "no task warnings" rather than an exit.
    try:
        EditorHandler.task_configs = tuple(
            task_bindings.discover_task_configs(HERE.parents[2]))
        EditorHandler.known_scene_names = frozenset(
            task_bindings.known_scene_names(HERE.parents[2]))
        bound = EditorHandler.binding_warnings()
        print(f"Task bindings: {len(EditorHandler.task_configs)} config(s) read"
              + (f"; {len(bound)} already unsatisfied for this scene" if bound else ""))
        for warning in bound[:5]:
            print(f"  ! {warning.get('message')}")
    except Exception as e:  # noqa: BLE001
        print(f"Task bindings: unavailable ({type(e).__name__}: {e})")
    # Never reset, so a tab holding the previous scene can tell it changed.
    EditorHandler.session_revision += 1
    for attribute, value in camera_state.items():
        setattr(EditorHandler, attribute, value)
    EditorHandler.robot_cameras = tuple(robot_cams)
    EditorHandler.robot_camera_observation = robot_obs

    EditorHandler.background_row, EditorHandler.table_estimate = _resolve_table(
        base_scene, scene_json, options, announce
    )

    _announce_ground_plane(announce)
    announce(f"Editing {scene_json}")
    return {
        "scene": str(scene_json),
        "name": scene_name,
        "background": background,
        "objects": len(manifest.get("objects", [])),
        "editable": sorted(EditorHandler.editable_names),
        "degraded": degraded,
        "session_revision": EditorHandler.session_revision,
        "cameras": len(camera_state.get("cameras") or []),
    }


def _announce_ground_plane(announce):
    """Say what this scene rests on.

    A Gaussian background is loaded ``visual_only`` with no mesh prims, so a
    splat scene without a ground plane has no collision geometry at all.
    """
    state = EditorHandler.ground_plane_state()
    plane, background = state["plane"], state["background"]
    if plane is not None:
        announce(f"Ground plane: {describe_ground_plane(plane)}")
    elif background == "splat":
        announce("Ground plane: none. This scene's room is a Gaussian splat, which "
                 "has no collision geometry — add one on the Objects tab or props "
                 "will fall through the desk.")
    else:
        announce("Ground plane: none; the run config's own floor plane stands.")


def _resolve_table(scene, scene_json, options, announce):
    """Find this scene's room in the registry, and seed a table centre for it.

    Best-effort: a scene with no room, or a room with no sidecar, simply has
    no table marker.

    Returns:
        tuple[dict or None, dict or None]: (registry row, scan estimate).
    """
    try:
        records = list(iter_objects(scene, scene_json, robot_asset_dir=None,
                                    usd_facts=False))
        room = next((r for r in records if r["kind"] == "background" and r["usd"]), None)
        if room is None:
            return None, None

        roots = background_roots(scene_json, options.repo_root)
        target = os.path.realpath(room["usd"])
        row = next(
            (b for b in discover_backgrounds(roots) if os.path.realpath(b["usd"]) == target),
            None,
        )
        if row is None:
            announce(f"Table: {Path(room['usd']).stem} has no .background.json, "
                     "so its table centre cannot be remembered")
            return None, None

        # The row's own pose is the registered one; this scene may carry a
        # nudged copy, and the marker has to sit on the room as *this* scene
        # places it.
        row = dict(row, position=room["position"], orientation=room["orientation"])
        centre = background_table_centre(row)
        estimate = None
        if centre is None:
            props = [r["position"][:2] for r in records if r["editable"]]
            estimate = estimate_table(row, props) if props else None
            if estimate:
                announce(f"Table: no centre saved for {row['id']}; seeding the marker from "
                         f"the scan ({estimate['extent'][0]:.2f} x {estimate['extent'][1]:.2f} m "
                         "patch — check it)")
        else:
            announce(f"Table: centre for {row['id']} at "
                     + ", ".join(f"{v:+.3f}" for v in centre))
        return row, estimate
    except Exception as e:  # noqa: BLE001
        announce(f"Table: unavailable ({type(e).__name__}: {e})")
        return None, None


def _resolve_cameras(scene_name, background, options, announce):
    """Camera state for one scene, as a dict of EditorHandler attributes.

    Returns everything blank when the server was started without ``--cameras``,
    which is what keeps a switch from carrying a stale rig across.
    """
    blank = {
        "cameras": None, "camera_document": None, "camera_source_path": None,
        "camera_out_path": None, "camera_template_path": None,
        "camera_background": None, "camera_resumed": False,
        "camera_observation": None,
        # Nothing loaded, nothing to guard.
        "camera_digests": {},
    }
    if not options.cameras:
        return blank

    template = resolve_camera_config(options.cameras, options.repo_root)
    # Resume previously authored poses when they exist; --cameras is a
    # starting point, not an override.
    out_path, source = camera_config_paths(
        options.repo_root, template, background=background,
        scene_name=scene_name, explicit_out=options.cameras_out,
    )
    cameras, document = load_cameras(source)

    where = f"background {background}" if background else f"scene {scene_name}"
    if source == out_path:
        announce(f"Cameras: resuming the placement saved for {where} ({source.name})")
    elif source != template:
        announce(f"Cameras: found {source.name}; the next save moves it to {out_path.name}")
    else:
        announce(f"Cameras: no placement saved for {where} yet, starting from {template.name}")
    announce(f"Cameras: {len(cameras)} from {source.name} -> saves to {out_path.name}")

    # Keyed off the template stem: a saved placement is renamed
    # <background>_cameras, which no task config selects, but it is the
    # same rig being aimed.
    observation = observation_key_map(
        [camera["name"] for camera in cameras], options.repo_root,
        sensors_cfg=template.stem, task_cfg=options.task_cfg,
    )
    for camera in cameras:
        info = observation["cameras"].get(camera["name"], {})
        key = info.get("key") or "not consumed by the eval path"
        flag = "" if info.get("certain", True) else "  <-- task configs disagree"
        announce(f"Cameras:   {camera['name']} -> {key}{flag}")
    if observation["note"]:
        announce(f"Cameras: {observation['note']}")

    return {
        "cameras": cameras, "camera_document": document,
        "camera_source_path": source, "camera_out_path": out_path,
        "camera_template_path": template,
        "camera_background": background, "camera_resumed": source != template,
        "camera_observation": observation,
        # Expected digest of the file this session will write; None when it
        # does not exist yet, telling a first save apart from an overwrite.
        "camera_digests": {str(Path(out_path).resolve()): file_digest(out_path)},
    }


def main():
    parser = argparse.ArgumentParser(description="Serve the light scene editor")
    parser.add_argument("--scene", required=True, help="Path to a scene_state JSON")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind. Defaults to loopback. Use 0.0.0.0 to let a "
             "teammate on the LAN connect -- note there is no authentication, "
             "so anyone who can reach the port can overwrite scenes, and "
             "concurrent editors silently clobber each other (see README).",
    )
    parser.add_argument("--no-extract", action="store_true", help="Reuse existing web/data")
    parser.add_argument("--no_textures", action="store_true", help="Extract geometry only")
    parser.add_argument(
        "--splat-budget",
        type=int,
        default=DEFAULT_SPLAT_BUDGET,
        help="Most gaussians a Gaussian-splat background may keep in the browser; "
             f"0 keeps all of them (default: {DEFAULT_SPLAT_BUDGET:,}, about 40 MB "
             "to download and 52 MB of texture). All of them can be more than the "
             "browser will hold: the gaussians go into a texture 2048 wide, so the "
             "ceiling is the device's own max texture size -- about 5.6 M under "
             "software rendering. Over it the room is refused with the count to "
             "pass here, rather than drawn wrong or not at all. Lower it (100000 "
             "is comfortable) when running headless without a GPU, where the "
             "default renders about two frames a minute.",
    )
    parser.add_argument(
        "--allow-hosts",
        default="",
        help="Comma-separated extra Host header names to accept, for a non-loopback bind.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Serve even when some objects have no usable visual proxy. Off by default: "
             "placing an object you cannot see is how a scene silently goes wrong.",
    )
    parser.add_argument(
        "--settle-after-save",
        action="store_true",
        help="Opt in to launching OmniGibson out of process after each save",
    )
    parser.add_argument(
        "--settle-python",
        default=None,
        help="Interpreter with OmniGibson. Auto-detected from $SIMFOUNDRY_PYTHON "
             "or the conda envs if omitted.",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="external_sensors config to place cameras from — a bare name such as "
             "'nv_franka_droid' or a path.",
    )
    parser.add_argument(
        "--cameras-out",
        default=None,
        help="Name for the written config. Defaults to <background>_cameras, which is "
             "what makes a placement authored in one room come back automatically for "
             "every scene shot in that room; pass a name to override the key.",
    )
    parser.add_argument(
        "--task-cfg",
        default=None,
        help="Task config (a bare stem from scripts/cfg/) whose base_camera_*_name "
             "settles which camera is exterior_image_1_left. Only needed when the "
             "task configs disagree — the editor says so when they do.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Directory holding the dataset trees that DatasetObject entries name — "
             "OmniGibson's gm.DATA_PATH. Defaults to <repo>/deps/BEHAVIOR-1K/datasets, "
             "which is where install_simfoundry.sh puts them.",
    )
    parser.add_argument(
        "--asset-root",
        action="append",
        default=None,
        help="Directory to search for importable USD assets. Repeatable. Defaults to "
             "the scene's own objects/ tree plus its sibling scenes.",
    )
    parser.add_argument(
        "--scene-root",
        action="append",
        default=None,
        help="Directory holding scene directories, for the launcher. Repeatable. "
             "Defaults to the parent of the scene being opened plus this checkout's "
             "assets/scenes. These are also the only places a scene may be opened "
             "from at runtime.",
    )
    parser.add_argument(
        "--compose-root",
        default=None,
        help="Where 'New scene from template' creates scene directories. Defaults to "
             "the first scene root, which is what keeps a background referenced as "
             "../../mesh_backgrounds/... resolving.",
    )
    parser.add_argument(
        "--recents-file",
        default=None,
        help="Where the recently-opened list is kept. Defaults to "
             "$XDG_STATE_HOME/simfoundry/light_editor/recent_scenes.json.",
    )
    parser.add_argument("--settle-steps", type=int, default=240, help="Physics steps per settle")
    parser.add_argument("--settle-tolerance", type=float, default=0.005,
                        help="Metres of movement that counts as unsettled")
    args = parser.parse_args()

    scene_json = Path(args.scene).resolve()
    if not scene_json.exists():
        sys.exit(f"ERROR: scene JSON not found: {scene_json}")

    repo_root = HERE.parents[2]
    settle_runner = None
    if args.settle_after_save:
        print("Locating an interpreter with OmniGibson for post-save settling ...")
        settle_python = find_settle_python(repo_root, args.settle_python)
        if settle_python is None:
            sys.exit(
                "ERROR: --settle-after-save needs an OmniGibson interpreter. "
                "Pass --settle-python <path> or set SIMFOUNDRY_PYTHON."
            )
        print(f"  settling with {settle_python}")
        settle_runner = SettleRunner(
            settle_python, HERE / "settle.py", args.settle_steps, args.settle_tolerance,
            on_promote=EditorHandler.note_settle_promotion,
            expect_latest=lambda: EditorHandler.expected_latest_sha256,
        )

    EditorHandler.options = EditorOptions(args, repo_root)
    # The roots the catalog searches are also the only roots a scene may be
    # opened from.
    EditorHandler.scene_roots = tuple(
        scene_roots(scene_json, repo_root, extra=args.scene_root or [])
    )
    EditorHandler.recents = RecentScenes(args.recents_file)
    EditorHandler.compose_root = (
        Path(args.compose_root).resolve() if args.compose_root
        else (EditorHandler.scene_roots[0] if EditorHandler.scene_roots else scene_json.parent.parent)
    )
    EditorHandler.settle_runner = settle_runner

    try:
        bind_scene(
            scene_json, EditorHandler.options,
            reuse_cache="require" if args.no_extract else "never",
        )
    except (SceneEditError, OSError, ValueError, json.JSONDecodeError) as e:
        sys.exit(f"ERROR: {e}")
    EditorHandler.recents.record(scene_json)

    if EditorHandler.asset_roots:
        print("Asset library: " + ", ".join(str(r) for r in EditorHandler.asset_roots))
    else:
        print("Asset library: no importable asset roots found; the Add panel will be empty.")
    print("Scene roots:   " + (", ".join(str(r) for r in EditorHandler.scene_roots) or "none"))
    EditorHandler.mutation_token = secrets.token_urlsafe(32)
    EditorHandler.bind_host = args.host
    allowed = {h.strip().lower() for h in args.allow_hosts.split(",") if h.strip()}
    if args.host not in ("127.0.0.1", "localhost"):
        # A non-loopback bind accepts this machine's own names; the token,
        # not the Host header, is what authorises a write.
        allowed |= local_host_names()
    EditorHandler.extra_hosts = frozenset(allowed)

    server = ThreadingHTTPServer((args.host, args.port), EditorHandler)
    print(f"\nOpen http://localhost:{args.port}   (Ctrl-C to stop)")
    if args.host not in ("127.0.0.1", "localhost"):
        # Print the addresses a teammate can actually type.
        reachable = sorted(
            h for h in EditorHandler.extra_hosts
            if h not in LOOPBACK_HOSTS and not h.startswith("127.")
        )
        for name in reachable:
            print(f"  teammates: http://{name}:{args.port}")
        # Saves rebuild from the immutable startup scene plus one client's full
        # snapshot, so two browsers editing at once discard each other's work.
        print(
            f"\nWARNING: bound to {args.host} with no authentication.\n"
            "         Anyone who can reach this port can overwrite the scene.\n"
            "         Only one person should edit at a time -- concurrent saves\n"
            "         silently discard the other editor's changes.\n"
            "         Add --allow-hosts <name> if you reach it by a name not listed above."
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
