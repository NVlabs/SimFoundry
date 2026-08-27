# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Package one evaluation: a scene, its cameras, its task, and the command, plus
a manifest recording exactly which files went together.

The command names an absolute scene path, never a scene name — a name would
resolve to ``<name>_scene_state_latest.json`` at run time, not the layout just
reviewed. It also pins the task's instruction as ``s15_eval.prompt``, which
takes precedence over the task YAML in the eval stage.
"""

import hashlib
import json
import re
import shlex
from datetime import datetime
from pathlib import Path

# Where the C_application runner lives, relative to the repo root.
RUNNER = "scripts/pipeline/C_application/run.sh"

# Hydra reads task configs as a config group rooted here; the override is the
# path under it without the suffix -- "droid/droid_desk_serve_fruits".
TASK_GROUP_ROOT = Path("scripts/cfg") / "task"

#: The external_sensors group root, spelled the way the eval stage builds it
#: (``f"{SIMFOUNDRY_CFG_DIR}/external_sensors/{cfg}.yaml"``).
CAMERA_GROUP_ROOT = Path("scripts/cfg") / "external_sensors"

#: Where a snapshot taken at export time lands inside each group:
#: ``task=exports/<name>``, ``s15_eval.external_sensors_cfg=exports/<name>``.
SNAPSHOT_SUBDIR = "exports"

MANIFEST_SUFFIX = "_export.json"

#: The unique tail `scene_io.scene_output_path` gives an exported scene
#: (``<ts>_<uuid8>``), reused as the export's id so every artifact of one
#: export shares a name.
_EXPORT_ID = re.compile(r"_scene_state_.*?_(\d{8}_\d{6}_\d{6}_[0-9a-f]{8})$")


def export_id(scene_json):
    """One identifier shared by every artifact of a single export.

    Taken from the scene filename rather than generated, so all artifacts of
    one export carry the same stamp.

    Args:
        scene_json (str or Path): The exported scene path.

    Returns:
        str: The shared id.
    """
    stem = Path(scene_json).stem
    match = _EXPORT_ID.search(stem)
    if match:
        return match.group(1)
    # A scene named some other way still gets a stable id.
    return hashlib.sha256(stem.encode("utf-8")).hexdigest()[:16]


def snapshot_path(repo_root, group_root, name):
    """Where a config snapshot taken at export time is written.

    Args:
        repo_root (str or Path): Repo root.
        group_root (Path): :data:`TASK_GROUP_ROOT` or :data:`CAMERA_GROUP_ROOT`.
        name (str): Bare name, already made safe by the caller.

    Returns:
        tuple[Path, str]: The file, and the group override that selects it.
    """
    safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name)).strip("_-")
    if not safe:
        raise ValueError(f"{name!r} leaves no usable config name")
    return (Path(repo_root) / group_root / SNAPSHOT_SUBDIR / f"{safe}.yaml",
            f"{SNAPSHOT_SUBDIR}/{safe}")


def shadowing_task_config(repo_root, task_name, group):
    """The config the eval stage would actually load, when it is not *group*.

    ``resolve_task_config_path`` tries ``task/<task_name>.yaml`` **before** the
    group Hydra selected, so a ``task=`` override can name one file while the
    run loads another. Restated here rather than imported because
    ``python_utils`` is unreachable from the light editor's env; keep the two
    implementations in sync.

    Args:
        repo_root (str or Path): Repo root.
        task_name (str): The config's own ``task_name``.
        group (str): The group the command names.

    Returns:
        Path or None: The shadowing file, or None when *group* really is what
        would load.
    """
    if not task_name or not group:
        return None
    root = Path(repo_root) / TASK_GROUP_ROOT
    flat = root / f"{task_name}.yaml"
    if flat.is_file() and flat != (root / f"{group}.yaml"):
        return flat
    return None


def sha256_text(text):
    """Digest of a config's bytes, as recorded in a manifest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_group_name(task_path, repo_root):
    """Hydra group override for a task config.

    Args:
        task_path (str or Path): The config file.
        repo_root (str or Path): Repo root.

    Returns:
        str or None: e.g. ``droid/droid_desk_serve_fruits``; None when the file
        is not under the task config group at all, in which case naming it in a
        command would produce a Hydra error rather than a useful override.
    """
    try:
        relative = Path(task_path).resolve().relative_to(Path(repo_root).resolve() / TASK_GROUP_ROOT)
    except (ValueError, OSError):
        return None
    return relative.with_suffix("").as_posix()


def _hydra_quoted(value):
    """Escape a string so it survives as a double-quoted Hydra override value."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def eval_command(*, repo_root, scene_name, scene_json, cameras_cfg=None,
                 task_group=None, prompt=None, root_dir=None, extra=()):
    """Build the evaluation command for an exported bundle.

    Args:
        repo_root (str or Path): Repo root; the command is meant to be run here.
        scene_name (str): ``--scene-name``, which also selects the Data
            subdirectory the run writes into.
        scene_json (str or Path): Absolute path to the exported scene.
        cameras_cfg (str or None): ``external_sensors`` config *stem*, or None to
            leave the run config's own value in place.
        task_group (str or None): From :func:`task_group_name`.
        prompt (str or None): The task's language instruction, pinned as
            ``s15_eval.prompt``. None leaves the run config's own prompt.
        root_dir (str or Path or None): ``--root-dir`` when it is not the default.
        extra (iterable[str]): Further Hydra overrides, appended verbatim.

    Returns:
        dict: ``argv`` (list of tokens), ``command`` (a copy-pasteable string),
        and ``cwd``.
    """
    argv = ["bash", RUNNER, "--mode", "eval", "--scene-name", str(scene_name)]
    if root_dir:
        argv += ["--root-dir", str(root_dir)]
    overrides = [f"s15_eval.scene_json={Path(scene_json).resolve()}"]
    if cameras_cfg:
        overrides.append(f"s15_eval.external_sensors_cfg={cameras_cfg}")
    if task_group:
        overrides.append(f"task={task_group}")
    if prompt:
        # `s15_eval.prompt` takes precedence over the task YAML's
        # `language_instruction`, so the instruction is pinned here. The inner
        # quotes belong to the override, not the shell: Hydra reads a bare
        # comma as a list separator.
        overrides.append(f's15_eval.prompt="{_hydra_quoted(prompt)}"')
    overrides += list(extra)
    argv += ["--"] + overrides
    return {
        "argv": argv,
        "command": " ".join(shlex.quote(token) for token in argv),
        "cwd": str(Path(repo_root).resolve()),
    }


def manifest_path(scene_json):
    """Where the record of an export goes: beside the scene it describes."""
    scene_json = Path(scene_json)
    return scene_json.with_name(scene_json.stem + MANIFEST_SUFFIX)


def build_manifest(*, scene, cameras, task, command, warnings, timestamp=None,
                   artifacts=(), export=None):
    """Assemble the record written beside an exported scene.

    Args:
        scene (dict): ``path``, plus the change counts the export applied.
        cameras (dict): ``cfg_name``, ``path``, ``written``, ``background``.
        task (dict or None): ``name``, ``path``, ``group``, ``instruction``,
            ``association``.
        command (dict): From :func:`eval_command`.
        warnings (dict): ``layout`` and ``task`` lists, as shown in the review.
        timestamp (str or None): ISO-8601; generated when omitted.
        artifacts (iterable[dict]): One entry per file this export publishes:
            ``role``, ``path``, ``sha256``. The digest makes the recorded
            inputs checkable after the shared configs are edited.
        export (dict or None): ``id`` and how the configs were handled.

    Returns:
        dict: The manifest document.
    """
    return {
        # v2 adds `artifacts` and `export`; a v1 reader still finds every
        # field it knew about.
        "schema": "simfoundry.light_editor.export.v2",
        "exported_at": timestamp or datetime.now().astimezone().isoformat(timespec="seconds"),
        "export": export or {},
        "scene": scene,
        "cameras": cameras,
        "task": task,
        "command": command,
        "artifacts": list(artifacts),
        # Recorded even though the export went ahead anyway.
        "warnings": warnings,
    }


def write_manifest(manifest, scene_json, atomic_write):
    """Write a manifest beside its scene and return the path.

    Args:
        manifest (dict): From :func:`build_manifest`.
        scene_json (str or Path): The exported scene.
        atomic_write (callable): ``(path, text)`` writer, injected so this stays
            testable and shares ``scene_io``'s temp-file-and-rename behaviour.
    """
    target = manifest_path(scene_json)
    atomic_write(target, json.dumps(manifest, indent=2, allow_nan=False))
    return target


def summarise(manifest):
    """One line per part, for the server's terminal log."""
    scene = manifest.get("scene") or {}
    cameras = manifest.get("cameras") or {}
    task = manifest.get("task") or {}
    warnings = manifest.get("warnings") or {}
    lines = [f"scene   {scene.get('path')}"]
    if cameras.get("cfg_name"):
        lines.append(
            f"cameras {cameras['cfg_name']}"
            + ("  (written)" if cameras.get("written") else "  (unchanged)")
        )
    else:
        lines.append("cameras (none loaded — the run config's own value stands)")
    lines.append(
        f"task    {task.get('group') or '(none — the run config default stands)'}"
    )
    ground = scene.get("ground_plane")
    if ground:
        shown = {True: "visible", False: "hidden"}.get(
            ground.get("visible"), "visibility from the run config")
        lines.append(f"ground  z={ground['position'][2]:+.3f} m, {shown}")
    for artifact in manifest.get("artifacts") or []:
        lines.append(f"{artifact.get('role', 'file'):<7} {artifact.get('path')}  "
                     f"sha256:{(artifact.get('sha256') or '')[:12]}")
    count = len(warnings.get("layout") or []) + len(warnings.get("task") or [])
    if count:
        lines.append(f"warnings {count} recorded in the manifest")
    return lines
