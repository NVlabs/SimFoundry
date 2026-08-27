# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Read and write OmniGibson scene-state JSON without OmniGibson.

Where the format keeps what a placement editor needs:

    geometry   objects_info.init_info[name].args.usd_path   (relative to the JSON)
    scale      objects_info.init_info[name].args.scale
    pose       state.registry.object_registry[name].root_link.pos / .ori

Robots name a class instead of a ``usd_path``, and the reconstruction pipeline
emits ``DatasetObject`` entries identified by a category/model/dataset triple;
both are resolved to a USD separately, for display only. Editing writes scale
and pose, never geometry, so each object's ``expected_file_hash`` stays valid.

Quaternions are stored in OmniGibson's (x, y, z, w) order, matching three.js.
"""

import contextlib
import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from numbers import Real
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - this tree is Linux-only
    # Without fcntl the cross-process lock is skipped; the SHA compare-and-swap
    # below still refuses a conflicting write.
    fcntl = None

import usd_cache

# Display USDs for robots, which carry a class name rather than a usd_path.
# Keyed by (class_name, end_effector); "gripper" (FrankaPanda's default) and
# the legacy "panda" spelling select the same model.
ROBOT_ASSETS = {
    ("FrankaPanda", "robotiq"): "models/franka/franka_robotiq/usd/franka_robotiq.usda",
    ("FrankaPanda", "gripper"): "models/franka/franka_panda/usd/franka_panda.usda",
    ("FrankaPanda", "panda"): "models/franka/franka_panda/usd/franka_panda.usda",
    ("FrankaPanda", None): "models/franka/franka_panda/usd/franka_panda.usda",
    ("FrankaPanda", "inspire"): "models/franka/franka_dexhand/franka_inspire.usd",
    ("FrankaPanda", "leap"): "models/franka/franka_dexhand/franka_leap.usd",
    ("FrankaPanda", "allegro"): "models/franka/franka_dexhand/franka_allegro.usd",
    ("Yam", None): "models/yam/usd/yam.usda",
}

DEFAULT_ROBOT_ASSET_DIR = "deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets"
DEFAULT_DATASET_DIR = "deps/BEHAVIOR-1K/datasets"

# Sanity ranges for the physics fields the editor writes: they catch a slipped
# decimal point, not real-world variation. `asset_import` shares MASS_BOUNDS.
MASS_BOUNDS = (0.01, 50.0)
# PhysX accepts any non-negative coefficient; the ceiling only catches typos.
FRICTION_BOUNDS = (0.0, 2.0)
# Joint coordinates: metres for a prismatic joint, radians for a revolute one.
# Deliberately wider than any authored limit -- saved scenes legitimately store
# values outside the USD limits, so these bounds only catch unit mistakes.
JOINT_VALUE_BOUNDS = (-100.0, 100.0)


class SceneEditError(ValueError):
    """Raised when an edit cannot safely be compiled into a scene JSON."""


def load_scene(scene_json_path):
    """Read a scene-state JSON.

    Args:
        scene_json_path (str or Path): Path to a ``*_scene_state_*.json``.

    Returns:
        dict: The parsed document, unmodified.
    """
    return json.loads(Path(scene_json_path).read_text(encoding="utf-8"))


def scene_sha256(scene_json_path):
    """Return the SHA-256 of a scene file as it exists on disk."""
    return hashlib.sha256(Path(scene_json_path).read_bytes()).hexdigest()


def resolve_robot_usd(init_info, robot_asset_dir):
    """Resolve a robot's visual USD from its init_info.

    Args:
        init_info (dict): Entry from ``objects_info.init_info``.
        robot_asset_dir (str or Path): Root of ``omnigibson-robot-assets``.

    Returns:
        Path or None: Path to the robot USD, or None if unmapped/missing.
    """
    key = (init_info.get("class_name"), init_info.get("args", {}).get("end_effector"))
    rel = ROBOT_ASSETS.get(key)
    if rel is None:
        return None
    path = Path(robot_asset_dir) / rel
    return path if path.exists() else None


def resolve_dataset_usd(init_info, dataset_dir):
    """Resolve a ``DatasetObject``'s visual USD from its init_info.

    ``DatasetObject`` entries carry a ``category``/``model``/``dataset_name``
    triple instead of a ``usd_path``; look the USD up the way OmniGibson's
    ``DatasetObject.get_usd_path`` does.

    Args:
        init_info (dict): Entry from ``objects_info.init_info``.
        dataset_dir (str or Path): Directory holding the dataset trees, i.e.
            OmniGibson's ``gm.DATA_PATH``.

    Returns:
        Path or None: Path to the object's USD, or None when it is not a
        dataset object or the file is absent.
    """
    if init_info.get("class_name") != "DatasetObject":
        return None

    args = init_info.get("args", {})
    category = args.get("category")
    model = args.get("model")
    dataset_name = args.get("dataset_name", "behavior-1k-assets")
    if not category or not model:
        return None

    # These values come from the scene file and land in a path, so each must be
    # a single ordinary directory component.
    parts = (str(category), str(model), str(dataset_name))
    if any(os.sep in part or (os.altsep and os.altsep in part) or part in (".", "..")
           for part in parts):
        return None

    # `behavior-1k-assets` ships encrypted (`.encrypted.usd`) assets standalone
    # OpenUSD cannot open; those miss here on purpose.
    path = Path(dataset_dir) / dataset_name / "objects" / category / model / "usd" / f"{model}.usd"
    return path if path.exists() else None


def dataset_object_present(init_info, dataset_dir):
    """Return whether the dataset carries this ``DatasetObject``'s asset.

    Unlike :func:`resolve_dataset_usd`, an encrypted asset counts as present:
    OmniGibson can load it even though this editor cannot draw it.

    Args:
        init_info (dict): Entry from ``objects_info.init_info``.
        dataset_dir (str or Path): OmniGibson's ``gm.DATA_PATH``.

    Returns:
        bool or None: True/False, or None when the entry does not name a
        category/model pair at all.
    """
    args = init_info.get("args", {})
    category = args.get("category")
    model = args.get("model")
    dataset_name = args.get("dataset_name", "behavior-1k-assets")
    if not category or not model:
        return None
    parts = (str(category), str(model), str(dataset_name))
    if any(os.sep in part or (os.altsep and os.altsep in part) or part in (".", "..")
           for part in parts):
        return None
    directory = Path(dataset_dir) / dataset_name / "objects" / category / model / "usd"
    return any((directory / f"{model}{suffix}").exists()
               for suffix in (".usd", ".encrypted.usd", ".usda", ".usdc"))


def read_usd_physics(usd_path):
    """Read the rigid-body link names and authored mass out of a USD.

    Returns:
        dict: ``link`` (the first rigid body), ``links`` (every rigid body, in
        traversal order) and ``mass``. Empty/None when the USD is missing,
        unreadable, or carries no rigid body -- never raises. Friction edits
        must be validated against ``links``, never against a browser-supplied
        name: OmniGibson looks each link up in ``self.links[...]`` and a wrong
        name makes the scene fail to load.
    """
    facts = read_usd_facts(usd_path)
    return {"link": facts["link"], "links": list(facts["links"]), "mass": facts["mass"]}


#: Characters three.js strips from every node name it reads
#: (``PropertyBinding.sanitizeNodeName``); whitespace is rewritten to ``_``.
_THREE_STRIPS = "[].:/"


def node_name_for(prim_path):
    """Encode a USD prim path as a glTF node name three.js will not mangle.

    three.js strips :data:`_THREE_STRIPS` from every node name, so a raw prim
    path loses its separators on load. The leading slash is dropped and every
    other ``/`` becomes ``__`` -- do not tidy the ``__`` back into a ``/``.
    USD prim names are restricted to ``[A-Za-z_][A-Za-z0-9_]*``, so the result
    is a fixed point of the sanitize rule. The encoding is for recognition,
    not decoding -- ``a__b`` encodes the same as ``a/b`` -- and consumers match
    by equality against the ``child_node`` that :func:`read_usd_joints` emits.

    Args:
        prim_path (str or Sdf.Path): Absolute prim path.

    Returns:
        str: The encoded name, e.g. ``osulvv__lid_link__visuals__lid_link``.
    """
    return str(prim_path).lstrip("/").replace("/", "__")


#: USD joint types that contribute one coordinate to ``joint_pos``. Fixed and
#: spherical joints are part of the graph but not of the DOF vector.
_DOF_JOINT_TYPES = ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")


def read_usd_joints(usd_path):
    """Read every degree of freedom an asset has, in USD traversal order.

    The scene JSON stores articulation as a bare ``joint_pos`` array with no
    names and no limits, so both come from the asset here. The array's index
    order is PhysX metadata and cannot be derived; see `joint_facts`. USD
    authors revolute limits in degrees; they are converted to radians here to
    match ``joint_pos``.

    Args:
        usd_path (str or Path or None): The object's resolved USD.

    Returns:
        list[dict]: One entry per DOF joint, each with ``name``, ``type``
        (``revolute`` or ``prismatic``), ``axis``, ``lower``, ``upper`` and
        ``parent`` (the body path it hangs off), plus the geometry an animated
        preview needs: ``child`` (body1 -- the link that actually moves),
        ``child_node`` (that path as :func:`node_name_for` writes it into the
        glTF), ``pivot`` and ``direction`` (the axis in the frame the proxy is
        baked into) and ``rest`` (the coordinate it is baked at). The four
        geometry fields are null together when the joint cannot be placed; see
        `_joint_geometry`. Empty when the USD is missing, unreadable or has no
        joints -- never raises.
    """
    return [dict(joint) for joint in read_usd_facts(usd_path)["joints"]]


#: Facts kept per asset, keyed by `usd_cache.cache_key`; the limit only stops
#: unbounded growth across a long session of scene switches.
_FACTS_LIMIT = 256

_FACTS = OrderedDict()

#: Guards `_FACTS`: the editor's server is threaded, and the LRU bookkeeping
#: is not safe under concurrent access.
_FACTS_LOCK = threading.Lock()


def _empty_facts():
    return {"link": None, "links": [], "mass": None, "joints": []}


def clear_usd_facts():
    """Forget every memoised asset fact. The counterpart of `usd_cache.clear`."""
    with _FACTS_LOCK:
        _FACTS.clear()


#: Unit vectors for USD's joint axis tokens, in the joint's own frame.
_JOINT_AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}

#: How far (metres) a joint's frame at coordinate zero may sit from its child
#: body's authored transform before the joint is reported as unplaceable.
#: Matches `robot_pose.REST_TOLERANCE` (robot_pose imports this module, so the
#: value is copied, not imported).
_JOINT_REST_TOLERANCE = 1e-3


def _rigid_frame(pos, rot):
    """Build a USD row-vector 4x4 from a joint's local position and rotation.

    Args:
        pos (Gf.Vec3f or None): Translation; None means zero.
        rot (Gf.Quatf or None): Rotation; None means identity.

    Returns:
        Gf.Matrix4d: Row-vector transform, translation in the last row.
    """
    from pxr import Gf

    matrix = Gf.Matrix4d(1.0)
    if rot is not None:
        matrix.SetRotate(Gf.Quatd(rot.GetReal(), Gf.Vec3d(*rot.GetImaginary())))
    if pos is not None:
        matrix.SetTranslateOnly(Gf.Vec3d(*pos))
    return matrix


def _joint_geometry(prim, stage, cache, axis):
    """Compute where one joint's axis lies in the frame the proxy is baked into.

    `extract.load_visual_scene` bakes every mesh into the stage's root frame,
    so a pivot and direction expressed there let the browser animate the glTF
    it already has. A joint's frame is authored inside body0
    (``localPos0``/``localRot0``); USD is row-vector (``world = local * M``,
    translation in the last row), so the joint frame is ``local0 * body0``.
    body1's authored transform must equal the joint at coordinate zero, within
    `_JOINT_REST_TOLERANCE`; a joint that fails is reported unplaceable.

    The pivot is computed against the parent at rest, which assumes a flat
    joint tree (every joint hanging off one base body). A nested tree passes
    the rest check but would preview correctly only while its ancestors sit at
    zero.

    Args:
        prim (Usd.Prim): The joint prim.
        stage (Usd.Stage): The stage it was found on.
        cache (UsdGeom.XformCache): Shared transform cache, as extract.py uses.
        axis (str): The joint's axis token, ``X``, ``Y`` or ``Z``.

    Returns:
        dict: ``child`` (body1's prim path), ``child_node`` (that path as it is
            written into the glTF), ``pivot`` and ``direction``. Any of them is
            None when it cannot be resolved; a client that reads a null shows
            the joint's numbers and says why it will not animate them.
    """
    from pxr import Gf, UsdPhysics

    out = {"child": None, "child_node": None, "pivot": None, "direction": None}
    joint = UsdPhysics.Joint(prim)
    targets = joint.GetBody1Rel().GetTargets()
    if not targets:
        return out
    out["child"] = str(targets[0])
    out["child_node"] = node_name_for(targets[0])
    child = stage.GetPrimAtPath(targets[0])
    if not child:
        return out

    parents = joint.GetBody0Rel().GetTargets()
    if parents:
        parent = stage.GetPrimAtPath(parents[0])
        if not parent:
            return out
        parent_world = cache.GetLocalToWorldTransform(parent)
    else:
        # An empty body0 anchors the joint to the world, so its frame is
        # already stage-root.
        parent_world = Gf.Matrix4d(1.0)

    joint_world = _rigid_frame(
        joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()
    ) * parent_world
    child_frame = _rigid_frame(
        joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()
    )
    at_rest = child_frame.GetInverse() * joint_world
    authored = cache.GetLocalToWorldTransform(child)
    error = max(abs(at_rest[r][c] - authored[r][c]) for r in range(4) for c in range(4))
    if error > _JOINT_REST_TOLERANCE:
        return out

    direction = joint_world.TransformDir(Gf.Vec3d(*_JOINT_AXES.get(axis, _JOINT_AXES["X"])))
    if direction.GetLength() < 1e-9:
        return out
    out["pivot"] = [round(float(v), 6) for v in joint_world.ExtractTranslation()]
    out["direction"] = [round(float(v), 6) for v in direction.GetNormalized()]
    return out


def read_usd_facts(usd_path):
    """Read every fact this editor needs from an asset's USD, in one traversal.

    The result is memoised on the file's identity (`usd_cache.cache_key`:
    resolved path, mtime and size), so an asset reimported over an old one is
    re-read and an unchanged one is not.

    Returns:
        dict: ``link`` (the first rigid body), ``links`` (every rigid body, in
        traversal order), ``mass`` (the first authored ``physics:mass``) and
        ``joints`` (one entry per DOF joint, with the geometry
        `read_usd_joints` documents). Empty throughout when the USD is
        missing, unreadable, or carries neither -- never raises: one
        unreadable asset greys out a panel, it does not fail the scene load.
    """
    if not usd_path:
        return _empty_facts()
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        return _empty_facts()

    key = usd_cache.cache_key(usd_path)
    if key is not None:
        with _FACTS_LOCK:
            cached = _FACTS.get(key)
            if cached is not None:
                _FACTS.move_to_end(key)
                return cached

    facts = _empty_facts()
    try:
        stage = usd_cache.open_stage(usd_path)
        if stage is None:
            return facts
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        links, mass, joints = [], None, []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                links.append(prim.GetName())
                if mass is None:
                    attribute = prim.GetAttribute("physics:mass")
                    value = attribute.Get() if attribute and attribute.IsValid() else None
                    mass = round(float(value), 6) if value is not None else None
            if not prim.IsA(UsdPhysics.Joint):
                continue
            type_name = str(prim.GetTypeName())
            if type_name not in _DOF_JOINT_TYPES:
                continue
            revolute = type_name == "PhysicsRevoluteJoint"
            typed = (UsdPhysics.RevoluteJoint if revolute
                     else UsdPhysics.PrismaticJoint)(prim)
            lower = typed.GetLowerLimitAttr().Get()
            upper = typed.GetUpperLimitAttr().Get()
            if revolute and lower is not None and upper is not None:
                lower, upper = math.radians(lower), math.radians(upper)
            axis = str(typed.GetAxisAttr().Get() or "X")
            bodies = UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()
            joints.append({
                "name": prim.GetName(),
                "type": "revolute" if revolute else "prismatic",
                "axis": axis,
                # A joint authored with lower > upper (or both zero) is
                # unlimited -- a continuous hinge -- and has no range to show.
                "lower": None if lower is None else round(float(lower), 6),
                "upper": None if upper is None else round(float(upper), 6),
                "parent": str(bodies[0]) if bodies else None,
                # Pivot/axis in the frame the proxy is baked into, so the
                # browser can animate the link it already has.
                **_joint_geometry(prim, stage, cache, axis),
                # The proxy is baked from the USD as authored, which is the
                # joint at coordinate zero.
                "rest": 0.0,
            })
        if links:
            facts.update({"link": links[0], "links": links, "mass": mass})
        facts["joints"] = joints
    except Exception:
        facts = _empty_facts()

    if key is not None:
        # Concurrent misses compute equivalent answers, so last-writer-wins;
        # the lock guards the eviction walking the same dict.
        with _FACTS_LOCK:
            _FACTS[key] = facts
            while len(_FACTS) > _FACTS_LIMIT:
                _FACTS.popitem(last=False)
    return facts


def _validated_joint_limits(args, joints):
    """The scene's limit overrides, keyed by joint name, filtered to real joints."""
    override = args.get("joint_limits")
    if not isinstance(override, dict):
        return {}
    known = {joint["name"] for joint in joints}
    clean = {}
    for key, value in override.items():
        if key not in known or not isinstance(value, dict):
            continue
        pair = {}
        for side in ("lower", "upper"):
            number = value.get(side)
            if isinstance(number, Real) and not isinstance(number, bool):
                pair[side] = float(number)
        if pair:
            clean[key] = pair
    return clean


def _prismatic_scale(args):
    """Report how the object's scale changes a prismatic joint's real travel.

    OmniGibson multiplies a prismatic joint's range by the object's scale at
    load (``EntityPrim._update_joint_limits``), so the authored range is not
    what the simulator enforces on a scaled object. Only a uniform scale gets
    a factor: computing the non-uniform case needs the link frame this
    function does not have, so it reports that the range is scaled without
    claiming a number.

    Returns:
        dict or None: ``{"factor": float or None, "uniform": bool}``, or None
        when the object is at unit scale and nothing changes.
    """
    scale = args.get("scale")
    if isinstance(scale, Real) and not isinstance(scale, bool):
        scale = [scale, scale, scale]
    if not isinstance(scale, (list, tuple)) or len(scale) != 3:
        return None
    try:
        values = [float(v) for v in scale]
    except (TypeError, ValueError):
        return None
    if all(abs(v - 1.0) < 1e-9 for v in values):
        return None
    uniform = max(values) - min(values) < 1e-9
    return {"factor": values[0] if uniform else None, "uniform": uniform}


def joint_facts(args, usd, registry_entry, *, is_robot=False):
    """Collect what the joints panel needs about one object.

    Returns None for anything with no degrees of freedom, which is what makes
    the panel appear for a drawer and not for a plate.

    ``addressable`` reports whether the asset has exactly as many degrees of
    freedom as the saved ``joint_pos`` array. The array's index order is
    PhysX's, so when the counts disagree the client shows the raw array
    instead of pairing values with the wrong joints. The robot is excluded:
    its ``joint_pos`` carries gripper DOFs beyond the arm joints, and its
    configuration is rig calibration rather than scene dressing.

    Args:
        args (dict): The object's ``init_info.args``.
        usd (Path or None): Its resolved USD.
        registry_entry (dict): Its ``object_registry`` entry, holding ``joint_pos``.
        is_robot (bool): Whether this object is the robot.

    Returns:
        dict or None: ``joints`` (from the asset), ``values`` (the saved array),
        ``limits`` (the scene's overrides, by joint name), ``addressable`` and
        ``scale``.
    """
    if is_robot:
        return None
    joints = read_usd_joints(usd)
    if not joints:
        return None
    values = registry_entry.get("joint_pos") if isinstance(registry_entry, dict) else None
    values = [float(v) for v in values] if values else []
    return {
        "joints": joints,
        "values": values,
        "limits": _validated_joint_limits(args, joints),
        "addressable": len(values) == len(joints),
        # What the simulator really enforces on a scaled prismatic joint.
        "scale": _prismatic_scale(args),
    }


def _authored_friction(args, link):
    """The static friction already in an object's init args, if any."""
    materials = args.get("link_physics_materials")
    if not isinstance(materials, dict):
        return None
    entry = materials.get(link) if link else None
    if not isinstance(entry, dict):
        # A scene written against a different link name still has a coefficient
        # worth showing; a single entry is unambiguous.
        entries = [v for v in materials.values() if isinstance(v, dict)]
        entry = entries[0] if len(entries) == 1 else None
    if not isinstance(entry, dict):
        return None
    value = entry.get("static_friction")
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) else None


#: The category `background_io.background_spec` stamps on a room it adds, and
#: the one every consumer of a scene reads a room back by.
_BACKGROUND_CATEGORY = "mesh_background"

#: Names a room is written under in older scenes that carry no category:
#: `mesh_background`/`mesh_background_<n>` from this repo's editors and
#: `gs_background` from the reconstruction pipeline. Matched exactly, never as
#: a substring, so an imported prop named e.g. `blue_background_panel` is not
#: mistaken for scenery.
_BACKGROUND_NAME = re.compile(r"(?:mesh|gs)_background(?:_\d+)?", re.I)


def _asset_status(info, class_name, is_robot, usd_rel, usd, robot_asset_dir, dataset_dir):
    """Report whether an object names geometry, and whether it is present.

    ``missing`` and ``unchecked`` are distinct: "looked and not there" is a
    broken scene, "not given the root to look in" is not.

    Returns:
        dict: ``expected`` (does this object need an asset), ``status`` (one of
        ``none``/``ok``/``missing``/``unchecked``) and ``detail`` -- a sentence
        naming what is missing, or None.
    """
    if usd_rel:
        if usd is not None:
            return {"expected": True, "status": "ok", "detail": None}
        return {"expected": True, "status": "missing",
                "detail": f"usd_path {usd_rel} does not resolve"}

    if class_name == "DatasetObject":
        args = info.get("args", {})
        library = (None if dataset_dir is None else
                   Path(dataset_dir) / str(args.get("dataset_name", "behavior-1k-assets"))
                   / "objects")
        if library is None or not library.is_dir():
            # An uninstalled library is not a missing object; only the second
            # is a broken scene.
            return {"expected": True, "status": "unchecked",
                    "detail": f"no dataset library at {library}, so this "
                              "DatasetObject's asset was not looked for"}
        present = dataset_object_present(info, dataset_dir)
        if present is None:
            return {"expected": True, "status": "missing",
                    "detail": "DatasetObject names no usable category/model pair"}
        if present:
            return {"expected": True, "status": "ok", "detail": None}
        return {"expected": True, "status": "missing",
                "detail": f"no asset for category {args.get('category')!r} "
                          f"model {args.get('model')!r} in "
                          f"{args.get('dataset_name', 'behavior-1k-assets')}"}

    if is_robot:
        key = (info.get("class_name"), info.get("args", {}).get("end_effector"))
        relative = ROBOT_ASSETS.get(key)
        if relative is None:
            # An unmapped robot is not a broken scene: ROBOT_ASSETS only maps
            # display proxies, and OmniGibson resolves a robot's own USD from
            # its own asset root.
            return {"expected": True, "status": "unchecked",
                    "detail": f"no proxy asset is mapped for {key[0]}"}
        if robot_asset_dir is None or not Path(robot_asset_dir).is_dir():
            return {"expected": True, "status": "unchecked",
                    "detail": f"no robot asset library at {robot_asset_dir}, so the "
                              "robot's asset was not looked for"}
        if usd is not None:
            return {"expected": True, "status": "ok", "detail": None}
        return {"expected": True, "status": "missing",
                "detail": f"robot asset {relative} is not under {robot_asset_dir}"}

    # Primitives, lights and native OmniGibson objects are constructed without
    # an asset on disk, so there is nothing to be missing.
    return {"expected": False, "status": "none", "detail": None}


def iter_objects(scene, scene_json_path, robot_asset_dir=None, dataset_dir=None,
                 *, usd_facts=True):
    """Yield one record per object in the scene.

    Args:
        scene (dict): Parsed scene JSON.
        scene_json_path (str or Path): Path the JSON was read from; relative
            ``usd_path`` values resolve against its parent directory.
        robot_asset_dir (str or Path or None): Root of ``omnigibson-robot-assets``.
        dataset_dir (str or Path or None): OmniGibson's ``gm.DATA_PATH``, used
            to resolve ``DatasetObject`` entries that carry no ``usd_path``.
        usd_facts (bool): Read ``physics`` and ``joints`` out of each object's
            USD. This opens and traverses every asset in the scene; off, those
            two fields are None and no USD is opened at all.

    Yields:
        dict: ``name``, ``usd`` (Path or None), ``position``, ``orientation``,
            ``scale``, ``category``, ``kind`` (one of object/robot/background),
            ``editable`` (position, orientation and scale all writable) and
            ``posable`` (position and orientation writable -- true for the
            robot too, a strict superset of ``editable``).

            Plus three fields about the *asset*: ``usd_expected`` (does this
            object need one at all), ``usd_status``
            (``none``/``ok``/``missing``/``unchecked``) and ``usd_detail``.
            A ``DatasetObject`` and a robot name their geometry by class
            rather than by path, so a missing ``usd_path`` does not mean there
            is nothing to resolve. See `_asset_status`.
    """
    scene_dir = Path(scene_json_path).parent
    registry = scene.get("state", {}).get("registry", {}).get("object_registry", {})

    for name, info in scene.get("objects_info", {}).get("init_info", {}).items():
        args = info.get("args", {})
        usd_rel = args.get("usd_path")

        root = registry.get(name, {}).get("root_link", {})
        class_module = str(info.get("class_module", ""))
        class_name = str(info.get("class_name", ""))
        category = str(args.get("category", class_name))
        # Native OG objects can also lack a usd_path, so prefer the serialized
        # class identity, with the conventional name as a legacy fallback.
        is_robot = class_module.startswith("omnigibson.robots.") or (
            name.startswith("robot") and usd_rel is None
        )

        if usd_rel:
            usd_path = Path(usd_rel)
            usd = (usd_path if usd_path.is_absolute() else scene_dir / usd_path).resolve()
            usd = usd if usd.exists() else None
        elif dataset_dir is not None and class_name == "DatasetObject":
            usd = resolve_dataset_usd(info, dataset_dir)
        elif robot_asset_dir is not None and is_robot:
            usd = resolve_robot_usd(info, robot_asset_dir)
        else:
            usd = None
        asset = _asset_status(info, class_name, is_robot, usd_rel, usd,
                              robot_asset_dir, dataset_dir)
        is_background = (category.lower() == _BACKGROUND_CATEGORY
                         or bool(_BACKGROUND_NAME.fullmatch(name)))

        raw_scale = args.get("scale", [1.0, 1.0, 1.0])
        if isinstance(raw_scale, Real) and not isinstance(raw_scale, bool):
            raw_scale = [raw_scale, raw_scale, raw_scale]

        yield {
            "name": name,
            "usd": usd,
            "usd_reference": usd_rel,
            # Whether this object needs an asset at all, and whether the one
            # it needs is there; see `_asset_status`.
            "usd_expected": asset["expected"],
            "usd_status": asset["status"],
            "usd_detail": asset["detail"],
            "category": category,
            "position": list(root.get("pos", [0.0, 0.0, 0.0])),
            "orientation": list(root.get("ori", [0.0, 0.0, 0.0, 1.0])),
            "scale": list(raw_scale),
            "kind": "robot" if is_robot else ("background" if is_background else "object"),
            # Three permission flags:
            #   editable  props only -- the set Select All, Arrange, Duplicate
            #             and Remove operate on.
            #   posable   position and orientation: props, the robot and the
            #             room. Props hold world poses and are not parented to
            #             the room, so the room slides underneath them.
            #   scalable  props and the room, never the robot: scaling a robot
            #             resizes the mesh without rescaling the joint frames,
            #             collision geometry or actuator limits under it.
            "editable": not is_robot and not is_background,
            "posable": True,
            "scalable": not is_robot,
            # `authored_mass` is the USD's own number, `mass` the scene's
            # override of it; `link` is what a friction material is addressed
            # to -- see read_usd_physics.
            "physics": (physics_facts(args, usd)
                        if usd_facts and not is_background else None),
            # None for anything with no degrees of freedom; see `joint_facts`.
            "joints": (joint_facts(args, usd, registry.get(name, {}), is_robot=is_robot)
                       if usd_facts and not is_background else None),
        }


def authored_state(args, registry_entry, *, link=None, joints=None):
    """Read everything about one object that the scene document decides.

    The JSON-only counterpart to the asset-derived :func:`physics_facts` and
    :func:`joint_facts`; used to refresh served state without re-extracting
    anything.

    Args:
        args (dict): The object's ``objects_info.init_info[name].args``.
        registry_entry (dict): Its ``state.registry.object_registry[name]``.
        link (str or None): The rigid body a friction material is addressed to,
            from the asset. Without it a friction written against a different
            link name is still reported.
        joints (list or None): Joint names from the asset, used to validate the
            limit map. None reports the limits unfiltered.

    Returns:
        dict: ``position``, ``orientation``, ``scale``, ``mass``, ``friction``,
        ``joint_values`` and ``joint_limits``.
    """
    args = args if isinstance(args, dict) else {}
    entry = registry_entry if isinstance(registry_entry, dict) else {}
    root = entry.get("root_link") if isinstance(entry.get("root_link"), dict) else {}

    raw_scale = args.get("scale", [1.0, 1.0, 1.0])
    if isinstance(raw_scale, Real) and not isinstance(raw_scale, bool):
        raw_scale = [raw_scale, raw_scale, raw_scale]

    mass = args.get("mass")
    values = entry.get("joint_pos")
    return {
        "position": [float(v) for v in root.get("pos", [0.0, 0.0, 0.0])],
        "orientation": [float(v) for v in root.get("ori", [0.0, 0.0, 0.0, 1.0])],
        "scale": [float(v) for v in raw_scale],
        "mass": float(mass) if isinstance(mass, Real) and not isinstance(mass, bool) else None,
        "friction": _authored_friction(args, link),
        "joint_values": [float(v) for v in values] if values else [],
        "joint_limits": _validated_joint_limits(args, joints) if joints else {},
    }


def apply_authored_state_to_manifest(manifest, scene):
    """Re-point a served manifest at what a scene document now says.

    The manifest holds geometry proxies and the authored numbers laid over
    them. Geometry cannot change under a save, so only the numbers are
    re-read. Objects the document no longer has are marked ``removed`` rather
    than deleted from the manifest: the browser must still account for each
    one in its complete snapshot.

    Args:
        manifest (dict): The served manifest. Mutated in place.
        scene (dict): The scene document to read.

    Returns:
        dict: ``updated``, ``removed`` and ``restored`` name lists.
    """
    init_info = scene.get("objects_info", {}).get("init_info", {}) or {}
    registry = scene.get("state", {}).get("registry", {}).get("object_registry", {}) or {}
    updated, removed, restored = [], [], []

    for entry in manifest.get("objects", []):
        name = entry.get("name")
        info = init_info.get(name)
        if not isinstance(info, dict):
            if not entry.get("removed"):
                removed.append(name)
            entry["removed"] = True
            continue
        if entry.get("removed"):
            entry["removed"] = False
            restored.append(name)
        # An import that a save has written is no longer pending.
        if entry.get("added"):
            entry["added"] = False
        physics = entry.get("physics") or {}
        joints = entry.get("joints") or {}
        state = authored_state(
            info.get("args") or {}, registry.get(name) or {},
            link=physics.get("link"), joints=joints.get("joints"),
        )
        entry["position"] = state["position"]
        entry["orientation"] = state["orientation"]
        entry["scale"] = state["scale"]
        if physics:
            physics["mass"] = state["mass"]
            physics["friction"] = state["friction"]
        if joints:
            joints["values"] = state["joint_values"]
            joints["limits"] = state["joint_limits"]
            joints["addressable"] = len(state["joint_values"]) == len(
                joints.get("joints") or [])
        updated.append(name)

    return {"updated": updated, "removed": removed, "restored": restored}


def physics_facts(args, usd):
    """The mass and friction an object currently has, and where they came from."""
    from_usd = read_usd_physics(usd)
    override = args.get("mass")
    return {
        "link": from_usd["link"],
        # Friction is one object-wide number applied to every rigid body, and
        # this asset-read list is the only thing a friction edit may be
        # validated against (see `validate_edits`).
        "links": list(from_usd.get("links") or []),
        "authored_mass": from_usd["mass"],
        "mass": float(override) if isinstance(override, Real)
        and not isinstance(override, bool) else None,
        "friction": _authored_friction(args, from_usd["link"]),
    }


def _validated_vector(name, field, value, length, *, positive=False, normalize=False):
    """Validate and normalize one JSON transform vector."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise SceneEditError(f"{name}.{field} must contain exactly {length} numbers")

    result = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, Real):
            raise SceneEditError(f"{name}.{field} must contain only numbers")
        number = float(component)
        if not math.isfinite(number):
            raise SceneEditError(f"{name}.{field} must contain only finite numbers")
        if positive and number <= 0.0:
            raise SceneEditError(f"{name}.{field} values must be greater than zero")
        result.append(number)

    if normalize:
        norm = math.sqrt(sum(component * component for component in result))
        if norm <= 1e-12:
            raise SceneEditError(f"{name}.{field} must be a non-zero quaternion")
        # Keep a near-unit quaternion byte-for-byte: re-normalizing every
        # object creates meaningless scene diffs and can zero velocities on
        # objects the user never touched.
        if abs(norm - 1.0) > 1e-6:
            result = [component / norm for component in result]
    return result


def _validated_number(name, field, value, low, high):
    """Validate one scalar physics field against a sanity range."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SceneEditError(f"{name}.{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise SceneEditError(f"{name}.{field} must be finite")
    if not low <= number <= high:
        raise SceneEditError(f"{name}.{field} must be between {low} and {high}")
    return number


def _validated_limit_map(name, value, known_joints=None):
    """Validate a joint-limit override map: joint name -> lower/upper.

    Keyed by name because the index order belongs to PhysX and is not recorded
    in this format. A pair with ``lower > upper`` is refused: PhysX reads that
    as an unlimited joint.
    """
    if not isinstance(value, dict):
        raise SceneEditError(f"{name}.joint_limits must be an object keyed by joint name")
    # Fail closed when the asset could not be read: an unchecked joint name
    # would be recorded as though it described the asset.
    if not known_joints:
        raise SceneEditError(
            f"{name}'s joint limits cannot be recorded: its asset could not be read, "
            "so the joint names cannot be checked"
        )
    valid = {str(joint.get("name")) for joint in known_joints if isinstance(joint, dict)}
    clean = {}
    for joint, pair in value.items():
        if not isinstance(joint, str) or not joint.strip():
            raise SceneEditError(f"{name}.joint_limits has an unnamed joint")
        if joint.strip() not in valid:
            raise SceneEditError(
                f"{name} has no joint named {joint.strip()!r}. Its joints are: "
                + ", ".join(sorted(valid))
            )
        if not isinstance(pair, dict) or not {"lower", "upper"} <= set(pair):
            raise SceneEditError(
                f"{name}.joint_limits[{joint}] needs both a lower and an upper"
            )
        lower = _validated_number(name, f"joint_limits[{joint}].lower",
                                  pair["lower"], *JOINT_VALUE_BOUNDS)
        upper = _validated_number(name, f"joint_limits[{joint}].upper",
                                  pair["upper"], *JOINT_VALUE_BOUNDS)
        if lower > upper:
            raise SceneEditError(
                f"{name}.joint_limits[{joint}] has a lower limit above its upper"
            )
        clean[joint.strip()] = {"lower": lower, "upper": upper}
    return clean


def validate_edits(scene, edits, editable_names=None, posable_names=None,
                   scalable_names=None, *, asset_facts=None):
    """Validate an edit payload and return a normalized copy.

    Args:
        scene (dict): Parsed base scene.
        edits (dict): Browser edit payload.
        editable_names (set[str] or None): Names this caller permits the full
            set of edits on, physics and articulation included. When omitted,
            unrestricted.
        posable_names (set[str] or None): Names this caller permits a
            position/orientation change on -- a superset of *editable_names*
            (the robot and the room are posable but not editable: see
            `iter_objects`). Defaults to *editable_names*.
        scalable_names (set[str] or None): Names this caller permits a scale
            change on -- props and the room, never the robot. Defaults to
            *editable_names*.
        asset_facts (dict or None): ``{name: {"links": [...], "joints": [...]}}``
            read out of each object's own USD by the caller; the only trusted
            source of link and joint names. Omitted means nothing was
            inspected, and every edit that needs such a name is refused.
    """
    if not isinstance(edits, dict):
        raise SceneEditError("edits must be an object")
    if posable_names is None:
        posable_names = editable_names
    if scalable_names is None:
        scalable_names = editable_names

    registry = scene.get("state", {}).get("registry", {}).get("object_registry", {})
    init_info = scene.get("objects_info", {}).get("init_info", {})
    known_names = set(registry) & set(init_info)
    allowed_names = known_names if posable_names is None else known_names & set(posable_names)
    normalized = {}

    for name, edit in edits.items():
        if not isinstance(name, str) or name not in known_names:
            raise SceneEditError(f"unknown scene object: {name!r}")
        if name not in allowed_names:
            raise SceneEditError(f"object is locked and cannot be edited: {name}")
        if not isinstance(edit, dict):
            raise SceneEditError(f"edit for {name} must be an object")
        facts = (asset_facts or {}).get(name) or {}

        unknown_fields = set(edit) - {"position", "orientation", "scale",
                                      # No `friction_link`: friction is
                                      # object-wide and the links come from
                                      # the server's own read of the USD.
                                      "mass", "friction",
                                      "joint_values", "joint_limits"}
        if unknown_fields:
            raise SceneEditError(
                f"unsupported field(s) for {name}: {', '.join(sorted(unknown_fields))}"
            )
        if not edit:
            raise SceneEditError(f"edit for {name} is empty")
        # Posable was checked above for every name; scalable and editable gate
        # the narrower fields.
        if "scale" in edit and scalable_names is not None and name not in set(scalable_names):
            raise SceneEditError(f"{name}'s scale is locked")
        # Physics belongs to props: a robot's mass and contact materials come
        # from its rig, not from this panel.
        for field in ("mass", "friction"):
            if (field in edit and editable_names is not None
                    and name not in set(editable_names)):
                raise SceneEditError(f"{name}'s {field} is locked")
        # Articulation likewise: a robot's joint vector is its rig's
        # configuration (see `joint_facts`).
        for field in ("joint_values", "joint_limits"):
            if (field in edit and editable_names is not None
                    and name not in set(editable_names)):
                raise SceneEditError(f"{name}'s articulation is locked")

        # A pose lives under root_link and a scale under init_info.args; an
        # object serialized without root_link can only be scaled.
        if ("position" in edit or "orientation" in edit) and not isinstance(
            registry.get(name, {}).get("root_link"), dict
        ):
            raise SceneEditError(
                f"{name} has no state.registry.object_registry.{name}.root_link, "
                "so its pose cannot be edited"
            )
        # `joint_pos` is the array being replaced, so an object serialized
        # without one has nothing to write to, and the length check is the
        # only guard that the client means the same articulation.
        if "joint_values" in edit:
            saved = registry.get(name, {}).get("joint_pos")
            if not isinstance(saved, list) or not saved:
                raise SceneEditError(
                    f"{name} has no state.registry.object_registry.{name}.joint_pos, "
                    "so its joints cannot be edited"
                )
            if not isinstance(edit["joint_values"], list) or len(edit["joint_values"]) != len(saved):
                raise SceneEditError(
                    f"{name}.joint_values must contain exactly {len(saved)} numbers"
                )

        clean = {}
        if "position" in edit:
            clean["position"] = _validated_vector(name, "position", edit["position"], 3)
        if "orientation" in edit:
            clean["orientation"] = _validated_vector(
                name, "orientation", edit["orientation"], 4, normalize=True
            )
        if "scale" in edit:
            clean["scale"] = _validated_vector(name, "scale", edit["scale"], 3, positive=True)
        # Null means "take the override off", distinct from zero and from
        # omitting the field; Revert needs it to undo a saved physics edit.
        if "mass" in edit:
            clean["mass"] = None if edit["mass"] is None else _validated_number(
                name, "mass", edit["mass"], *MASS_BOUNDS)
        if "friction" in edit and edit["friction"] is None:
            clean["friction"] = None
        elif "friction" in edit:
            clean["friction"] = _validated_number(
                name, "friction", edit["friction"], *FRICTION_BOUNDS)
            # The links come from the server's own read of the asset, never
            # the request: OmniGibson raises on a link it cannot find in
            # `self.links[...]`, so a wrong name is a scene that will not
            # load. Fail closed when the asset could not be read.
            if not facts.get("links"):
                raise SceneEditError(
                    f"{name}'s friction cannot be set: its asset could not be read, "
                    "so the links to apply it to are unknown"
                )
            clean["friction_links"] = [str(link) for link in facts["links"]]
        if "joint_values" in edit:
            clean["joint_values"] = [
                _validated_number(name, f"joint_values[{index}]", value, *JOINT_VALUE_BOUNDS)
                for index, value in enumerate(edit["joint_values"])
            ]
        # Null clears the whole override, the same way a null mass does.
        if "joint_limits" in edit and edit["joint_limits"] is None:
            clean["joint_limits"] = None
        elif "joint_limits" in edit:
            clean["joint_limits"] = _validated_limit_map(
                name, edit["joint_limits"], facts.get("joints"))
        normalized[name] = clean

    return normalized


def editable_object_names(scene, scene_json_path):
    """Return names the light editor is allowed to modify."""
    return {
        record["name"]
        for record in iter_objects(scene, scene_json_path, robot_asset_dir=None)
        if record["editable"]
    }


def robot_object_names(scene, scene_json_path):
    """Return the robot's name(s) in *scene*.
    """
    return {
        record["name"]
        for record in iter_objects(scene, scene_json_path, robot_asset_dir=None)
        if record["kind"] == "robot"
    }


def background_object_names(scene, scene_json_path):
    """Return the scanned room's name(s) in *scene*.
    """
    return {
        record["name"]
        for record in iter_objects(scene, scene_json_path, robot_asset_dir=None)
        if record["kind"] == "background"
    }


def _slug(text):
    """Reduce a string to a component that is safe to build a filename from.

    The value comes from scene JSON and lands in a path, so anything that
    could traverse or collide is folded to an underscore.
    """
    cleaned = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(text))
    return cleaned.strip("_-")


def background_id(scene, scene_json_path):
    """Identify the scanned room a scene is laid out in.

    Every scene names its background ``mesh_background_0``, so the referenced
    USD, not the name, identifies the room. Camera placement is a property of
    the room, which makes this the key for remembering placements across
    scenes.

    Args:
        scene (dict): Parsed scene JSON.
        scene_json_path (str or Path): Path the JSON was read from.

    Returns:
        str or None: e.g. ``droid_desk_mesh``, or None when the scene has no
        identifiable background.
    """
    return background_id_from_records(
        iter_objects(scene, scene_json_path, robot_asset_dir=None, usd_facts=False)
    )


def background_id_from_records(records):
    """`background_id`, for a caller that has already iterated the scene.

    Args:
        records (iterable): Records from `iter_objects`.

    Returns:
        str or None: e.g. ``droid_desk_mesh``, or None when the scene has no
        identifiable background.
    """
    for record in records:
        if record["kind"] != "background":
            continue
        reference = record["usd_reference"] or record["usd"]
        stem = Path(str(reference)).stem if reference else ""
        return _slug(stem) or _slug(record["name"]) or None
    return None


def _different(current, authored, tolerance=1e-12):
    if not isinstance(current, (list, tuple)) or len(current) != len(authored):
        return True
    try:
        return any(abs(float(a) - float(b)) > tolerance for a, b in zip(current, authored))
    except (TypeError, ValueError):
        return True


def apply_edits(scene, edits, *, editable_names=None, posable_names=None,
                scalable_names=None, asset_facts=None):
    """Write edited poses and scales back into a scene document, in place.

    Velocities are zeroed for every edited object. Saved states carry live
    ``lin_vel``/``ang_vel`` from whenever the sim was paused, and replaying
    those against a hand-placed pose makes objects drift the moment physics
    starts.

    Args:
        scene (dict): Parsed scene JSON. Mutated.
        edits (dict): ``{name: {"position": [3], "orientation": [4], "scale": [3]}}``.
            Missing keys are left untouched.
        editable_names, posable_names, scalable_names, asset_facts:
            See `validate_edits`.

    Returns:
        list[str]: Names that were changed.
    """
    edits = validate_edits(scene, edits, editable_names=editable_names,
                           posable_names=posable_names, scalable_names=scalable_names,
                           asset_facts=asset_facts)
    registry = scene.get("state", {}).get("registry", {}).get("object_registry", {})
    init_info = scene.get("objects_info", {}).get("init_info", {})
    changed = []

    for name, edit in edits.items():
        touched = False

        root = registry.get(name, {}).get("root_link")
        if root is not None:
            if "position" in edit and _different(root.get("pos"), edit["position"]):
                root["pos"] = edit["position"]
                touched = True
            if "orientation" in edit and _different(root.get("ori"), edit["orientation"]):
                root["ori"] = edit["orientation"]
                touched = True

        # The joint vector lives in the saved state beside the pose;
        # OmniGibson restores it on load, so a value written here takes
        # effect directly.
        entry = registry.get(name)
        if "joint_values" in edit and isinstance(entry, dict):
            if _different(entry.get("joint_pos"), edit["joint_values"]):
                entry["joint_pos"] = edit["joint_values"]
                touched = True

        if "scale" in edit and name in init_info:
            args = init_info[name].setdefault("args", {})
            if _different(args.get("scale", [1.0, 1.0, 1.0]), edit["scale"]):
                args["scale"] = edit["scale"]
                touched = True

        # Physics rides in init_info.args, the dict OmniGibson hands to the
        # object's constructor. Friction lands in `link_physics_materials`, a
        # real kwarg applied to the link's collision meshes at load. Mass is
        # recorded only: OmniGibson has no load-time mass kwarg, and PhysX
        # reads `physics:mass` from the USD, which is shared across scenes --
        # a consumer applies the recorded mass after load
        # (`obj.root_link.mass = ...`).
        if ("mass" in edit or "friction" in edit
                or "joint_limits" in edit) and name in init_info:
            args = init_info[name].setdefault("args", {})
            # `_different` compares vectors and reports any scalar as changed,
            # so compare the mass directly.
            current_mass = args.get("mass")
            if "mass" in edit and edit["mass"] is None:
                # Reverted to the asset's own mass: remove the key rather than
                # writing the same number.
                if "mass" in args:
                    del args["mass"]
                    touched = True
            elif "mass" in edit and (
                isinstance(current_mass, bool)
                or not isinstance(current_mass, Real)
                or abs(float(current_mass) - edit["mass"]) > 1e-12
            ):
                args["mass"] = edit["mass"]
                touched = True
            # Limit overrides are recorded only, like mass: PhysX reads a
            # joint's range from the shared USD, so a consumer applies the
            # override after load (`obj.joints[name].lower_limit = ...`).
            if "joint_limits" in edit and edit["joint_limits"] is None:
                if "joint_limits" in args:
                    del args["joint_limits"]
                    touched = True
            elif "joint_limits" in edit:
                if args.get("joint_limits") != edit["joint_limits"]:
                    args["joint_limits"] = edit["joint_limits"]
                    touched = True
            # `link_physics_materials` is a shared map: keyed by every rigid
            # body, and entries can carry properties this panel does not edit
            # (restitution among them). Friction is merged in and taken out
            # key by key, never replaced wholesale.
            if "friction" in edit:
                # Absent and empty are the same state.
                stored = args.get("link_physics_materials")
                base = copy.deepcopy(stored) if isinstance(stored, dict) else {}
                materials = copy.deepcopy(base)
                # `material`, not `entry`: `entry` is the registry record the
                # velocity zeroing at the bottom of this loop writes to.
                if edit["friction"] is None:
                    for link, material in list(materials.items()):
                        if not isinstance(material, dict):
                            continue
                        material.pop("static_friction", None)
                        material.pop("dynamic_friction", None)
                        # Drop a link left empty by removing its friction;
                        # keep one that carries anything else.
                        if not material:
                            del materials[link]
                else:
                    # Every rigid body the server read out of the asset: the
                    # panel offers one object-wide number. An entry already
                    # there keeps whatever else it holds.
                    for link in edit["friction_links"]:
                        material = materials.get(link)
                        material = dict(material) if isinstance(material, dict) else {}
                        material["static_friction"] = edit["friction"]
                        material["dynamic_friction"] = edit["friction"]
                        materials[link] = material
                if materials != base:
                    if materials:
                        args["link_physics_materials"] = materials
                    elif "link_physics_materials" in args:
                        del args["link_physics_materials"]
                    touched = True

        if touched:
            # A scale-only edit still invalidates any saved motion state.
            if root is not None:
                if "lin_vel" in root:
                    root["lin_vel"] = [0.0, 0.0, 0.0]
                if "ang_vel" in root:
                    root["ang_vel"] = [0.0, 0.0, 0.0]
            # Zero joint velocities too: a repositioned object with a live
            # joint velocity slides its own joints the moment physics starts,
            # even when the edit never touched a joint.
            if isinstance(entry, dict) and isinstance(entry.get("joint_vel"), list):
                entry["joint_vel"] = [0.0] * len(entry["joint_vel"])
            changed.append(name)

    return changed


def add_objects(scene, specs):
    """Insert added objects into a scene document, in place.

    An object needs an entry in both places: without
    ``objects_info.init_info`` nothing constructs it, and without a
    ``state.registry.object_registry`` entry it is never given a pose and
    lands at the origin.

    Args:
        scene (dict): Parsed scene JSON. Mutated.
        specs (iterable[dict]): Specs from ``asset_library.object_spec``.

    Returns:
        list[str]: Names that were added.

    Raises:
        SceneEditError: If a name is already taken by a serialized object.
    """
    init_info = scene.setdefault("objects_info", {}).setdefault("init_info", {})
    registry = (
        scene.setdefault("state", {})
        .setdefault("registry", {})
        .setdefault("object_registry", {})
    )

    added = []
    for spec in specs:
        name = spec["name"]
        if name in init_info or name in registry:
            raise SceneEditError(f"cannot add {name}: the scene already has an object by that name")
        init_info[name] = copy.deepcopy(spec["init_info"])
        registry[name] = copy.deepcopy(spec["registry"])
        added.append(name)
    return added


def remove_objects(scene, names, *, removable_names=None):
    """Delete objects from a scene document, in place.

    Args:
        scene (dict): Parsed scene JSON. Mutated.
        names (iterable[str]): Objects to delete.
        removable_names (set[str] or None): Names this caller permits removing.
            The HTTP server passes its editable set, so a request cannot
            delete the robot or the background.

    Returns:
        list[str]: Names that were removed.
    """
    init_info = scene.get("objects_info", {}).get("init_info", {})
    registry = scene.get("state", {}).get("registry", {}).get("object_registry", {})
    known = set(init_info) | set(registry)

    removed = []
    for name in names:
        if not isinstance(name, str) or name not in known:
            raise SceneEditError(f"unknown scene object: {name!r}")
        if removable_names is not None and name not in removable_names:
            raise SceneEditError(f"object is locked and cannot be removed: {name}")
        init_info.pop(name, None)
        registry.pop(name, None)
        removed.append(name)
    return removed


# Top-level keys the simulator owns and re-derives on every save; everything
# else is additive SimFoundry state -- see merge_settled_scene.
SIMULATOR_OWNED_KEYS = ("versions", "metadata", "state", "init_info", "objects_info")


def strip_nested_scene_file(scene):
    """Remove the recursive scene snapshot from ``init_info.args.scene_file``.

    ``og.sim.save()`` embeds the entire previous scene document there; left
    alone, each settle nests one level deeper and the file grows without bound.

    Args:
        scene (dict): Scene document. Mutated in place.

    Returns:
        bool: True if a nested ``init_info`` was removed.
    """
    scene_file = scene.get("init_info", {}).get("args", {}).get("scene_file")
    if isinstance(scene_file, dict) and "init_info" in scene_file:
        del scene_file["init_info"]
        return True
    return False


def merge_settled_scene(envelope, settled, scene_json_path=None):
    """Fold a simulator serialization back into the authored document.

    ``og.sim.save()`` emits only the keys in :data:`SIMULATOR_OWNED_KEYS`; the
    authored document also carries additive fields (``ground_plane_info``,
    ``viewer_camera_state``, ``lighting_state``, ``mesh_background_state``)
    that nothing downstream can reconstruct. So the authored document is the
    envelope and the simulator supplies fresh values for the keys it owns;
    unknown top-level keys survive by construction.

    ``usd_path`` values come back from the simulator as absolute paths; a path
    that still resolves to the same file keeps the authored (relative)
    spelling, so the scene directory stays portable.

    Args:
        envelope (dict): The authored scene document, pre-settle.
        settled (dict): What ``og.sim.save()`` wrote.
        scene_json_path (str or Path or None): Where the merged document will
            live; relative ``usd_path`` values are resolved against its parent.
            None skips the path repair, which is only right when the caller
            knows both documents already agree.

    Returns:
        tuple[dict, list[str]]: The merged document, and the names of additive
        keys that the simulator output would have dropped.
    """
    merged = copy.deepcopy(envelope)
    for key in SIMULATOR_OWNED_KEYS:
        if key in settled:
            merged[key] = copy.deepcopy(settled[key])
    preserved = [k for k in merged if k not in settled]
    strip_nested_scene_file(merged)
    if scene_json_path is not None:
        restore_authored_usd_paths(merged, envelope, scene_json_path)
    return merged, preserved


def restore_authored_usd_paths(merged, envelope, scene_json_path):
    """Put back the ``usd_path`` spellings the authored document used.

    Only where the two name the *same file*: a settle that genuinely changed
    which asset an object uses keeps the simulator's answer.

    Args:
        merged (dict): Document being assembled. Mutated in place.
        envelope (dict): The authored document, holding the spellings to keep.
        scene_json_path (str or Path): Where *merged* will be written.

    Returns:
        list[str]: Objects whose path was put back.
    """
    scene_dir = Path(scene_json_path).parent
    authored = envelope.get("objects_info", {}).get("init_info", {})
    current = merged.get("objects_info", {}).get("init_info", {})
    if not isinstance(authored, dict) or not isinstance(current, dict):
        return []

    def resolved(value):
        if not value:
            return None
        candidate = Path(str(value))
        try:
            return (candidate if candidate.is_absolute()
                    else scene_dir / candidate).resolve()
        except OSError:  # pragma: no cover - a path the OS refuses to normalise
            return None

    restored = []
    for name, entry in current.items():
        args = entry.get("args") if isinstance(entry, dict) else None
        was = (authored.get(name) or {}).get("args") if isinstance(
            authored.get(name), dict) else None
        if not isinstance(args, dict) or not isinstance(was, dict):
            continue
        authored_path, now = was.get("usd_path"), args.get("usd_path")
        if not authored_path or not now or authored_path == now:
            continue
        if resolved(authored_path) is not None and resolved(authored_path) == resolved(now):
            args["usd_path"] = authored_path
            restored.append(name)
    return restored


def atomic_write_text(path, text, mode=0o644):
    """Atomically replace *path* with *text* using a sibling temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass
        raise


def exclusive_write_text(path, text, mode=0o644):
    """Create *path* with *text*, and fail if anything already has that name.

    Atomic *and* exclusive: the file is staged as a sibling temp file and
    published with `os.link`, which fails with FileExistsError if the
    destination exists. Filesystems that refuse hard links fall back to a
    plain ``O_EXCL`` create, which keeps the exclusivity and gives up only the
    atomicity.

    Raises:
        FileExistsError: If *path* already exists.
        OSError: As for any write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        try:
            os.link(tmp_name, path)
        except FileExistsError:
            raise
        except OSError:
            # No hard links on this filesystem: keep the exclusivity, give up
            # the atomicity.
            handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(text)
                out.flush()
                os.fsync(out.fileno())
    finally:
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass


class _Unchecked:
    """Sentinel for "this caller states no expectation about the target".

    Distinct from ``None``, which is itself an expectation -- "there is no file
    there" -- and is how a create is asked for.
    """

    def __repr__(self):  # pragma: no cover - debugging aid
        return "UNCHECKED"


UNCHECKED = _Unchecked()


class TargetChanged(SceneEditError):
    """A publish target no longer holds the bytes the caller last saw.

    The cross-process counterpart to the in-process ``scene_revision`` check:
    a second server, a text editor or a settle job can replace a file between
    a handler reading its digest and publishing over it.

    Attributes:
        path (Path): What was about to be written.
        expected (str or None): The digest the caller believed was there; None
            means it believed nothing was.
        found (str or None): What is actually there now, or None for absent.
    """

    def __init__(self, path, expected, found):
        self.path = Path(path)
        self.expected = expected
        self.found = found
        if expected is None:
            detail = "it did not exist when this write was planned"
        elif found is None:
            detail = "it has been deleted since this write was planned"
        else:
            detail = (f"it held {expected[:12]}… when this write was planned and "
                      f"now holds {found[:12]}…")
        super().__init__(
            f"{self.path.name} was changed by something outside this editor — "
            f"{detail}. Nothing was written."
        )


def file_digest(path):
    """SHA-256 of a file's bytes, or None when there is no such file.

    None is a value, not an error: it is how :func:`guarded_write_text` tells
    creating a file from replacing one.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


# Advisory locks live in one directory under the system temp dir, keyed by a
# digest of the resolved target path, so no lock files land beside scenes or
# checked-in config. The lock only spans processes that share a temp directory;
# everything else is caught by the digest compare in `guarded_write_text`.
_LOCK_DIR_NAME = "simfoundry-editor-locks"

# A publish never waits on a network or a subprocess; a lock held longer than
# this means another holder died badly, and blocking a request handler forever
# is worse than failing it.
LOCK_TIMEOUT_S = 30.0


def _lock_path(target):
    directory = Path(tempfile.gettempdir()) / _LOCK_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(Path(target).resolve()).encode("utf-8")).hexdigest()
    return directory / f"{digest[:32]}.lock"


@contextlib.contextmanager
def publish_lock(target, timeout=LOCK_TIMEOUT_S):
    """Hold the cross-process lock for one publish target.

    Args:
        target (str or Path): The file about to be written. The lock is keyed by
            its resolved path, so two callers spelling it differently still
            exclude each other.
        timeout (float): Seconds to wait before giving up.

    Raises:
        TimeoutError: If the lock could not be taken in *timeout* seconds.
    """
    if fcntl is None:  # pragma: no cover - this tree is Linux-only
        yield None
        return
    path = _lock_path(target)
    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"another process has been writing {Path(target).name} for "
                        f"more than {timeout:g}s"
                    ) from None
                time.sleep(0.02)
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


def guarded_write_text(path, text, *, expect, mode=0o644, atomic=True):
    """Publish *text* to *path*, but only over the bytes the caller expected.

    Compare-and-swap at the moment of publication: replacing is
    read-then-rename and so runs under the cross-process lock; creating is one
    exclusive step and needs none.

    Args:
        path (str or Path): Target.
        text (str): New contents.
        expect (str or None): The digest *path* must currently have. ``None``
            means it must not exist at all, which is how a create is asked for
            -- and which needs no lock, because an exclusive create is one step.
        mode (int): Permissions for the published file.
        atomic (bool): Publish by rename. False writes in place, which is only
            correct for a target nothing else reads concurrently.

    Returns:
        str: The digest of what was written, so the caller can hold it as the
        new expectation without re-reading the file.

    Raises:
        TargetChanged: If *path* does not currently hold *expect*.
        TimeoutError: If the publish lock could not be taken.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if expect is None:
        # A create needs no lock: `exclusive_write_text` is atomic and
        # exclusive on its own.
        try:
            exclusive_write_text(path, text, mode=mode)
        except FileExistsError:
            # Translated so every caller sees one failure kind for "the
            # target is not what you thought".
            raise TargetChanged(path, None, file_digest(path)) from None
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    with publish_lock(path):
        # Read-then-replace is not one step, so it needs the lock.
        found = file_digest(path)
        if found != expect:
            raise TargetChanged(path, expect, found)
        if atomic:
            atomic_write_text(path, text, mode=mode)
        else:  # pragma: no cover - no caller needs this yet
            path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scene_output_path(scene_json_path, suffix="light_edit"):
    """Name the unique file a save would write, without writing it.

    Split out of :func:`save_scene` so an export can name its output before
    publishing anything.

    Args:
        scene_json_path (str or Path): The source scene, used for naming.
        suffix (str): Tag embedded in the filename.

    Returns:
        Path: A path in the source scene's directory that does not exist.
    """
    src = Path(scene_json_path)
    stem = src.stem
    scene_name = stem.split("_scene_state_")[0] if "_scene_state_" in stem else stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique = uuid.uuid4().hex[:8]
    return src.parent / f"{scene_name}_scene_state_{suffix}_{timestamp}_{unique}.json"


def latest_path(scene_json_path):
    """``<scene>_scene_state_latest.json`` beside *scene_json_path*.

    The one spelling of this join, shared by every code path that promotes.
    """
    src = Path(scene_json_path)
    stem = src.stem
    scene_name = stem.split("_scene_state_")[0] if "_scene_state_" in stem else stem
    return src.parent / f"{scene_name}_scene_state_latest.json"


def scene_text(scene):
    """Serialize a scene document to the exact bytes it is written as.

    Shared so the timestamped output, the promoted ``_latest`` and any
    recomputed digest agree byte-for-byte.
    """
    return json.dumps(scene, indent=4, allow_nan=False)


def save_scene(scene, scene_json_path, suffix="light_edit", promote_latest=False,
               *, expect_latest=UNCHECKED):
    """Write a scene document beside the original, timestamped.

    Naming matches the OmniGibson editor's so the result loads unchanged via
    ``--load_scene``.

    ``_scene_state_latest.json`` is **not** updated unless ``promote_latest`` is
    set. That file is what every downstream stage picks up by default, and this
    tool has no physics to confirm an edited pose is actually resting on a
    surface — so promoting is left as a deliberate act.

    Args:
        scene (dict): Scene document to write.
        scene_json_path (str or Path): The original path, used for naming.
        suffix (str): Tag embedded in the filename.
        promote_latest (bool): Also overwrite ``_scene_state_latest.json``.
        expect_latest (str or None): The digest ``_latest`` must currently hold
            for a promotion to go ahead; None means it must not exist.
            :data:`UNCHECKED`, the default, promotes without comparing, which is
            only right for a caller with nothing to compare against.

    Returns:
        Path: The timestamped file that was written.

    Raises:
        TargetChanged: If a promotion would overwrite bytes the caller did not
            expect. The timestamped file has already been written when this
            happens: nothing is lost, but ``_latest`` is not this edit.
    """
    src = Path(scene_json_path)
    document = prepare_scene_document(scene)
    text = scene_text(document)
    out = scene_output_path(src, suffix)
    mode = scene_file_mode(src)
    # The name carries a timestamp and a uuid, so an existing file there is
    # somebody else's.
    guarded_write_text(out, text, expect=None, mode=mode)
    if promote_latest:
        promote_scene_text(text, src, expect=expect_latest, mode=mode)
    return out


def scene_file_mode(scene_json_path):
    """Permissions a file written beside *scene_json_path* should carry."""
    src = Path(scene_json_path)
    try:
        return src.stat().st_mode & 0o777
    except OSError:
        return 0o644


def prepare_scene_document(scene):
    """Copy *scene* into the shape a saved file is allowed to have.

    A nested ``init_info`` left over from a previous ``og.sim.save()`` is a
    second, stale answer to the same question, so it is stripped.
    """
    document = copy.deepcopy(scene)
    strip_nested_scene_file(document)
    return document


def promote_scene_text(text, scene_json_path, *, expect=UNCHECKED, mode=None):
    """Overwrite ``_scene_state_latest.json`` with *text*, guarding what is there.

    ``_latest`` is the file every downstream stage opens by default, and its
    writers cannot see each other, so promotion is a compare-and-swap against
    the bytes the caller last saw.

    Args:
        text (str): The document, already serialized, so the promoted file is
            byte-identical to the timestamped one beside it.
        scene_json_path (str or Path): Any scene file in the target directory;
            the ``_latest`` name is derived from it.
        expect (str or None): Digest ``_latest`` must currently hold, or None
            for "it must not exist". :data:`UNCHECKED` skips the comparison,
            which is only correct for a caller that has no expectation to state.
        mode (int or None): Permissions; derived from *scene_json_path* when
            omitted.

    Returns:
        str: The digest now on disk.

    Raises:
        TargetChanged: If ``_latest`` holds something else.
    """
    target = latest_path(scene_json_path)
    if mode is None:
        mode = scene_file_mode(scene_json_path)
    if expect is UNCHECKED:
        with publish_lock(target):
            atomic_write_text(target, text, mode=mode)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    # `expect=None` means "create it"; `guarded_write_text` tells a create
    # from a replace.
    return guarded_write_text(target, text, expect=expect, mode=mode)
