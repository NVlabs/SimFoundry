# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pose a robot's links from the joint configuration saved in a scene state.

A saved scene stores the arm as ``state.registry.object_registry.<name>.joint_pos``
while the robot's USD holds only its rest pose. This module rebuilds the kinematic
tree from ``UsdPhysics.Joint`` prims and runs forward kinematics with standalone
``usd-core`` — no OmniGibson, no Isaac Sim, no GPU — producing per-link correction
matrices that :mod:`extract` applies while baking meshes.

Scope: arm joints only. The pinned joints in :data:`ROBOT_ARM_JOINTS` stop at the
wrist; gripper linkages can close kinematic loops a tree walk cannot solve, so every
unpinned joint is evaluated at zero — exactly the configuration the USD authors, so
the hand rides rigidly on the wrist at its rest opening.

``joint_pos`` is a bare positional array in PhysX articulation order, which is not
recoverable from USD traversal order, so each supported robot pins its arm joints
explicitly, keyed exactly as ``scene_io.ROBOT_ASSETS``. Anything that does not line
up — an unknown robot, a joint name the USD does not carry, a chain whose zero pose
disagrees with the authored rest pose — returns None and the caller falls back to
the authored geometry.
"""

import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parent))
import usd_cache  # noqa: E402
from scene_io import SceneEditError  # noqa: E402

# Arm joints of every robot ``scene_io.ROBOT_ASSETS`` can resolve, in the order
# their values appear in ``joint_pos`` (PhysX articulation order). Keys mirror
# that table exactly, so a robot added there and not here degrades to the
# authored rest pose rather than to a wrong one.
ROBOT_ARM_JOINTS = {
    ("FrankaPanda", "robotiq"): tuple(f"panda_joint{i}" for i in range(1, 8)),
    ("FrankaPanda", "gripper"): tuple(f"panda_joint{i}" for i in range(1, 8)),
    ("FrankaPanda", "panda"): tuple(f"panda_joint{i}" for i in range(1, 8)),
    ("FrankaPanda", None): tuple(f"panda_joint{i}" for i in range(1, 8)),
    ("FrankaPanda", "inspire"): tuple(f"panda_joint{i}" for i in range(1, 8)),
    ("FrankaPanda", "leap"): tuple(f"panda_joint{i}" for i in range(1, 8)),
    ("FrankaPanda", "allegro"): tuple(f"panda_joint{i}" for i in range(1, 8)),
    # Unverified against the Yam USD, but checked before use: solve_arm_pose
    # rejects a USD without joints named joint1..6, and the zero-configuration
    # check rejects a misread chain.
    ("Yam", None): tuple(f"joint{i}" for i in range(1, 7)),
}

# Joint values below this are settled-simulation noise, not articulation anyone
# can see. Used to decide whether an unposed object is worth reporting.
JOINT_EPS = 1e-4

# How far a link computed at zero may sit from where the USD authors it before
# the solve is rejected: at zero the two must agree, because the authored pose
# *is* the zero configuration. This catches a misread axis, a units mistake or
# a reversed parent/child. 1 mm is loose enough for float32 round-off and far
# tighter than any real error.
REST_TOLERANCE = 1e-3

_AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def _rigid(pos, quat):
    """Build a USD-convention (row-vector) 4x4 from a translation and a quaternion.

    Args:
        pos (Gf.Vec3f or None): Translation; None means zero.
        quat (Gf.Quatf or None): Rotation; None means identity.

    Returns:
        np.ndarray: (4, 4) matrix with the translation in the last row, matching
            ``UsdGeom.XformCache.GetLocalToWorldTransform``.
    """
    matrix = np.eye(4)
    if quat is not None:
        rotation = Gf.Rotation(Gf.Quatd(quat.GetReal(), Gf.Vec3d(*quat.GetImaginary())))
        matrix[:3, :3] = np.asarray(Gf.Matrix3d(rotation), dtype=np.float64)
    if pos is not None:
        matrix[3, :3] = np.asarray(pos, dtype=np.float64)
    return matrix


def _joint_motion(joint_type, axis, value, meters_per_unit):
    """Motion of a joint's child frame relative to its parent frame at *value*.

    Args:
        joint_type (str): USD type name, e.g. ``PhysicsRevoluteJoint``.
        axis (str): Axis token, one of ``X``/``Y``/``Z``.
        value (float): Joint coordinate as OmniGibson stores it — radians for a
            revolute joint, metres for a prismatic one.
        meters_per_unit (float): Stage metric, used to convert a prismatic value
            into the stage's linear units.

    Returns:
        np.ndarray: (4, 4) row-vector matrix; identity for a joint with no DOF.
    """
    direction = _AXES[axis]
    if joint_type == "PhysicsRevoluteJoint":
        # Gf.Rotation takes degrees; joint_pos is radians, so the conversion
        # belongs here and nowhere else.
        rotation = Gf.Rotation(Gf.Vec3d(*direction), float(np.degrees(value)))
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(Gf.Matrix3d(rotation), dtype=np.float64)
        return matrix
    if joint_type == "PhysicsPrismaticJoint":
        matrix = np.eye(4)
        matrix[3, :3] = np.asarray(direction) * (float(value) / meters_per_unit)
        return matrix
    return np.eye(4)


class Joint:
    """One articulation edge, reduced to what forward kinematics needs."""

    def __init__(self, prim):
        """Read a joint prim.

        Args:
            prim (Usd.Prim): A prim typed as some ``UsdPhysics`` joint.
        """
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        self.name = prim.GetName()
        self.path = str(prim.GetPath())
        self.type = str(prim.GetTypeName())
        # An empty body0 means "attached to the world"; the child is then a
        # root of the tree rather than the child of anything.
        self.parent = str(body0[0]) if body0 else None
        self.child = str(body1[0]) if body1 else None
        self.axis = "X"
        if self.type == "PhysicsRevoluteJoint":
            self.axis = str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() or "X")
        elif self.type == "PhysicsPrismaticJoint":
            self.axis = str(UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get() or "X")
        self.parent_frame = _rigid(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        self.child_frame = _rigid(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())

    @property
    def has_dof(self):
        """Whether this joint contributes a value to ``joint_pos``."""
        return self.type in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")

    def child_to_world(self, parent_to_world, value, meters_per_unit):
        """Place this joint's child body given its parent's world transform.

        Args:
            parent_to_world (np.ndarray): (4, 4) row-vector transform of body0.
            value (float): Joint coordinate (radians or metres).
            meters_per_unit (float): Stage metric.

        Returns:
            np.ndarray: (4, 4) row-vector transform of body1.
        """
        motion = _joint_motion(self.type, self.axis, value, meters_per_unit)
        # Row-vector composition, read right to left: start in the parent body,
        # step to the joint frame on the parent (parent_frame), apply the joint's
        # motion, then step back out through the joint frame on the child.
        return np.linalg.inv(self.child_frame) @ motion @ self.parent_frame @ parent_to_world


class Articulation:
    """The joint graph of one USD stage, plus the rest transform of every body."""

    def __init__(self, stage):
        """Read every joint and rigid body on a stage.

        Args:
            stage (Usd.Stage): An opened stage.
        """
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        self.meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0
        self.joints = []
        self.authored = {}
        bodies = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                path = str(prim.GetPath())
                bodies.append(path)
                self.authored[path] = np.asarray(
                    cache.GetLocalToWorldTransform(prim), dtype=np.float64
                )
            if prim.IsA(UsdPhysics.Joint):
                self.joints.append(Joint(prim))

        self.by_name = {}
        self.children = {}
        attached = set()
        for joint in self.joints:
            self.by_name.setdefault(joint.name, joint)
            if joint.parent is None or joint.child is None:
                continue
            self.children.setdefault(joint.parent, []).append(joint)
            attached.add(joint.child)
        # A body nothing hangs off is a root. Anchoring joints (empty body0) leave
        # their child unattached here, which is what makes such a base a root too.
        self.roots = [body for body in bodies if body not in attached]

    def solve(self, values):
        """Run forward kinematics from every root.

        Args:
            values (dict[str, float]): Joint name to coordinate. Joints absent
                from this mapping are evaluated at zero, which for an unpinned
                joint reproduces the pose the USD authors.

        Returns:
            tuple[dict[str, np.ndarray], set[str]]: Body path to world transform,
                and the names of joints whose child was already placed — loop
                closures the tree cannot represent.
        """
        world = {root: self.authored[root].copy() for root in self.roots}
        skipped = set()
        queue = list(self.roots)
        while queue:
            parent = queue.pop(0)
            for joint in self.children.get(parent, ()):
                if joint.child in world:
                    skipped.add(joint.name)
                    continue
                if joint.child not in self.authored:
                    # A joint target that is not a rigid body has no rest frame
                    # to correct against.
                    skipped.add(joint.name)
                    continue
                world[joint.child] = joint.child_to_world(
                    world[parent], float(values.get(joint.name, 0.0)), self.meters_per_unit
                )
                queue.append(joint.child)
        return world, skipped


class RobotJointPose:
    """Per-link corrections that carry a robot from its rest pose to a saved one."""

    def __init__(self, robot, joint_names, joint_values, corrections, posed_joints):
        """Hold a solved pose.

        Args:
            robot (str): Label for the pinned robot, e.g. ``FrankaPanda/robotiq``.
            joint_names (tuple[str, ...]): Arm joints that were applied.
            joint_values (tuple[float, ...]): Their values, in the same order.
            corrections (dict[str, np.ndarray]): Body prim path to a (4, 4)
                row-vector matrix that maps rest-baked points to posed points.
            posed_joints (int): Number of joints the tree walk placed.
        """
        self.robot = robot
        self.joint_names = tuple(joint_names)
        self.joint_values = tuple(float(v) for v in joint_values)
        self.corrections = corrections
        self.posed_joints = posed_joints

    def correction_for(self, prim_path):
        """Find the correction that applies to a prim, by nearest posed ancestor.

        Mesh prims live *under* the link they belong to, so the lookup has to walk
        up the path. A prim under no posed link gets None and is baked as authored.

        Args:
            prim_path (str or Sdf.Path): Absolute prim path.

        Returns:
            np.ndarray or None: (4, 4) row-vector matrix.
        """
        path = str(prim_path)
        while path and path != "/":
            correction = self.corrections.get(path)
            if correction is not None:
                return correction
            path = path.rsplit("/", 1)[0]
        return None

    def as_manifest(self):
        """Describe this pose for the manifest, so the browser can tell it happened.

        Returns:
            dict: JSON-safe record with ``applied`` true.
        """
        return {
            "applied": True,
            "robot": self.robot,
            "joints": list(self.joint_names),
            "values": [float(v) for v in self.joint_values],
            "posedLinks": int(self.posed_joints),
            "note": "arm joints only; gripper/hand joints left at their rest pose",
        }


def robot_key(class_name, end_effector):
    """Key a robot the same way ``scene_io.ROBOT_ASSETS`` does.

    Args:
        class_name (str or None): ``objects_info.init_info[name].class_name``.
        end_effector (str or None): ``args.end_effector``.

    Returns:
        tuple: ``(class_name, end_effector)``.
    """
    return (class_name, end_effector)


def solve_arm_pose(usd_path, class_name, end_effector, joint_pos, on_warning=print):
    """Compute per-link corrections for a robot at its saved joint configuration.

    Args:
        usd_path (str or Path): Robot USD, as resolved by
            ``scene_io.resolve_robot_usd``.
        class_name (str or None): Serialized robot class, e.g. ``FrankaPanda``.
        end_effector (str or None): Serialized ``end_effector`` argument.
        joint_pos (list[float] or None): Saved ``joint_pos`` array.
        on_warning (callable): Receives one string per reason the pose was not
            applied. Defaults to ``print``.

    Returns:
        RobotJointPose or None: None whenever the robot is unknown, the USD does
            not carry the pinned joints, the values are unusable, or the chain
            disagrees with the USD's own rest pose. The caller must then fall back
            to the authored geometry.

    Raises:
        SceneEditError: If *usd_path* is None — the caller is expected to have
            resolved the asset already.
    """
    if usd_path is None:
        raise SceneEditError("solve_arm_pose needs a resolved robot USD path")

    label = f"{class_name}/{end_effector}" if end_effector else str(class_name)
    arm_joints = ROBOT_ARM_JOINTS.get(robot_key(class_name, end_effector))
    if arm_joints is None:
        on_warning(
            f"{label}: no pinned joint ordering for this robot; drawing its rest pose. "
            "Add it to robot_pose.ROBOT_ARM_JOINTS once its joint order is verified."
        )
        return None

    if not isinstance(joint_pos, (list, tuple)) or len(joint_pos) < len(arm_joints):
        on_warning(
            f"{label}: saved joint_pos has {0 if not joint_pos else len(joint_pos)} value(s), "
            f"fewer than the {len(arm_joints)} arm joints; drawing its rest pose."
        )
        return None
    try:
        values = [float(v) for v in joint_pos[: len(arm_joints)]]
    except (TypeError, ValueError):
        on_warning(f"{label}: saved joint_pos is not numeric; drawing its rest pose.")
        return None
    if not all(np.isfinite(values)):
        on_warning(f"{label}: saved joint_pos contains non-finite values; drawing its rest pose.")
        return None

    stage = usd_cache.open_stage(usd_path)
    if stage is None:
        on_warning(f"{label}: could not open {usd_path}; drawing its rest pose.")
        return None

    articulation = Articulation(stage)
    missing = [name for name in arm_joints if name not in articulation.by_name]
    if missing:
        on_warning(
            f"{label}: {Path(usd_path).name} has no joint(s) named {', '.join(missing)}; "
            "the pinned ordering does not describe this asset. Drawing its rest pose."
        )
        return None
    if not articulation.roots:
        on_warning(f"{label}: articulation has no root body; drawing its rest pose.")
        return None

    named = dict(zip(arm_joints, values))
    rest_world, skipped = articulation.solve({})
    posed_world, _ = articulation.solve(named)

    # A pinned joint the walk could not follow means the arm itself is not a
    # tree here, and the values would land on the wrong links.
    unfollowed = [name for name in arm_joints if name in skipped]
    if unfollowed:
        on_warning(
            f"{label}: arm joint(s) {', '.join(unfollowed)} close a loop in the "
            "articulation and cannot be posed; drawing its rest pose."
        )
        return None

    # The authored transforms *are* the zero configuration; checking the arm
    # links rules out a misread axis or radians/degrees slip before drawing.
    for name in arm_joints:
        child = articulation.by_name[name].child
        computed = rest_world.get(child)
        authored = articulation.authored.get(child)
        if computed is None or authored is None:
            on_warning(f"{label}: joint {name} has no rigid-body child; drawing its rest pose.")
            return None
        error = float(np.abs(computed - authored).max())
        if error > REST_TOLERANCE:
            on_warning(
                f"{label}: zero configuration puts {child.rsplit('/', 1)[-1]} {error:.4g} from "
                "where the USD authors it, so this chain is not being read correctly; "
                "drawing its rest pose."
            )
            return None

    # Correct against the computed zero pose, not the authored one: some leaf
    # frames are authored off their own joint frames, and referencing the zero
    # pose keeps them where the asset puts them while riding rigidly with the arm.
    corrections = {}
    for path, posed in posed_world.items():
        rest = rest_world[path]
        correction = np.linalg.inv(rest) @ posed
        if np.abs(correction - np.eye(4)).max() > 1e-9:
            corrections[path] = correction

    return RobotJointPose(label, arm_joints, values, corrections, len(posed_world))


def saved_joint_pos(scene, name):
    """Read one object's saved ``joint_pos`` out of a scene state.

    Args:
        scene (dict): Parsed scene JSON.
        name (str): Object name.

    Returns:
        list or None: The saved array, or None when the object stores no joints.

    Raises:
        SceneEditError: If *name* is not in the scene's object registry.
    """
    registry = scene.get("state", {}).get("registry", {}).get("object_registry", {})
    if name not in registry:
        raise SceneEditError(f"unknown scene object: {name!r}")
    joint_pos = registry[name].get("joint_pos")
    return joint_pos if joint_pos else None


def is_articulated(joint_pos):
    """Whether a saved ``joint_pos`` describes a visibly non-rest configuration.

    A settled scene stores numerical noise for every joint it never moved, so
    "has joints" is not the same question as "is drawn wrong when ignored".

    Args:
        joint_pos (list or None): Saved joint array.

    Returns:
        bool: True when some joint is at least :data:`JOINT_EPS` from zero.
    """
    if not joint_pos:
        return False
    try:
        return bool(max(abs(float(v)) for v in joint_pos) > JOINT_EPS)
    except (TypeError, ValueError):
        return False
