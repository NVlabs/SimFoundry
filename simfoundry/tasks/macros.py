# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OmniGibson macro overrides and hand-tuned robot constants for SimFoundry tasks.

Importing this module applies the macro overrides below as a side effect. Task
modules (e.g. ``pick_place_task``) import it so the overrides are in place
whenever a SimFoundry task is used.
"""

import torch as th
from omnigibson.macros import macros
from omnigibson.object_states.open_state import m as open_state_macros
from omnigibson.utils.constants import JointType

open_state_macros.JOINT_THRESHOLD_BY_TYPE = {
    JointType.JOINT_REVOLUTE: 0.1,  # TODO: maybe adjust later
    JointType.JOINT_PRISMATIC: 0.1,
}

# Override specific omnigibson macros specific for this task
macros.utils.object_state_utils.ON_TOP_RAY_CASTING_SAMPLING_PARAMS.verify_cuboid_empty = False
macros.utils.sampling_utils.DEFAULT_HIT_PROPORTION = 0.7

# Hand-tuned joint gains, limits, and effort/velocity caps per robot model
GAINS = {
    "franka_panda": {
        "kp": th.tensor([400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 25.0, 25.0]),
        "kv": th.tensor([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 10.0, 10.0]),
        "max_velocity": th.tensor([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 1.0, 1.0]),
        "max_effort": th.tensor([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 24.0, 24.0]),
        "joint_lower_limits": th.tensor([-2.6973, -1.5628, -2.6973, -2.8718, -2.6973, 0.1175, -2.773, 0.0, 0.0]),
        "joint_upper_limits": th.tensor([2.6973, 1.5628, 2.6973, -0.2698, 2.6973, 3.5525, 2.6973, 0.04, 0.04]),
    },
    "franka_robotiq": {
        "kp": th.tensor([400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 25.0, 25.0]),
        "kv": th.tensor([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 10.0, 10.0]),
        "max_velocity": th.tensor([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 1.0, 1.0]),
        "max_effort": th.tensor([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 24.0, 24.0]),
        "joint_lower_limits": th.tensor([-2.6973, -1.5628, -2.6973, -2.8718, -2.6973, 0.1175, -2.773,  0.0000,
         0.0000]),
        "joint_upper_limits": th.tensor([2.6973,  1.5628,  2.6973, -0.2698,  2.6973,  3.5525,  2.6973, 0.7854, 0.7854]),
    },
    "Yam": {
        "kp": th.tensor([400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0]),
        "kv": th.tensor([25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 10.0, 10.0]),
        "max_velocity": th.tensor([3.49, 3.49, 3.49, 10.47, 10.47, 10.47, 0.5, 0.5]),
        "max_effort": th.tensor([87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 120.0, 120.0]),
    },
}
