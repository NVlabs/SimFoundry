#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

eval "$(mamba shell hook --shell bash)"

mamba activate simfoundry_teleop

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_PATH="${REPO_ROOT}/Data"
SCENE_NAME=droid_desk_put_away_trash

python interactive_scene_editor.py \
--scene_name ${SCENE_NAME} \
--mesh_background ${REPO_ROOT}/assets/mesh_backgrounds/droid_desk_mesh.usd \
--cam2world ${DATA_PATH}/${SCENE_NAME}/s4_frame/image_0_cam2world.npy \
--scene_objects_info ${DATA_PATH}/${SCENE_NAME}/s10_sim/scene_objects_info.json \
--pb_scene_poses ${DATA_PATH}/${SCENE_NAME}/s11_physics/pb_scene_poses.json \
--scene_objects_categories blue_cup black_trash_can \
--robot FrankaPanda:robotiq

# Example: resume from a saved scene state
# python -m pdb interactive_scene_editor.py \
#   --load_scene ${REPO_ROOT}/assets/scenes/droid_desk_stack_dishware/droid_desk_stack_dishware_scene_state_latest.json \
#   --scene_name droid_desk_stack_dishware
