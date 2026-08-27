# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finding the task YAML a run asked for.

Hydra names a task by config-group path and the stages name it by `task_name`,
and the two stopped agreeing the moment a task config was filed in a
subdirectory. The failure was expensive: the eval connected to the policy,
loaded the scene, and only then died naming a file nobody had asked for.
"""

import pytest

from simfoundry.utils.python_utils import resolve_task_config_path


def write(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("og_task_config: {}\n", encoding="utf-8")
    return path


def test_a_flat_config_is_found_by_task_name(tmp_path):
    wanted = write(tmp_path / "task" / "serve_fruits.yaml")
    assert resolve_task_config_path(tmp_path, "serve_fruits") == str(wanted)


def test_a_config_in_a_subdirectory_is_found_by_the_group_hydra_chose(tmp_path):
    """`task=droid/cluttered_scene/x` still carries a bare `task_name`."""
    wanted = write(tmp_path / "task" / "droid" / "cluttered_scene" / "place_ball.yaml")
    found = resolve_task_config_path(
        tmp_path, "place_ball", group_choice="droid/cluttered_scene/place_ball")
    assert found == str(wanted)


def test_the_scene_specific_copy_wins(tmp_path):
    write(tmp_path / "task" / "serve_fruits.yaml")
    wanted = write(tmp_path / "task" / "nv_desk" / "serve_fruits.yaml")
    found = resolve_task_config_path(tmp_path, "serve_fruits", scene_name="nv_desk")
    assert found == str(wanted)


def test_an_overridden_task_name_beats_the_group(tmp_path):
    """`task=load_scene task.task_name=serve_the_orange` is how a scene picks
    its own task, so the name has to keep winning over the group."""
    write(tmp_path / "task" / "load_scene.yaml")
    wanted = write(tmp_path / "task" / "serve_the_orange.yaml")
    found = resolve_task_config_path(
        tmp_path, "serve_the_orange", group_choice="load_scene")
    assert found == str(wanted)


def test_nothing_found_names_every_path_it_tried(tmp_path):
    with pytest.raises(FileNotFoundError) as caught:
        resolve_task_config_path(
            tmp_path, "missing", group_choice="droid/missing", scene_name="nv_desk")
    message = str(caught.value)
    assert "task/nv_desk/missing.yaml" in message
    assert "task/missing.yaml" in message
    assert "task/droid/missing.yaml" in message
