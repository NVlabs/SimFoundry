# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shlex
import sys
from pathlib import Path

import pytest

from simfoundry.pipeline.orchestrator import (
    StageSpec,
    articulation_available,
    build_cmd,
    get_stage_plan,
    run_pipeline,
    run_stage_subsequence_streaming,
)


def _write_stage_script(path: Path, stage_id: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
import json
import os
import sys
from pathlib import Path

out = None
for arg in sys.argv[1:]:
    if arg.startswith('out_dir='):
        out = arg.split('=', 1)[1]
if out is None:
    out = 'tmp_out'
Path(out).mkdir(parents=True, exist_ok=True)
with open(Path(out) / 'ran_stages.jsonl', 'a', encoding='utf-8') as f:
    f.write(json.dumps({'stage': '%s'}) + '\\n')
""".strip()
        % stage_id,
        encoding="utf-8",
    )


def test_build_cmd_modes():
    spec = StageSpec("2", "scripts/pipeline/A_reconstruction/stages/2_run_depth.py", "s2_depth", "da3", "Run depth")
    cmd = build_cmd(spec, env_map={"da3": "da3", "simfoundry": "simfoundry", "hunyuan": "hunyuan", "b1k": "b1k"}, exec_mode="mamba", python_bin="python", extra_overrides=["scene_name=x"])
    assert cmd[:4] == ["mamba", "run", "-n", "da3"]
    assert cmd[-2:] == ["scripts/pipeline/A_reconstruction/stages/2_run_depth.py", "scene_name=x"]


def test_named_stage_plans_use_new_subdirectories():
    reconstruction = get_stage_plan("video", pipeline_name="reconstruction")
    augmentation = get_stage_plan("video", pipeline_name="augmentation", include_p2p=True)
    application = get_stage_plan("video", pipeline_name="application")

    assert reconstruction[0].script == "scripts/pipeline/A_reconstruction/stages/1b_process_raw_video.py"
    # Stage 2c is opt-in via --bg-splat, so it is absent from the default plan.
    assert not any(spec.stage_id == "2c" for spec in reconstruction)
    with_splat = get_stage_plan("video", pipeline_name="reconstruction", bg_splat=True)
    stage_2c = next(spec for spec in with_splat if spec.stage_id == "2c")
    assert stage_2c.env == "nerfstudio"
    assert any(spec.script.endswith("B_augmentation/stages/8_match_cousin_p2p.py") for spec in augmentation)
    assert application[0].stage_id == "smoke"
    assert application[1].script.endswith("C_application/stages/1_eval_policy_og_scene.py")


def test_stage_2c_uses_dedicated_nerfstudio_environment():
    stage_2c = next(
        spec
        for spec in get_stage_plan("video", pipeline_name="reconstruction", bg_splat=True)
        if spec.stage_id == "2c"
    )

    cmd = build_cmd(
        stage_2c,
        env_map={"nerfstudio": "custom-nerfstudio"},
        exec_mode="mamba",
        python_bin="python",
        extra_overrides=[],
    )

    assert cmd[:4] == ["mamba", "run", "-n", "custom-nerfstudio"]
    assert cmd[5] == "scripts/pipeline/A_reconstruction/stages/2c_train_bg_splat.py"


@pytest.mark.skipif(
    not articulation_available(),
    reason="articulation stage 9 is not available in this release",
)
def test_reconstruction_stage_plan_can_insert_articulation():
    default_ids = [spec.stage_id for spec in get_stage_plan("video", pipeline_name="reconstruction")]
    articulation_ids = [spec.stage_id for spec in get_stage_plan("video", pipeline_name="reconstruction", detect_articulation=True)]

    assert "9" not in default_ids
    assert articulation_ids[articulation_ids.index("8") + 1] == "9"
    assert articulation_ids[articulation_ids.index("9") + 1] == "10"


def test_run_pipeline_partial_multi_step(tmp_path, monkeypatch):
    cwd = tmp_path
    out_dir = tmp_path / "out"

    s1 = tmp_path / "scripts/pipeline/s1.py"
    s2 = tmp_path / "scripts/pipeline/s2.py"
    s3 = tmp_path / "scripts/pipeline/s3.py"
    _write_stage_script(s1, "1")
    _write_stage_script(s2, "2")
    _write_stage_script(s3, "3")

    def fake_plan(input_mode: str, **_kwargs):
        del input_mode
        return [
            StageSpec("1", os.path.relpath(s1, cwd), "s1", "simfoundry", "one"),
            StageSpec("2", os.path.relpath(s2, cwd), "s2", "simfoundry", "two"),
            StageSpec("3", os.path.relpath(s3, cwd), "s3", "simfoundry", "three"),
        ]

    import simfoundry.pipeline.orchestrator as orch

    monkeypatch.setattr(orch, "get_stage_plan", fake_plan)

    run_pipeline(
        cwd=str(cwd),
        input_mode="video",
        include_ids_csv="1,3",
        exclude_ids_csv=None,
        exec_mode="direct",
        python_bin=sys.executable,
        env_map={"simfoundry": "simfoundry", "da3": "da3", "hunyuan": "hunyuan", "b1k": "b1k"},
        dry_run=False,
        stream_subseq_enabled=False,
        stream_start_stage=5,
        stream_end_stage=8,
        extra_overrides=[f"out_dir={out_dir}", "scene_name=scene_under_test", f"root_dir={tmp_path / 'Data'}"],
    )

    lines = [json.loads(l) for l in (out_dir / "ran_stages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["stage"] for x in lines] == ["1", "3"]

    timing_log = tmp_path / "Data" / "scene_under_test" / "pipeline_timing.log"
    log_text = timing_log.read_text(encoding="utf-8")
    assert "stage=1" in log_text
    assert "stage=3" in log_text
    assert "stage=WALL" in log_text


def test_run_pipeline_stream_collapse(monkeypatch, tmp_path):
    calls = {"stream": 0, "single": []}

    import simfoundry.pipeline.orchestrator as orch

    def fake_plan(input_mode: str, **_kwargs):
        del input_mode
        return [
            StageSpec("6", "scripts/pipeline/A_reconstruction/stages/6_upsample_object_images.py", "s6", "simfoundry", "six"),
            StageSpec("7", "scripts/pipeline/A_reconstruction/stages/7_generate_object_meshes.py", "s7", "hunyuan", "seven"),
            StageSpec("8", "scripts/pipeline/A_reconstruction/stages/8_match_object_poses.py", "s8", "simfoundry", "eight"),
        ]

    def fake_stream(**_kwargs):
        calls["stream"] += 1
        return 1.0

    def fake_run_stage(spec, **_kwargs):
        calls["single"].append(spec.stage_id)
        return 0.1

    monkeypatch.setattr(orch, "get_stage_plan", fake_plan)
    monkeypatch.setattr(orch, "run_stage_subsequence_streaming", fake_stream)
    monkeypatch.setattr(orch, "run_stage", fake_run_stage)

    run_pipeline(
        cwd=".",
        input_mode="video",
        include_ids_csv=None,
        exclude_ids_csv=None,
        exec_mode="direct",
        python_bin="python",
        env_map={"simfoundry": "simfoundry", "da3": "da3", "hunyuan": "hunyuan", "b1k": "b1k"},
        dry_run=False,
        stream_subseq_enabled=True,
        stream_start_stage=6,
        stream_end_stage=7,
        # root_dir must be overridden, or the report/timing artifacts land in the repo's own Data/.
        extra_overrides=[f"root_dir={tmp_path}", "scene_name=scene_under_test"],
    )

    assert calls["stream"] == 1
    assert calls["single"] == ["8"]


def test_streaming_stage_cmds_forward_overrides(monkeypatch):
    captured = {}

    def fake_subprocess_run(cmd, cwd, check, env):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["check"] = check
        captured["env"] = env

    import simfoundry.pipeline.orchestrator as orch

    monkeypatch.setattr(orch.subprocess, "run", fake_subprocess_run)

    run_stage_subsequence_streaming(
        stream_start_stage=5,
        stream_end_stage=6,
        env_map={"simfoundry": "simfoundry", "da3": "da3", "hunyuan": "hunyuan", "b1k": "b1k"},
        exec_mode="direct",
        python_bin="python",
        cwd=".",
        dry_run=False,
        extra_overrides=["scene_name=dining_1", "root_dir=/tmp/Data"],
        timing_log_path=None,
    )

    cmd = captured["cmd"]
    overrides = {
        arg.split("=", 1)[0]: arg.split("=", 1)[1]
        for arg in cmd[3:]
        if "=" in arg
    }
    s5_cmd = shlex.split(json.loads(overrides["stream_subseq.s5_cmd"]))
    s6_cmd = shlex.split(json.loads(overrides["stream_subseq.s6_cmd"]))

    assert s5_cmd[-2:] == ["scene_name=dining_1", "root_dir=/tmp/Data"]
    assert s6_cmd[-2:] == ["scene_name=dining_1", "root_dir=/tmp/Data"]


def test_skip_successful_filters_recorded_stages(tmp_path):
    from simfoundry.pipeline.orchestrator import filter_previously_successful

    overrides = [f"root_dir={tmp_path}", "scene_name=scene"]
    specs = [
        StageSpec("2", "x.py", "s2_depth", "da3", "Run depth"),
        StageSpec("3", "x.py", "s3_ground", "simfoundry", "Segment ground plane"),
        StageSpec("4", "x.py", "s4_frame", "simfoundry", "Unify world frame"),
        StageSpec("5", "x.py", "s5_scene", "simfoundry", "Decompose scene"),
    ]

    def mark(dirname, success):
        stage_dir = tmp_path / "scene" / dirname
        stage_dir.mkdir(parents=True)
        (stage_dir / "stage_info.json").write_text(json.dumps({"success": success}))

    mark("s2_da", True)     # stage 2 records under the selected backend's dir (da3 default)
    mark("s3_ground", True)
    mark("s4_frame", False)  # completed but unsuccessful -> must re-run
    # stage 5 has no marker at all -> must run

    kept = filter_previously_successful(specs, overrides)
    assert [s.stage_id for s in kept] == ["4", "5"]
