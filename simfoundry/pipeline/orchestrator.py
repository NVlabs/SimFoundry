# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate script-based SimFoundry sub-pipelines with configurable environments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Iterable

from simfoundry import configure_omnigibson_data_path
from simfoundry.pipeline.reporting import write_pipeline_report
from simfoundry.pipeline.resource_scheduler import query_gpu_memory_used_gb


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    script: str
    cfg_key: str
    # Env role key ("simfoundry", "nerfstudio", "da3", "mesh", "b1k"), resolved
    # through env_map to a concrete mamba env. "mesh" runs in hunyuan or simfoundry
    # depending on the backend.
    env: str
    description: str


@dataclass(frozen=True)
class PipelineTimingResult:
    stage_durations_s: dict[str, float]
    wall_time_s: float
    memory_samples: dict[str, dict[str, float | None]]
    report_path: Path | None = None
    manifest_path: Path | None = None


TIMING_LOG_ENV_VAR = "SIMFOUNDRY_PIPELINE_TIMING_LOG"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG_PATH = REPO_ROOT / "scripts" / "cfg" / "real2sim_cfg.yaml"

# Stage 8b (articulation) is optional: it joins the plan only when it can actually run, so an
# install without it degrades to a warning rather than a failure.
ARTICULATION_STAGE_SCRIPT = "scripts/pipeline/A_reconstruction/stages/8b_articulate_objects.py"
ARTICULATION_DEPS_DIR = "deps/articulate-anything"
# Mirrors CONDA_ENVS in 8b_articulate_objects.py; stage 8b shells out to these.
ARTICULATION_ENVS = ("articulate-anything-hunyuan", "articulate-anything-partfield")


def _conda_envs_dirs() -> list[Path]:
    """Directories that may contain named conda envs, derived without shelling out."""
    candidates = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(Path(conda_exe).resolve().parent.parent / "envs")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix).resolve()
        # Either <base>/envs/<name> (so envs is the parent) or the base env itself.
        candidates.append(prefix.parent if prefix.parent.name == "envs" else prefix / "envs")
    return [d for d in dict.fromkeys(candidates) if d.is_dir()]


def articulation_available() -> bool:
    """Whether stage 8b can actually run here.

    The stage script alone is not enough: it ships in every checkout, so testing only for it
    meant the documented "ignored with a warning" path never triggered and stage 8b was
    scheduled into plans that could not run it. Stage 8b also needs the articulate-anything
    checkout and one of its conda envs, so check for those too. When the conda layout cannot be
    determined, report unavailable — degrading to a warning is the documented behaviour and is
    safer than scheduling a stage that will fail mid-run.
    """
    if not (REPO_ROOT / ARTICULATION_STAGE_SCRIPT).is_file():
        return False
    if not (REPO_ROOT / ARTICULATION_DEPS_DIR).is_dir():
        return False
    envs_dirs = _conda_envs_dirs()
    return any((envs_dir / name).is_dir() for envs_dir in envs_dirs for name in ARTICULATION_ENVS)


def format_duration(seconds: float) -> str:
    """Format a duration in a compact human-readable form."""
    if seconds < 60:
        return f"{seconds:.2f}s"

    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.2f}s"

    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {rem:.2f}s"


def _parse_overrides(extra_overrides: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in extra_overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def _load_default_runtime_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in DEFAULT_CFG_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for key in ("root_dir", "scene_name"):
            prefix = f"{key}:"
            if stripped.startswith(prefix):
                values[key] = stripped[len(prefix):].split("#", 1)[0].strip().strip("'\"")
        if len(values) == 2:
            break
    if "root_dir" not in values or "scene_name" not in values:
        raise RuntimeError(f"Failed to read root_dir/scene_name defaults from {DEFAULT_CFG_PATH}")
    return values


def resolve_scene_dir(extra_overrides: list[str]) -> Path:
    overrides = _parse_overrides(extra_overrides)
    defaults = _load_default_runtime_values()

    scene_name = overrides.get("scene_name", defaults["scene_name"])
    root_dir_value = overrides.get("root_dir", defaults["root_dir"])
    root_dir = Path(root_dir_value)
    if not root_dir.is_absolute():
        root_dir = (DEFAULT_CFG_PATH.parent / root_dir).resolve()

    return root_dir / scene_name


def resolve_scene_timing_log_path(extra_overrides: list[str]) -> Path:
    scene_dir = resolve_scene_dir(extra_overrides)
    scene_dir.mkdir(parents=True, exist_ok=True)
    return scene_dir / "pipeline_timing.log"


def append_timing_log(log_path: Path, *, stage_id: str, description: str, elapsed_s: float, status: str = "completed") -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{timestamp} | stage={stage_id} | status={status} | elapsed_s={elapsed_s:.6f} | elapsed={format_duration(elapsed_s)} | {description}\n"
        )


def get_reconstruction_stage_plan(input_mode: str, *, detect_articulation: bool = False, bg_splat: bool = False) -> list[StageSpec]:
    """Return ordered stage plan for the real2sim reconstruction pipeline."""
    if input_mode not in {"video", "stereo"}:
        raise ValueError(f"Unsupported input_mode={input_mode}")

    base = "scripts/pipeline/A_reconstruction/stages"
    step1 = StageSpec("1b", f"{base}/1b_process_raw_video.py", "s1_video", "simfoundry", "Process raw video")
    if input_mode == "stereo":
        step1 = StageSpec("1a", f"{base}/1a_take_stereo_images.py", "s1_zed", "simfoundry", "Capture stereo images")

    plan = [step1]
    if bg_splat:
        plan.append(StageSpec("2c", f"{base}/2c_train_bg_splat.py", "s2c_gs", "nerfstudio", "Train background GS splat"))
    plan += [
        StageSpec("2", f"{base}/2_run_depth.py", "s2_depth", "da3", "Run depth backend"),
        StageSpec("3", f"{base}/3_segment_ground_plane.py", "s3_ground", "simfoundry", "Segment ground plane"),
        StageSpec("4", f"{base}/4_unify_world_frame.py", "s4_frame", "simfoundry", "Unify world frame"),
        StageSpec("5", f"{base}/5_decompose_scene.py", "s5_scene", "simfoundry", "Decompose scene"),
        StageSpec("6", f"{base}/6_upsample_object_images.py", "s6_upsample", "simfoundry", "Upsample objects"),
        StageSpec("7", f"{base}/7_generate_object_meshes.py", "s7_mesh", "mesh", "Generate meshes"),
        StageSpec("8", f"{base}/8_match_object_poses.py", "s8_pose", "simfoundry", "Match object poses"),
    ]
    if detect_articulation:
        if articulation_available():
            plan.append(StageSpec("8b", f"{base}/8b_articulate_objects.py", "s8b_articulate_objects", "simfoundry", "Detect and decompose articulated objects"))
        else:
            print(
                "WARNING: articulation was requested but stage 8b is not available in this "
                "release; continuing without it.",
                file=sys.stderr,
            )
    plan.extend(
        [
            StageSpec("9", f"{base}/9_compile_scene.py", "s9_compile", "simfoundry", "Compile scene"),
            StageSpec("10", f"{base}/10_make_objects_sim_ready.py", "s10_sim", "simfoundry", "Make sim-ready objects"),
            StageSpec("11", f"{base}/11_stabilize_physics.py", "s11_physics", "simfoundry", "Stabilize physics"),
            StageSpec("12", f"{base}/12_import_usd.py", "s12_usd", "simfoundry", "Import USD assets"),
            StageSpec("13", f"{base}/13_create_og_scene.py", "s13_og", "simfoundry", "Create OmniGibson scene"),
        ]
    )
    return plan


def get_augmentation_stage_plan(*, include_p2p: bool = False) -> list[StageSpec]:
    """Return ordered stage plan for reconstructed scene augmentation."""
    base = "scripts/pipeline/B_augmentation/stages"
    plan = [
        StageSpec("1", f"{base}/1_prompt_object_cousins.py", "prompt_cousin_structured", "simfoundry", "Prompt object cousin variations"),
        StageSpec("2", f"{base}/2_generate_cousin_combinations.py", "generate_cousins_combination", "simfoundry", "Generate cousin combinations"),
        StageSpec("3", f"{base}/3_generate_cousin_meshes.py", "cousin_generation", "mesh", "Generate cousin meshes"),
        StageSpec("4", f"{base}/4_make_cousins_sim_ready.py", "sim", "simfoundry", "Make cousin assets sim-ready"),
        StageSpec("5", f"{base}/5_import_cousin_usd.py", "usd", "b1k", "Import cousin USD assets"),
        StageSpec("6", f"{base}/6_sample_reconstructed_scene.py", "s13_og", "simfoundry", "Sample reconstructed scene variations"),
        StageSpec("7", f"{base}/7_propose_scene_tasks.py", "propose_scene_task", "simfoundry", "Propose scene tasks"),
    ]
    if include_p2p:
        plan.append(StageSpec("8", f"{base}/8_match_cousin_p2p.py", "cousin_p2p_match", "simfoundry", "Match cousin point correspondences"))
    return plan


def get_application_stage_plan() -> list[StageSpec]:
    """Return ordered stage plan for evaluation, teleop, demo generation, and replay."""
    base = "scripts/pipeline/C_application/stages"
    return [
        StageSpec("smoke", f"{base}/0_smoke_random_actions.py", "application_smoke", "b1k", "Random-action application smoke test"),
        StageSpec("1", f"{base}/1_eval_policy_og_scene.py", "s15_eval", "b1k", "Evaluate policy in OG scene"),
        StageSpec("2", f"{base}/2_teleop_og_scene.py", "s14_teleop", "b1k", "Collect teleop demonstration"),
        StageSpec("3", f"{base}/3_annotate_src_demo.py", "s15_annotate", "b1k", "Annotate source demonstration"),
        StageSpec("3b", f"{base}/3b_modify_annotations.py", "s15_annotate", "b1k", "Modify annotations"),
        StageSpec("4", f"{base}/4_extract_waypoints.py", "s16_waypoints", "b1k", "Extract waypoints"),
        StageSpec("5", f"{base}/5_generate_demos.py", "s17_generate", "b1k", "Generate demonstrations"),
        StageSpec("6", f"{base}/6_replay_dataset.py", "s18_replay", "b1k", "Replay generated dataset"),
    ]


def get_stage_plan(
    input_mode: str,
    *,
    pipeline_name: str = "reconstruction",
    include_p2p: bool = False,
    detect_articulation: bool = False,
    bg_splat: bool = False,
) -> list[StageSpec]:
    """Return ordered stage plan for a named sub-pipeline."""
    if pipeline_name in {"reconstruction", "A_reconstruction", "A"}:
        return get_reconstruction_stage_plan(input_mode, detect_articulation=detect_articulation, bg_splat=bg_splat)
    if pipeline_name in {"augmentation", "B_augmentation", "B"}:
        return get_augmentation_stage_plan(include_p2p=include_p2p)
    if pipeline_name in {"application", "C_application", "C"}:
        return get_application_stage_plan()
    raise ValueError(f"Unsupported pipeline_name={pipeline_name}")


def _parse_csv_set(csv: str | None) -> set[str]:
    if not csv:
        return set()
    return {item.strip() for item in csv.split(",") if item.strip()}


def _filter_stream_stage_overrides(extra_overrides: list[str]) -> list[str]:
    """Return overrides that should be forwarded from the streaming controller to child stages."""
    return [item for item in extra_overrides if not item.startswith("stream_subseq.")]


def select_stages(
    plan: Iterable[StageSpec],
    *,
    include_ids: set[str],
    exclude_ids: set[str],
) -> list[StageSpec]:
    out = []
    for spec in plan:
        if include_ids and spec.stage_id not in include_ids:
            continue
        if spec.stage_id in exclude_ids:
            continue
        out.append(spec)
    return out


def build_cmd(spec: StageSpec, *, env_map: dict[str, str], exec_mode: str, python_bin: str, extra_overrides: list[str]) -> list[str]:
    base = [python_bin, spec.script, *extra_overrides]
    if exec_mode == "direct":
        return base
    if exec_mode == "mamba":
        env_name = env_map[spec.env]
        return ["mamba", "run", "-n", env_name, *base]
    raise ValueError(f"Unsupported exec_mode={exec_mode}")


def run_stage(
    spec: StageSpec,
    *,
    env_map: dict[str, str],
    exec_mode: str,
    python_bin: str,
    cwd: str,
    dry_run: bool,
    extra_overrides: list[str],
    timing_log_path: Path | None,
    memory_samples: dict[str, dict[str, float | None]] | None = None,
) -> float:
    cmd = build_cmd(spec, env_map=env_map, exec_mode=exec_mode, python_bin=python_bin, extra_overrides=extra_overrides)
    printable = " ".join(shlex.quote(p) for p in cmd)
    print(f"[Stage {spec.stage_id}] {spec.description}")
    print(f"[Stage {spec.stage_id}] cmd: {printable}")
    if dry_run:
        return 0.0

    before_gb = query_gpu_memory_used_gb()
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["OMNIGIBSON_DATA_PATH"] = configure_omnigibson_data_path(force=True)
    if timing_log_path is not None:
        env[TIMING_LOG_ENV_VAR] = str(timing_log_path)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)
    elapsed = time.perf_counter() - t0
    after_gb = query_gpu_memory_used_gb()
    if memory_samples is not None:
        memory_samples[spec.stage_id] = {"before_used_gb": before_gb, "after_used_gb": after_gb}
    print(f"[Stage {spec.stage_id}] completed in {elapsed:.2f}s")
    if timing_log_path is not None:
        append_timing_log(timing_log_path, stage_id=spec.stage_id, description=spec.description, elapsed_s=elapsed)
    return elapsed


def run_stage_subsequence_streaming(
    *,
    stream_start_stage: int,
    stream_end_stage: int,
    env_map: dict[str, str],
    exec_mode: str,
    python_bin: str,
    cwd: str,
    dry_run: bool,
    extra_overrides: list[str],
    timing_log_path: Path | None,
    memory_samples: dict[str, dict[str, float | None]] | None = None,
) -> float:
    """Run a streaming subsequence for any contiguous range in stages 5-8."""
    if not (5 <= stream_start_stage <= stream_end_stage <= 8):
        raise ValueError("stream_start_stage and stream_end_stage must satisfy 5 <= start <= end <= 8")

    forwarded_overrides = _filter_stream_stage_overrides(extra_overrides)

    # Build per-stage commands for the streaming controller.
    def _stage_cmd(stage: int) -> str:
        base = "scripts/pipeline/A_reconstruction/stages"
        script = f"{base}/{stage}_placeholder.py"
        if stage == 5:
            script = f"{base}/5_decompose_scene.py"
            env_key = "simfoundry"
        elif stage == 6:
            script = f"{base}/6_upsample_object_images.py"
            env_key = "simfoundry"
        elif stage == 7:
            script = f"{base}/7_generate_object_meshes.py"
            env_key = "mesh"
        elif stage == 8:
            script = f"{base}/8_match_object_poses.py"
            env_key = "simfoundry"
        else:
            raise ValueError(stage)

        base = [python_bin, script, *forwarded_overrides]
        if exec_mode == "mamba":
            return shlex.join(["mamba", "run", "-n", env_map[env_key], *base])
        return shlex.join(base)

    stream_overrides = [
        *extra_overrides,
        f"stream_subseq.start_stage={stream_start_stage}",
        f"stream_subseq.end_stage={stream_end_stage}",
        f"stream_subseq.s5_cmd={json.dumps(_stage_cmd(5))}",
        f"stream_subseq.s6_cmd={json.dumps(_stage_cmd(6))}",
        f"stream_subseq.s7_cmd={json.dumps(_stage_cmd(7))}",
        f"stream_subseq.s8_cmd={json.dumps(_stage_cmd(8))}",
    ]
    stream_spec = StageSpec(
        f"{stream_start_stage}{stream_end_stage}",
        "scripts/pipeline/A_reconstruction/stages/58_stream_subsequence.py",
        "stream_subseq",
        "simfoundry",
        f"Stream stages {stream_start_stage}->{stream_end_stage}",
    )
    cmd_stream = build_cmd(
        stream_spec,
        env_map=env_map,
        exec_mode=exec_mode,
        python_bin=python_bin,
        extra_overrides=stream_overrides,
    )

    if dry_run:
        print(f"[Stage {stream_start_stage}{stream_end_stage}] dry-run: would run streaming subsequence")
        print(f"[Stage {stream_start_stage}{stream_end_stage}] stream:", " ".join(cmd_stream))
        return 0.0

    before_gb = query_gpu_memory_used_gb()
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["OMNIGIBSON_DATA_PATH"] = configure_omnigibson_data_path(force=True)
    if timing_log_path is not None:
        env[TIMING_LOG_ENV_VAR] = str(timing_log_path)
    subprocess.run(cmd_stream, cwd=cwd, check=True, env=env)
    elapsed = time.perf_counter() - t0
    after_gb = query_gpu_memory_used_gb()
    if memory_samples is not None:
        memory_samples[f"{stream_start_stage}{stream_end_stage}"] = {"before_used_gb": before_gb, "after_used_gb": after_gb}
    return elapsed


def run_pipeline(
    *,
    cwd: str,
    pipeline_name: str = "reconstruction",
    input_mode: str,
    include_ids_csv: str | None,
    exclude_ids_csv: str | None,
    exec_mode: str,
    python_bin: str,
    env_map: dict[str, str],
    dry_run: bool,
    stream_subseq_enabled: bool,
    stream_start_stage: int,
    stream_end_stage: int,
    extra_overrides: list[str],
    include_p2p: bool = False,
    detect_articulation: bool = False,
    bg_splat: bool = False,
) -> PipelineTimingResult:
    wall_start = time.perf_counter()
    timing_log_path = None if dry_run else resolve_scene_timing_log_path(extra_overrides)
    scene_dir = None if dry_run else resolve_scene_dir(extra_overrides)
    plan = get_stage_plan(
        input_mode=input_mode,
        pipeline_name=pipeline_name,
        include_p2p=include_p2p,
        detect_articulation=detect_articulation,
        bg_splat=bg_splat,
    )
    selected = select_stages(plan, include_ids=_parse_csv_set(include_ids_csv), exclude_ids=_parse_csv_set(exclude_ids_csv))

    durations: dict[str, float] = {}
    memory_samples: dict[str, dict[str, float | None]] = {}
    i = 0
    stream_window = [str(v) for v in range(stream_start_stage, stream_end_stage + 1)]
    while i < len(selected):
        spec = selected[i]
        window_ok = (
            pipeline_name in {"reconstruction", "A_reconstruction", "A"}
            and
            stream_subseq_enabled
            and spec.stage_id == stream_window[0]
            and i + len(stream_window) <= len(selected)
            and [selected[i + j].stage_id for j in range(len(stream_window))] == stream_window
        )
        if window_ok:
            elapsed = run_stage_subsequence_streaming(
                stream_start_stage=stream_start_stage,
                stream_end_stage=stream_end_stage,
                env_map=env_map,
                exec_mode=exec_mode,
                python_bin=python_bin,
                cwd=cwd,
                dry_run=dry_run,
                extra_overrides=extra_overrides,
                timing_log_path=timing_log_path,
                memory_samples=memory_samples,
            )
            key = f"{stream_start_stage}{stream_end_stage}"
            durations[key] = elapsed
            if timing_log_path is not None:
                append_timing_log(
                    timing_log_path,
                    stage_id=key,
                    description=f"Streaming stages {stream_start_stage}->{stream_end_stage}",
                    elapsed_s=elapsed,
                )
            i += len(stream_window)
            continue

        durations[spec.stage_id] = run_stage(
            spec,
            env_map=env_map,
            exec_mode=exec_mode,
            python_bin=python_bin,
            cwd=cwd,
            dry_run=dry_run,
            extra_overrides=extra_overrides,
            timing_log_path=timing_log_path,
            memory_samples=memory_samples,
        )
        i += 1

    wall_time = time.perf_counter() - wall_start
    total = sum(durations.values())
    print("=== Pipeline Timing Summary ===")
    for sid, t in durations.items():
        print(f"{sid:>4}: {format_duration(t):>12}")
    print(f"{'SUM':>4}: {format_duration(total):>12}")
    print(f"{'WALL':>4}: {format_duration(wall_time):>12}")
    if timing_log_path is not None:
        append_timing_log(timing_log_path, stage_id="SUM", description="Sum of sequential stage durations", elapsed_s=total)
        append_timing_log(timing_log_path, stage_id="WALL", description="Pipeline wall time", elapsed_s=wall_time)

    report_path = None
    manifest_path = None
    if scene_dir is not None:
        written = write_pipeline_report(
            scene_dir=scene_dir,
            stage_durations_s=durations,
            wall_time_s=wall_time,
            memory_samples=memory_samples,
        )
        report_path = written["report_path"]
        manifest_path = written["manifest_path"]
        print(f"[Pipeline] manifest: {manifest_path}")
        print(f"[Pipeline] report: {report_path}")

    return PipelineTimingResult(
        stage_durations_s=durations,
        wall_time_s=wall_time,
        memory_samples=memory_samples,
        report_path=report_path,
        manifest_path=manifest_path,
    )
