#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Checkpoint Sweep Evaluator

Iterates over training checkpoints at a configurable interval, starts the
inference server (gr00t or openpi) for each checkpoint, runs OmniGibson sim
evaluation, collects results, and generates summary plots.

Supports two policy types:
  - gr00t: Uses ZMQ-based gr00t server (default port 5555)
  - openpi: Uses websocket-based OpenPI server (default port 8000)

Usage (gr00t):
    python sweep_eval.py \
        --policy-type gr00t \
        --checkpoint-dir /path/to/checkpoints \
        --checkpoint-interval 2000 \
        --eval-config put_away_trash \
        --embodiment-tag OXE_DROID_JOINT_POSITION_RELATIVE_RANDOM_VIEW \
        --output-dir ./sweep_results

Usage (openpi):
    python sweep_eval.py \
        --policy-type openpi \
        --checkpoint-dir /path/to/openpi/checkpoints/experiment_name \
        --checkpoint-interval 1000 \
        --eval-config put_away_trash \
        --openpi-policy-config pi05_droid_jointpos \
        --output-dir ./sweep_results

    # Dry run to see which checkpoints would be evaluated:
    python sweep_eval.py \
        --checkpoint-dir /path/to/checkpoints \
        --checkpoint-interval 2000 \
        --eval-config put_away_trash \
        --embodiment-tag OXE_DROID_JOINT_POSITION_RELATIVE_RANDOM_VIEW \
        --dry-run

    # Use a YAML config file (CLI args override YAML values):
    python sweep_eval.py --config sweep_config.yaml

    # YAML config file format (sweep_config.yaml):
    #   policy_type: gr00t              # or "openpi"
    #   checkpoint_dir: /path/to/checkpoints
    #   checkpoint_interval: 2000
    #   checkpoint_range: "2000:30000"  # optional
    #   eval_config: put_away_trash
    #   eval_overrides:                 # optional
    #     - "s15_eval.n_episodes=10"
    #   # --- gr00t-specific ---
    #   embodiment_tag: OXE_DROID_JOINT_POSITION_RELATIVE_RANDOM_VIEW
    #   num_inference_timesteps: 8
    #   gr00t_project_dir: ~/Projects/gr00t
    #   gr00t_venv_cmd: "uv run"
    #   # --- openpi-specific ---
    #   openpi_policy_config: pi05_droid_jointpos
    #   openpi_project_dir: ~/Projects/openpi
    #   openpi_venv_cmd: "uv run"
    #   xla_mem_fraction: 0.5
    #   # --- common ---
    #   server_port: 5555               # 5555 for gr00t, 8000 for openpi
    #   server_host: localhost
    #   server_startup_timeout: 180
    #   output_dir: ./sweep_results
    #   simfoundry_project_dir: ~/Projects/simfoundry
    #   simfoundry_venv_cmd: ""
    #   dry_run: false
    #   skip_plots: false
    #   resume: false
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None  # YAML config loading will raise a clear error if used without pyyaml


POLICY_TYPE_GROOT = "gr00t"
POLICY_TYPE_OPENPI = "openpi"


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(
    checkpoint_dir: str,
    interval: int,
    policy_type: str,
    checkpoint_range: Optional[str] = None,
) -> list[tuple[int, Path]]:
    """Find checkpoint directories matching the requested interval and range.

    For gr00t: expects checkpoint-NNNNN directories.
    For openpi: expects bare number directories (e.g. 1000, 2000, ...).

    Returns a sorted list of (step, path) tuples.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Parse optional range
    range_min, range_max = 0, float("inf")
    if checkpoint_range:
        parts = checkpoint_range.split(":")
        if len(parts) != 2:
            raise ValueError("--checkpoint-range must be in the form MIN:MAX (e.g. 2000:30000)")
        range_min = int(parts[0]) if parts[0] else 0
        range_max = int(parts[1]) if parts[1] else float("inf")

    if policy_type == POLICY_TYPE_GROOT:
        pattern = re.compile(r"^checkpoint-(\d+)$")
    else:  # openpi
        pattern = re.compile(r"^(\d+)$")

    matches: list[tuple[int, Path]] = []

    for entry in sorted(checkpoint_dir.iterdir()):
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m:
            step = int(m.group(1))
            if step % interval == 0 and range_min <= step <= range_max:
                matches.append((step, entry))

    matches.sort(key=lambda x: x[0])
    return matches


def checkpoint_display_name(step: int, prefix: Optional[str] = None) -> str:
    """Return a human-readable name for a checkpoint step.

    If prefix is provided, the name will be "{prefix}/checkpoint-{step}"
    so that results are saved under a subdirectory:
        Data/{scene}/s15_eval/gr00t/{prefix}/checkpoint-{step}/results_{timestamp}/
    """

    base = f"checkpoint-{step}"

    if prefix:
        return f"{prefix}/{base}"
    return base
   

# ---------------------------------------------------------------------------
# Server lifecycle helpers - gr00t
# ---------------------------------------------------------------------------

def _build_groot_server_cmd(
    gr00t_project_dir: str,
    checkpoint_path: str,
    embodiment_tag: str,
    num_inference_timesteps: int,
    server_port: int,
    gr00t_venv_cmd: str,
) -> tuple[str, str]:
    """Return (command_string, working_directory) for starting the gr00t server."""
    vla_dir = os.path.join(gr00t_project_dir, "groot", "vla")
    cmd = (
        f"{gr00t_venv_cmd} omni/oss_eval/run_gr00t_server.py"
        f" --use_sim_policy_wrapper"
        f" --embodiment-tag {embodiment_tag}"
        f" --model-path {checkpoint_path}"
        f" --num-inference-timesteps {num_inference_timesteps}"
        f" --port {server_port}"
    )
    return cmd, vla_dir


def start_groot_server(
    gr00t_project_dir: str,
    checkpoint_path: str,
    embodiment_tag: str,
    num_inference_timesteps: int,
    server_port: int,
    gr00t_venv_cmd: str,
    server_log_path: Optional[str] = None,
) -> subprocess.Popen:
    """Start the gr00t inference server as a background subprocess."""
    cmd, cwd = _build_groot_server_cmd(
        gr00t_project_dir,
        checkpoint_path,
        embodiment_tag,
        num_inference_timesteps,
        server_port,
        gr00t_venv_cmd,
    )
    print(f"  [server] Starting gr00t: {cmd}")
    print(f"  [server] CWD: {cwd}")

    log_file = None
    if server_log_path:
        os.makedirs(os.path.dirname(server_log_path), exist_ok=True)
        log_file = open(server_log_path, "w")

    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=log_file or subprocess.DEVNULL,
        stderr=log_file or subprocess.STDOUT if log_file else subprocess.DEVNULL,
        preexec_fn=os.setsid,  # Create new process group for clean kill
    )
    return proc


def wait_for_groot_server(host: str, port: int, timeout: float = 180.0, poll_interval: float = 3.0) -> bool:
    """Poll the gr00t server with ZMQ ping until it responds or timeout is reached."""
    import zmq

    print(f"  [server] Waiting for gr00t server at {host}:{port} (timeout={timeout}s)...")
    deadline = time.time() + timeout
    ctx = zmq.Context()

    while time.time() < deadline:
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 5000)  # 5s receive timeout
        sock.setsockopt(zmq.SNDTIMEO, 5000)  # 5s send timeout
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(f"tcp://{host}:{port}")
            # Send a ping using the same protocol as PolicyClient
            import msgpack
            request = {"endpoint": "ping"}
            sock.send(msgpack.packb(request))
            sock.recv()
            sock.close()
            ctx.term()
            print(f"  [server] Server is ready!")
            return True
        except (zmq.error.Again, zmq.error.ZMQError, Exception):
            sock.close()
            time.sleep(poll_interval)

    ctx.term()
    print(f"  [server] Timeout waiting for server!")
    return False


def kill_groot_server(host: str, port: int, proc: subprocess.Popen, timeout: float = 10.0):
    """Kill the gr00t server gracefully via the kill endpoint, falling back to SIGTERM."""
    import zmq
    import msgpack

    print(f"  [server] Sending kill command to gr00t server...")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.setsockopt(zmq.SNDTIMEO, 5000)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(f"tcp://{host}:{port}")
        request = {"endpoint": "kill"}
        sock.send(msgpack.packb(request))
        sock.recv()
    except Exception:
        pass
    finally:
        sock.close()
        ctx.term()

    # Wait for process to exit
    _wait_and_kill_proc(proc, timeout)


# ---------------------------------------------------------------------------
# Server lifecycle helpers - openpi
# ---------------------------------------------------------------------------

def _build_openpi_server_cmd(
    openpi_project_dir: str,
    checkpoint_path: str,
    openpi_policy_config: str,
    server_port: int,
    openpi_venv_cmd: str,
    xla_mem_fraction: float,
) -> tuple[str, str, dict]:
    """Return (command_string, working_directory, extra_env) for starting the openpi server."""
    cmd = (
        f"{openpi_venv_cmd} scripts/serve_policy.py"
        f" --port {server_port}"
        f" policy:checkpoint"
        f" --policy.config={openpi_policy_config}"
        f" --policy.dir={checkpoint_path}"
    )
    extra_env = {
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(xla_mem_fraction),
    }
    return cmd, openpi_project_dir, extra_env


def start_openpi_server(
    openpi_project_dir: str,
    checkpoint_path: str,
    openpi_policy_config: str,
    server_port: int,
    openpi_venv_cmd: str,
    xla_mem_fraction: float,
    server_log_path: Optional[str] = None,
) -> subprocess.Popen:
    """Start the OpenPI inference server as a background subprocess."""
    cmd, cwd, extra_env = _build_openpi_server_cmd(
        openpi_project_dir,
        checkpoint_path,
        openpi_policy_config,
        server_port,
        openpi_venv_cmd,
        xla_mem_fraction,
    )
    print(f"  [server] Starting openpi: {cmd}")
    print(f"  [server] CWD: {cwd}")
    print(f"  [server] XLA_PYTHON_CLIENT_MEM_FRACTION={xla_mem_fraction}")

    log_file = None
    if server_log_path:
        os.makedirs(os.path.dirname(server_log_path), exist_ok=True)
        log_file = open(server_log_path, "w")

    env = os.environ.copy()
    env.update(extra_env)

    proc = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=log_file or subprocess.DEVNULL,
        stderr=log_file or subprocess.STDOUT if log_file else subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_openpi_server(host: str, port: int, timeout: float = 180.0, poll_interval: float = 5.0) -> bool:
    """Poll the OpenPI server via HTTP /healthz until it responds or timeout is reached."""
    url = f"http://{host}:{port}/healthz"
    print(f"  [server] Waiting for openpi server at {url} (timeout={timeout}s)...")
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                print(f"  [server] Server is ready!")
                return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(poll_interval)

    print(f"  [server] Timeout waiting for server!")
    return False


def kill_openpi_server(proc: subprocess.Popen, timeout: float = 10.0):
    """Kill the OpenPI server via SIGTERM (no graceful kill endpoint)."""
    print(f"  [server] Sending SIGTERM to openpi server...")
    _wait_and_kill_proc(proc, timeout)


# ---------------------------------------------------------------------------
# Common process kill helper
# ---------------------------------------------------------------------------

def _wait_and_kill_proc(proc: subprocess.Popen, timeout: float = 10.0):
    """Try SIGTERM, then SIGKILL if process doesn't exit."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=timeout)
        print(f"  [server] Server exited (code={proc.returncode})")
    except subprocess.TimeoutExpired:
        print(f"  [server] Server did not exit after SIGTERM, sending SIGKILL...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass
    except ProcessLookupError:
        print(f"  [server] Process already exited")


# ---------------------------------------------------------------------------
# Unified server interface
# ---------------------------------------------------------------------------

def start_server(args, checkpoint_path: str, server_log_path: Optional[str] = None) -> subprocess.Popen:
    """Start the appropriate inference server based on policy type."""
    if args.policy_type == POLICY_TYPE_GROOT:
        return start_groot_server(
            gr00t_project_dir=args.gr00t_project_dir,
            checkpoint_path=checkpoint_path,
            embodiment_tag=args.embodiment_tag,
            num_inference_timesteps=args.num_inference_timesteps,
            server_port=args.server_port,
            gr00t_venv_cmd=args.gr00t_venv_cmd,
            server_log_path=server_log_path,
        )
    else:
        return start_openpi_server(
            openpi_project_dir=args.openpi_project_dir,
            checkpoint_path=checkpoint_path,
            openpi_policy_config=args.openpi_policy_config,
            server_port=args.server_port,
            openpi_venv_cmd=args.openpi_venv_cmd,
            xla_mem_fraction=args.xla_mem_fraction,
            server_log_path=server_log_path,
        )


def wait_for_server(args) -> bool:
    """Wait for the appropriate server to become ready."""
    if args.policy_type == POLICY_TYPE_GROOT:
        return wait_for_groot_server(
            host=args.server_host,
            port=args.server_port,
            timeout=args.server_startup_timeout,
        )
    else:
        return wait_for_openpi_server(
            host=args.server_host,
            port=args.server_port,
            timeout=args.server_startup_timeout,
        )


def kill_server(args, proc: subprocess.Popen):
    """Kill the appropriate server."""
    if args.policy_type == POLICY_TYPE_GROOT:
        kill_groot_server(args.server_host, args.server_port, proc)
    else:
        kill_openpi_server(proc)


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(
    simfoundry_project_dir: str,
    eval_config: str,
    checkpoint_name: str,
    server_port: int,
    policy_type: str,
    simfoundry_venv_cmd: str,
    eval_overrides: Optional[list[str]] = None,
) -> int:
    """Run the application eval stage as a blocking subprocess. Returns exit code."""
    pipeline_dir = simfoundry_project_dir

    overrides = [
        f"s15_eval.checkpoint={checkpoint_name}",
        f"s15_eval.policy={policy_type}",
        f"s15_eval.port={server_port}",
    ]
    if eval_overrides:
        overrides.extend(eval_overrides)

    override_str = " ".join(overrides)
    cmd = (
        f"{simfoundry_venv_cmd} python "
        f"scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py "
        f"--config-name={eval_config} {override_str}"
    )

    print(f"  [eval] Running: {cmd}")
    print(f"  [eval] CWD: {pipeline_dir}")

    result = subprocess.run(cmd, shell=True, cwd=pipeline_dir)
    return result.returncode


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def find_eval_results(simfoundry_project_dir: str, eval_config: str, checkpoint_name: str) -> Optional[dict]:
    """Search for the eval_results.json produced by the eval run.

    The eval script writes results into a directory tree like:
        {root_dir}/{scene_name}/s15_eval/{policy}/{checkpoint_name}/results_{timestamp}/eval_results.json

    We look for the most recently modified eval_results.json matching the checkpoint name.
    """
    # Search broadly under the simfoundry data directory for the result file
    data_dir = Path(simfoundry_project_dir) / "Data"
    if not data_dir.exists():
        data_dir = Path(simfoundry_project_dir) / "scripts" / "Data"

    # Recursively find all eval_results.json files
    candidates = []
    search_roots = [data_dir]

    # Also check the scripts/pipeline relative Data dir
    pipeline_data = Path(simfoundry_project_dir) / "scripts" / "pipeline" / "Data"
    if pipeline_data.exists():
        search_roots.append(pipeline_data)

    # Also try ../../Data relative to scripts/pipeline (common Hydra pattern)
    rel_data = Path(simfoundry_project_dir) / "scripts" / "Data"
    if rel_data.exists():
        search_roots.append(rel_data)

    for root in search_roots:
        if not root.exists():
            continue
        for results_file in root.rglob("eval_results.json"):
            # Check if this result belongs to our checkpoint
            results_str = str(results_file)
            if checkpoint_name in results_str:
                candidates.append(results_file)

    if not candidates:
        print(f"  [results] WARNING: Could not find eval_results.json for {checkpoint_name}")
        return None

    # Pick the most recently modified one
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    best = candidates[0]
    print(f"  [results] Found: {best}")

    with open(best, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Aggregation & plotting
# ---------------------------------------------------------------------------

def aggregate_results(all_results: dict[int, dict]) -> dict:
    """Create a summary structure keyed by step."""
    summary = {
        "checkpoints": {},
        "steps": [],
        "success_rates": [],
        "avg_rewards": [],
        "avg_milestone_progress": [],
    }

    for step in sorted(all_results.keys()):
        result = all_results[step]
        summary["steps"].append(step)
        summary["success_rates"].append(result.get("success_rate", 0.0))
        summary["avg_rewards"].append(result.get("avg_episode_reward", 0.0))
        summary["avg_milestone_progress"].append(result.get("avg_milestone_progress", 0.0))
        summary["checkpoints"][step] = result

    return summary


def generate_plots(summary: dict, output_dir: str):
    """Generate and save summary plots using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    steps = summary["steps"]
    if not steps:
        print("  [plots] No data to plot.")
        return

    # 1. Success rate vs training step
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, summary["success_rates"], "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Success Rate vs. Training Step", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "success_rate.png"), dpi=150)
    plt.close(fig)
    print(f"  [plots] Saved success_rate.png")

    # 2. Average episode reward vs training step
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, summary["avg_rewards"], "o-", linewidth=2, markersize=6, color="tab:orange")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Average Episode Reward", fontsize=12)
    ax.set_title("Average Episode Reward vs. Training Step", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "avg_reward.png"), dpi=150)
    plt.close(fig)
    print(f"  [plots] Saved avg_reward.png")

    # 3. Average milestone progress vs training step
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, summary["avg_milestone_progress"], "o-", linewidth=2, markersize=6, color="tab:green")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Average Milestone Progress", fontsize=12)
    ax.set_title("Average Milestone Progress vs. Training Step", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "milestone_progress.png"), dpi=150)
    plt.close(fig)
    print(f"  [plots] Saved milestone_progress.png")

    # 4. Per-milestone success rates (if available)
    milestone_names = None
    for step in steps:
        ckpt_data = summary["checkpoints"].get(step, {})
        if "milestone_success_rates" in ckpt_data and ckpt_data["milestone_success_rates"]:
            milestone_names = list(ckpt_data["milestone_success_rates"].keys())
            break

    if milestone_names:
        fig, ax = plt.subplots(figsize=(12, 7))
        for m_name in milestone_names:
            m_rates = []
            for step in steps:
                ckpt_data = summary["checkpoints"].get(step, {})
                msr = ckpt_data.get("milestone_success_rates", {})
                m_rates.append(msr.get(m_name, 0.0))
            ax.plot(steps, m_rates, "o-", linewidth=2, markersize=5, label=m_name)

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_title("Per-Milestone Success Rate vs. Training Step", fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "per_milestone_success.png"), dpi=150)
        plt.close(fig)
        print(f"  [plots] Saved per_milestone_success.png")

    # 5. Combined overview plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(steps, summary["success_rates"], "o-", linewidth=2, markersize=5)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, summary["avg_rewards"], "o-", linewidth=2, markersize=5, color="tab:orange")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Avg Episode Reward")
    axes[1].set_title("Average Reward")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, summary["avg_milestone_progress"], "o-", linewidth=2, markersize=5, color="tab:green")
    axes[2].set_xlabel("Training Step")
    axes[2].set_ylabel("Avg Milestone Progress")
    axes[2].set_title("Milestone Progress")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Checkpoint Sweep Evaluation Summary", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "combined_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] Saved combined_overview.png")


def print_summary_table(summary: dict):
    """Print a formatted summary table to stdout."""
    steps = summary["steps"]
    if not steps:
        print("No results to display.")
        return

    # Header
    header = f"{'Step':>10} | {'Success Rate':>13} | {'Avg Reward':>11} | {'Milestone %':>12}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for i, step in enumerate(steps):
        sr = summary["success_rates"][i]
        ar = summary["avg_rewards"][i]
        mp = summary["avg_milestone_progress"][i]
        print(f"{step:>10} | {sr:>13.1%} | {ar:>11.3f} | {mp:>12.1%}")

    print(sep)

    # Best checkpoint
    best_idx = max(range(len(steps)), key=lambda i: summary["success_rates"][i])
    print(f"\nBest checkpoint: step {steps[best_idx]} (success rate: {summary['success_rates'][best_idx]:.1%})")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def _load_yaml_config(yaml_path: str) -> dict:
    """Load a YAML config file and return as a flat dict matching CLI arg names."""
    if yaml is None:
        raise ImportError(
            "PyYAML is required for --config support. Install it with: pip install pyyaml"
        )
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    # Expand ~ in path values
    for key in cfg:
        if isinstance(cfg[key], str) and "~" in cfg[key]:
            cfg[key] = os.path.expanduser(cfg[key])
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Checkpoint Sweep Evaluator - evaluate multiple training checkpoints in sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # YAML config (loaded first, CLI args override)
    parser.add_argument("--config", default=None,
                        help="Path to a YAML config file. CLI args override YAML values.")

    # Policy type
    parser.add_argument("--policy-type", default=None, choices=["gr00t", "openpi"],
                        help="Policy type: 'gr00t' or 'openpi' (default: gr00t)")

    # Checkpoint selection
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory containing checkpoint folders")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="Evaluate every N-th checkpoint step (e.g. 2000)")
    parser.add_argument("--checkpoint-range", default=None,
                        help="Optional MIN:MAX range for checkpoint steps (e.g. 2000:30000)")
    parser.add_argument("--checkpoint-name-prefix", default=None,
                        help="Optional prefix for checkpoint names (e.g. 'put_away_marker_0214' -> 'put_away_marker_0214-checkpoint-5000')")

    # Evaluation config
    parser.add_argument("--eval-config", default=None,
                        help="Hydra config name for 15_eval_policy_og_scene.py (e.g. put_away_trash)")
    parser.add_argument("--eval-overrides", nargs="*", default=None,
                        help="Extra Hydra overrides forwarded to the eval script")

    # gr00t server config
    parser.add_argument("--embodiment-tag", default=None,
                        help="[gr00t] Embodiment tag (e.g. OXE_DROID_JOINT_POSITION_RELATIVE_RANDOM_VIEW)")
    parser.add_argument("--num-inference-timesteps", type=int, default=None,
                        help="[gr00t] Number of inference timesteps (default: 8)")

    # openpi server config
    parser.add_argument("--openpi-policy-config", default=None,
                        help="[openpi] Policy config name (e.g. pi05_droid_jointpos)")
    parser.add_argument("--xla-mem-fraction", type=float, default=None,
                        help="[openpi] XLA_PYTHON_CLIENT_MEM_FRACTION (default: 0.5)")

    # Common server config
    parser.add_argument("--server-port", type=int, default=None,
                        help="Port for the inference server (default: 5555 for gr00t, 8000 for openpi)")
    parser.add_argument("--server-host", default=None,
                        help="Host for the inference server (default: localhost)")
    parser.add_argument("--server-startup-timeout", type=float, default=None,
                        help="Timeout in seconds waiting for server startup (default: 180)")

    # Output
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save aggregated results and plots (default: ./sweep_results)")

    # Project directories
    parser.add_argument("--gr00t-project-dir", default=None,
                        help="[gr00t] Path to the gr00t repository")
    parser.add_argument("--simfoundry-project-dir", default=None,
                        help="Path to the SimFoundry repository")
    parser.add_argument("--openpi-project-dir", default=None,
                        help="[openpi] Path to the openpi repository")

    # Environment commands
    parser.add_argument("--gr00t-venv-cmd", default=None,
                        help="[gr00t] Command prefix for gr00t env (default: 'uv run')")
    parser.add_argument("--simfoundry-venv-cmd", default=None,
                        help="Command prefix for SimFoundry env (default: '' i.e. current env)")
    parser.add_argument("--openpi-venv-cmd", default=None,
                        help="[openpi] Command prefix for openpi env (default: 'uv run')")

    # Misc
    parser.add_argument("--dry-run", action="store_true",
                        help="Show which checkpoints would be evaluated without running anything")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--resume", action="store_true",
                        help="Skip checkpoints that already have results in the output directory")

    args = parser.parse_args()

    # Merge YAML config with CLI args (CLI takes precedence)
    # Defaults for each field
    defaults = {
        "policy_type": "gr00t",
        "checkpoint_dir": None,
        "checkpoint_interval": None,
        "checkpoint_range": None,
        "checkpoint_name_prefix": None,
        "eval_config": None,
        "eval_overrides": None,
        # gr00t
        "embodiment_tag": None,
        "num_inference_timesteps": 8,
        "gr00t_project_dir": os.path.expanduser("~/Projects/gr00t"),
        "gr00t_venv_cmd": "uv run",
        # openpi
        "openpi_policy_config": None,
        "openpi_project_dir": os.path.expanduser("~/Projects/openpi"),
        "openpi_venv_cmd": "uv run",
        "xla_mem_fraction": 0.5,
        # common
        "server_port": None,  # Will be set based on policy_type
        "server_host": "localhost",
        "server_startup_timeout": 180.0,
        "output_dir": "./sweep_results",
        "simfoundry_project_dir": os.path.expanduser("~/Projects/simfoundry"),
        "simfoundry_venv_cmd": "",
        "dry_run": False,
        "skip_plots": False,
        "resume": False,
    }

    # Load YAML if provided
    yaml_cfg = {}
    if args.config:
        yaml_cfg = _load_yaml_config(args.config)

    # Build final namespace: defaults < YAML < CLI
    final = argparse.Namespace()
    for key, default_val in defaults.items():
        cli_val = getattr(args, key, None)
        yaml_val = yaml_cfg.get(key, None)

        # For boolean flags from argparse, they default to False, so we need special handling
        if key in ("dry_run", "skip_plots", "resume"):
            if cli_val:
                setattr(final, key, True)
            elif yaml_val is not None:
                setattr(final, key, bool(yaml_val))
            else:
                setattr(final, key, default_val)
        elif cli_val is not None:
            setattr(final, key, cli_val)
        elif yaml_val is not None:
            setattr(final, key, yaml_val)
        else:
            setattr(final, key, default_val)

    # Set default server port based on policy type if not explicitly provided
    if final.server_port is None:
        final.server_port = 5555 if final.policy_type == POLICY_TYPE_GROOT else 8000

    # Validate required fields
    required_common = ["checkpoint_dir", "checkpoint_interval", "eval_config"]
    for field in required_common:
        if getattr(final, field) is None:
            parser.error(
                f"--{field.replace('_', '-')} is required (provide via CLI or YAML config)"
            )

    # Policy-type-specific validation
    if final.policy_type == POLICY_TYPE_GROOT:
        if final.embodiment_tag is None:
            parser.error("--embodiment-tag is required for gr00t policy type")
    elif final.policy_type == POLICY_TYPE_OPENPI:
        if final.openpi_policy_config is None:
            parser.error("--openpi-policy-config is required for openpi policy type")

    return final


def load_existing_results(output_dir: str) -> dict[int, dict]:
    """Load previously saved results from the sweep summary file."""
    summary_path = os.path.join(output_dir, "sweep_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
        # Convert string keys back to int
        return {int(k): v for k, v in data.get("checkpoints", {}).items()}
    return {}


def main():
    args = parse_args()

    # Discover checkpoints
    print("=" * 60)
    print("Checkpoint Sweep Evaluator")
    print("=" * 60)
    print(f"  Policy type: {args.policy_type}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Interval: every {args.checkpoint_interval} steps")
    print(f"  Range: {args.checkpoint_range or 'all'}")
    print(f"  Eval config: {args.eval_config}")
    if args.policy_type == POLICY_TYPE_GROOT:
        print(f"  Embodiment tag: {args.embodiment_tag}")
        print(f"  Inference timesteps: {args.num_inference_timesteps}")
    else:
        print(f"  OpenPI policy config: {args.openpi_policy_config}")
        print(f"  XLA mem fraction: {args.xla_mem_fraction}")
    print(f"  Server port: {args.server_port}")
    print(f"  Output dir: {args.output_dir}")
    print()

    checkpoints = discover_checkpoints(
        args.checkpoint_dir,
        args.checkpoint_interval,
        args.policy_type,
        args.checkpoint_range,
    )

    if not checkpoints:
        print("ERROR: No checkpoints found matching the specified criteria.")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoints to evaluate:")
    for step, path in checkpoints:
        print(f"  {checkpoint_display_name(step, args.checkpoint_name_prefix)}: {path}")
    print()

    if args.dry_run:
        print("[DRY RUN] Exiting without running evaluations.")
        sys.exit(0)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load existing results if resuming
    all_results: dict[int, dict] = {}
    if args.resume:
        all_results = load_existing_results(args.output_dir)
        if all_results:
            print(f"Resuming: loaded results for {len(all_results)} checkpoints: {sorted(all_results.keys())}")
            print()

    # Main loop
    total = len(checkpoints)
    for idx, (step, ckpt_path) in enumerate(checkpoints):
        checkpoint_name = checkpoint_display_name(step, args.checkpoint_name_prefix)

        print("=" * 60)
        print(f"[{idx + 1}/{total}] Evaluating {checkpoint_name}")
        print("=" * 60)

        # Skip if already evaluated (resume mode)
        if args.resume and step in all_results:
            print(f"  Skipping (already evaluated in previous run)")
            print()
            continue

        server_proc = None
        try:
            # 1. Start server
            # Use a safe filename for logs (replace / with _)
            safe_name = checkpoint_name.replace("/", "_")
            server_log = os.path.join(args.output_dir, "logs", f"server_{safe_name}.log")
            server_proc = start_server(args, str(ckpt_path), server_log)

            # 2. Wait for server to be ready
            ready = wait_for_server(args)
            if not ready:
                print(f"  ERROR: Server failed to start for {checkpoint_name}. Skipping.")
                if server_proc.poll() is not None:
                    print(f"  Server process exited with code {server_proc.returncode}")
                    print(f"  Check server log: {server_log}")
                continue

            # 3. Run evaluation
            exit_code = run_eval(
                simfoundry_project_dir=args.simfoundry_project_dir,
                eval_config=args.eval_config,
                checkpoint_name=checkpoint_name,
                server_port=args.server_port,
                policy_type=args.policy_type,
                simfoundry_venv_cmd=args.simfoundry_venv_cmd,
                eval_overrides=args.eval_overrides,
            )

            if exit_code != 0:
                print(f"  WARNING: Eval script exited with code {exit_code}")

            # 4. Collect results
            result = find_eval_results(args.simfoundry_project_dir, args.eval_config, checkpoint_name)
            if result:
                all_results[step] = result
                print(f"  Success rate: {result.get('success_rate', 'N/A')}")
            else:
                print(f"  WARNING: No results found for {checkpoint_name}")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving partial results...")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 5. Kill server
            if server_proc is not None and server_proc.poll() is None:
                kill_server(args, server_proc)

            # Save intermediate results after each checkpoint (inside finally
            # so results are persisted even if the process crashes afterwards)
            if all_results:
                summary = aggregate_results(all_results)
                summary_path = os.path.join(args.output_dir, "sweep_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"  [results] Saved intermediate sweep_summary.json ({len(all_results)} checkpoints)")

        print()

    # Final aggregation and plotting
    if not all_results:
        print("No results collected. Exiting.")
        sys.exit(1)

    print("=" * 60)
    print("Aggregating results and generating plots")
    print("=" * 60)

    summary = aggregate_results(all_results)

    # Save final summary
    summary_path = os.path.join(args.output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved sweep_summary.json to {summary_path}")

    # Generate plots
    if not args.skip_plots:
        generate_plots(summary, args.output_dir)

    # Print summary table
    print_summary_table(summary)

    print(f"\nAll results saved to: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
