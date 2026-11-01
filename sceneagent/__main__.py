"""Run the phase-1 SceneAgent slice.

    python -m sceneagent examples/phase1_place_scene.json --out runs/phase1_place
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sceneagent.pipeline import run_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="SceneAgent phase-1 conversion")
    parser.add_argument("capture", type=Path, help="Capture JSON, SimFoundry scene JSON, or an example mp4")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the ledger and report")
    args = parser.parse_args(argv)
    if args.capture.suffix.lower() == ".mp4":
        from sceneagent.video import run_video
        summary = run_video(args.capture, args.out)
        print(
            f"video={summary['video']['name']} bytes={summary['video']['bytes']} "
            f"duration_s={summary['video']['duration_s']} colmap={summary['colmap_available']} "
            f"decision={summary['colmap_decision']} reconstruction={summary['reconstruction']} "
            f"out={summary['out_dir']}"
        )
        return 0
    summary = run_path(args.capture, args.out)
    print(f"scene={summary['scene_name']} objects={summary['n_objects']} sisters={summary['n_sister_rows']} out={summary['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
