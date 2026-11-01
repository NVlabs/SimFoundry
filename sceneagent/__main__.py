"""Run the phase-1 SceneAgent slice.

    python -m sceneagent examples/phase1_place_scene.json --out runs/phase1_place
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sceneagent.pipeline import run_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="SceneAgent phase-1 conversion")
    parser.add_argument("capture", type=Path, help="Phase-1 capture JSON or a SimFoundry scene JSON")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the ledger, sisters, and report")
    args = parser.parse_args(argv)
    summary = run_path(args.capture, args.out)
    print(f"scene={summary['scene_name']} objects={summary['n_objects']} sisters={summary['n_sister_rows']} out={summary['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
