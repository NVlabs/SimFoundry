# Changelog

## Unreleased

- Added a `sceneagent` package on the SimFoundry fork. It does not change pipelines A, B, or C.
- Phase 1 writes the paper's file contract: plan, manifest, ledger, material vote, association cuts, model-free sisters, a USD physics contract (JSON only), and a scripted place grid with Wilson intervals.
- `docs/GAP_MATRIX.md` maps each SceneAgent block to what SimFoundry already does, the gap, and the tool the paper or the site names. The PDF prints no git URLs.
- `python -m sceneagent docs/assets/example_videos/Fruits.mp4 --out runs/fruits` reads that example mp4 (1,733,022 bytes, 15.067 s) and defers COLMAP. It does not train a splat.

## Not in this tree

Harvard's SceneAgent code is unreleased. Reconstruction, physics-MLP weights, Meshy sisters, Isaac Lab, MuJoCo, Unreal, and the pi0.5 fine-tune are still open. See the gap matrix.
