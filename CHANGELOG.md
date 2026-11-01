# Changelog

## Unreleased

- Added a `sceneagent` package on the SimFoundry fork. It does not change pipelines A, B, or C.
- Phase 1 writes the paper's file contract: plan, manifest, ledger, material vote, association cuts, model-free sisters, a USD physics contract (JSON only), and a scripted place grid with Wilson intervals.
- `docs/GAP_MATRIX.md` maps each SceneAgent block to what SimFoundry already does, the gap, and the tool the paper or the site names. The PDF prints no git URLs. Upstream links were added for LangSplat, SAM 3D Objects, PartSAM, Roblox Cube (CubePart), Articulate Anything, OpenUSD, Three.js, and the public pi0 code. Those tools stay missing until they actually run.
- Without `colmap` on PATH, `python -m sceneagent docs/assets/example_videos/Fruits.mp4` records that example (1,733,022 bytes, 15.067 s) and defers poses.
- On instance 58591955, `python -m sceneagent docs/assets/example_videos/Fruits.mp4 --out /workspace/fruits-colmap` accepted COLMAP 3.7 poses. The binary banner says "without CUDA", so SIFT ran on CPU. The runner sampled 15 frames at 1 fps (its own cap is 16). `colmap model_analyzer` printed Cameras 1, Images 15, Registered images 15, Points 674, Observations 2656, Mean track length 3.940653, Mean observations per image 177.066667, Mean reprojection error 0.892696px. splatfacto was not trained.

## Not in this tree

Harvard's SceneAgent code is unreleased. Reconstruction, physics-MLP weights, Meshy sisters, Isaac Lab, MuJoCo, Unreal, and the pi0.5 fine-tune are still open. See the gap matrix.
