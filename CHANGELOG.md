# Changelog

## Unreleased

- Added a `sceneagent` package on the SimFoundry fork. It does not change pipelines A, B, or C.
- Phase 1 writes the paper's file contract: plan, manifest, ledger, material vote, association cuts, model-free sisters, a USD physics contract (JSON only), and a scripted place grid with Wilson intervals.
- `docs/GAP_MATRIX.md` maps each SceneAgent block to what SimFoundry already does, the gap, and the tool the paper or the site names. The PDF prints no git URLs. Upstream links were added for LangSplat, SAM 3D Objects, PartSAM, Roblox Cube (CubePart), Articulate Anything, OpenUSD, Three.js, and the public pi0 code. Those tools stay missing until they actually run.
- Without `colmap` on PATH, `python -m sceneagent docs/assets/example_videos/Fruits.mp4` records that example (1,733,022 bytes, 15.067 s) and defers poses.
- On instance 58591955, `python -m sceneagent docs/assets/example_videos/Fruits.mp4 --out /workspace/fruits-colmap` accepted COLMAP 3.7 poses. The binary banner says "without CUDA", so SIFT ran on CPU. The runner sampled 15 frames at 1 fps (its own cap is 16). `colmap model_analyzer` printed Cameras 1, Images 15, Registered images 15, Points 674, Observations 2656, Mean track length 3.940653, Mean observations per image 177.066667, Mean reprojection error 0.892696px. splatfacto was not trained.
- `docs/EXPANSION_PLAN.md` expands sections 3.1 through 3.10 and the section 4 scores. A block is done only when its artifact exists.
- Instance 55073623 (Quebec, RTX 4090, driver 580.82.09, `ssh1.vast.ai:34206`) was started. Live `df` shows 60 GB free of 250 GB after the download. `nvidia-smi` reports driver 580.82.09 and 24564 MiB.
- Isaac Sim 5.1.0 standalone (`isaac-sim-standalone-5.1.0-linux-x86_64.zip`, 8,768,419,777 bytes) is unpacked at `/workspace/isaacsim`. Isaac Lab v2.3.2 is cloned at `/workspace/IsaacLab` and was not installed. The host is Ubuntu 20.04, glibc 2.31. `mount` is denied, and `/workspace/isaacsim/python.sh` exits 139. `isaaclab` was not imported, so the Isaac Lab row stays missing.
- The fork is at `/workspace/sceneagent-simfoundry` on that box. Instances 58357498 and 58591955 were not used.
- Association cuts and the section 4.1 F1/Chamfer functions moved from done to partial. The formulas are tested. No splat has been labeled, and GraspClutter6D has not been scored.

## Not in this tree

Harvard's SceneAgent code is unreleased. Reconstruction, physics-MLP weights, Meshy sisters, Isaac Lab, MuJoCo, Unreal, and the pi0.5 fine-tune are still open. See the gap matrix.
