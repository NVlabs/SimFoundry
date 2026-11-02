# Expansion plan

This plan grows the SimFoundry fork at `sceneagent-simfoundry` toward the behavior in Hollis, Fan, and Yang, *SceneAgent* (arXiv:2610.34745v1). Citations are section numbers from that PDF. The project site is https://computationalrobotics.seas.harvard.edu/SceneAgent/. The PDF prints no git URLs. A repository named below is the upstream project for a system the paper or the site names.

SceneAgent's code is unreleased. This tree does not copy another application repository. Pipelines A, B, and C stay in place. New work lands in the `sceneagent` package and, later, in calls to the tools named in the paper.

Paper metrics are not results of this repository until the same protocol is run here. That includes section 4.1 (Codex F1 60.75% and SimFoundry 6.9% F1 / 4.29 cm Chamfer on GraspClutter6D `hard/000091`) and section 4.2 (Gemini 4 Argon mean dynamic IoU 0.86% to 99.09% on 4DCodeBench B_09, Z_06, and Z_12). Section 4.4's pi0.5 eggplant table is also theirs.

Two published gaps stay explicit. Section 3.4's per-Gaussian MLP weights are not released. Section 3.1's four alignment thresholds are not published, so a missing threshold is undecided, not an accept.

## Machine

Isaac Sim 5 and Isaac Lab 2 are planned for the existing Canadian instance with a 580-series driver. No new machine is rented. Texas instance 58591955 (40 GB disk, CPU COLMAP only) does not receive this install. Estonia instance 58357498 is not the Canadian box and is left alone.

Checked 1 Nov 2026 through the Vast v1 instances API. Every instance was `exited`.

| Id | Geo | Driver | GPU | Disk | Status | Role |
|---|---|---|---|---|---|---|
| 55073623 | Quebec, CA | 580.82.09 | RTX 4090, 24564 MB | 250 GB allocated, API `disk_util` 173 GB (about 77 GB free) | exited | The Canadian 580 box. Install not started. |
| 58331400 | Quebec, CA | 595.99.02 | RTX 3090 | 280 GB, API use 197 GB | exited | Canada, but not a 580 driver. |
| 58357498 | Estonia, EE | 580.178.04 | RTX A4000 | 120 GB, API use 93 GB | exited | Left alone. |
| 58301075 | Texas, US | 595.99.02 | RTX 4090 | 500 GB | exited | Not used. |
| 58591955 | Texas, US | 595.91.07 | RTX 4060 Ti | 40 GB | exited | Prior CPU COLMAP only. |

Isaac Sim 5 plus Isaac Lab 2 needs on the order of 100 GB free. The API figure for 55073623 is about 77 GB free, and the box is stopped, so no live `df` was taken and no packages were downloaded. The next block is blocked on that instance being running with at least 100 GB free. `tools/canadian_isaac_instance.json` records the blocker. It has no API key. Isaac Sim and Isaac Lab versions are null because they were not installed.

CPU phase-1 tests stay on the developer machine. GPU stages, once the disk allows, run only on 55073623.

Later on 1 Nov 2026 the same instance was started. Live `nvidia-smi` shows driver 580.82.09 and an RTX 4090 with 24564 MiB. Live `df` shows 60 GB free of 250 GB after the Isaac Sim 5.1.0 zip (8,768,419,777 bytes) was unpacked at `/workspace/isaacsim`. Isaac Lab v2.3.2 is cloned at `/workspace/IsaacLab`. The host is Ubuntu 20.04 (glibc 2.31). `mount` is denied. A `proot` root with jammy's glibc 2.35 is the path used to finish the import. The fork is at `/workspace/sceneagent-simfoundry`.

## How a block is done

A block is done when its output artifact exists in a run directory and the tool the paper names has actually run. A unit test of a published formula is partial until that artifact exists. Naming an uninstalled repository is missing. Hunyuan is not labeled SAM 3D. DINOv3 is not labeled DINOv2.

## 3.1 Agent orchestration

Section 3.1: an orchestrator writes the plan, ingest builds the 3DGS, and a runner executes each tool with a manifest and a ledger of accepted or rejected files. A reviewer checks registration overlays, eight isolated views, and two assembled views. Figure 7 names the skills.

### Plan, manifest, ledger

- SimFoundry already writes a pipeline manifest for A/B/C. It does not repair a step from a simulator render.
- Add `sceneagent.pipeline`, which writes `plan.json`, one manifest per step, and `ledger.json`.
- Input: a capture JSON, a SimFoundry `objects_info` name list, or an mp4 path.
- Output: those three files.
- Tool: this package. No external repo.
- Depends on: nothing.
- Done when: a run directory contains the plan, the manifest, and a ledger whose video or capture entry is accept or reject. This artifact exists for the phase-1 JSON tests and for `Fruits.mp4`.
- Machine: developer machine. Already produced.

### Alignment acceptance gate

- SimFoundry stage 8 estimates poses. It does not emit the four residuals.
- Add `sceneagent.alignment`. Section 3.1 accepts a pose only when normalized median, 90th percentile, coverage, and silhouette all pass fixed thresholds. The numbers are not in the paper. Missing thresholds stay undecided.
- Input: four residuals plus caller-supplied thresholds. The residuals come from trimmed similarity ICP seeded by the SAM3D pose (section 3.3).
- Output: `alignment.json` with accept, reject, or undecided.
- Tool: ICP later; the gate is in-tree.
- Depends on: SAM3D meshes and Gaussian centers.
- Done when: a mesh-to-splat ICP writes all four residuals and the gate decides them without invented thresholds.
- Machine: Canadian box, after meshes exist. Not run.

### Inspector in Isaac Lab

- Section 3.1 and figure 7 skill `SCENEAGENT-INSPECT-ISAAC`: the inspector reviews the exported scene in Isaac Lab and reports fixes.
- SimFoundry has no Isaac Lab inspector. OmniGibson is the current simulator.
- Input: a USD or USDA scene from section 3.7.
- Output: `inspection/isaac_lab.json` with the import result and any fix the ledger accepts.
- Tool: https://github.com/isaac-sim/IsaacLab , paired with Isaac Sim 5.1.
- Depends on: Isaac Lab importing, and a written USD file.
- Done when: a headless Isaac Lab process opens that file and the ledger records accept or reject. Not run.
- Machine: 55073623 only.

## 3.2 Input preparation

Section 3.2: RGB sequences use COLMAP for cameras and Nerfstudio splatfacto big for the 3DGS. The scene is `S = (G_bg, {o_k})` with `o_k = (P_k, M_k, T_k, theta_k, J_k)`. The same section also accepts existing 3DGS, photogrammetry, LiDAR, and generated scenes (World Labs Marble, NVIDIA Lyra, Tencent HY-World). Dynamic video follows section 3.6.

### COLMAP camera poses

- SimFoundry has no COLMAP stage. Depth is DA3 or FoundationStereo.
- Add `sceneagent.colmap_stage`: ffmpeg sample, `feature_extractor`, `sequential_matcher`, `mapper`, `model_analyzer`. The 1 fps sample and the 16-frame cap are this runner's, not the paper's. A banner that says "without CUDA" forces CPU SIFT.
- Input: `docs/assets/example_videos/Fruits.mp4` (1,733,022 bytes, 15.067 s).
- Output: `colmap/sparse/0` and the analyzer text.
- Tool: https://github.com/colmap/colmap
- Depends on: the mp4.
- Done when: `model_analyzer` prints registered images for a full-rate sequence and those cameras feed splatfacto. Partial: Texas 58591955, COLMAP 3.7 without CUDA, 15/15 frames, 674 points, 0.892696 px. On 55073623, COLMAP 4.2.1 with CUDA, same 1 fps sample, 15/15 images, 603 points, 0.867273 px, exit 0, artifact `/workspace/fruits-cuda-colmap/colmap/sparse/0`. The 1 fps cap is this runner's.
- Machine: next attempt is the Canadian box, only after Isaac if disk remains. Texas stays as the CPU record.

### splatfacto big

- SimFoundry stage 2c can train a background splat when `--bg-splat` is set. Default video runs do not.
- Add `ns-train splatfacto` on the COLMAP sparse model, then one rendered view of the trained splat.
- Input: COLMAP images and `sparse/0`.
- Output: a splat checkpoint and one rendered PNG.
- Tool: https://github.com/nerfstudio-project/nerfstudio
- Depends on: COLMAP.
- Done when: the PNG exists and the ledger accepts it. A short train did run: `ns-train splatfacto-big`, 2000 iterations, exit 0. Render `/workspace/fruits-splat/renders/train/rgb/0002.jpg` (JPEG). Checkpoint `step-000001999.ckpt`. The iteration cap is this run's.
- Machine: 55073623 only, and only if free space remains after Isaac Sim 5.

### Photogrammetry, LiDAR, and existing 3DGS

- SimFoundry accepts video and ZED.
- Add a reader for an existing splat, a photogrammetry mesh, or a LiDAR asset, then the same object stages.
- Input: a file the caller already has. Polycam is a capture source on the site and in section 3.5's figure captions, not a git repository.
- Output: the same scene directory as a video run.
- Tool: the reader's format (existing 3DGS, photogrammetry mesh, or LiDAR). No extra git URL is printed in the PDF.
- Depends on: sections 3.3 onward.
- Done when: one non-video asset passes the ledger. Missing.
- Machine: Canadian box once the video path works.

### Example video metadata

- Section 3.2 starts from an RGB sequence. SimFoundry already ships the example videos.
- Input: `docs/assets/example_videos/Fruits.mp4`.
- Output: `video.json` with byte size and `mvhd` duration.
- Tool: `sceneagent.video`. Repo: https://github.com/NVlabs/SimFoundry
- Depends on: the file in the tree.
- Done when: the header duration is recorded. Partial: duration is known; the splat view is not.
- Machine: developer machine for the header; Canadian box for the splat.

## 3.3 Object reconstruction

Section 3.3 follows LangSplat's approach of attaching features to Gaussians. SAM masks every tenth view by default (stride 1 to 10 across their scenes). OpenCLIP ViT-B-16 embeds each crop to 512 dimensions. DINOv2 ViT-S/14 is the other feature. Neither feature assigns Gaussians. Equation (1) does: the center projects inside the mask and the depth is within `max(0.08, 0.03 * z_hat)` of the front of a cell of side `min(16, round(short_side/192))`. Masks group when centroids are within `min(0.10, 0.6 e)` and containment is at least 0.18 or Jaccard is at least 0.08. SAM3D builds the mesh. Trimmed similarity ICP starts from the SAM3D pose. Background completion uses detected planes and optional LaMa. The site also names GroundingDINO beside SAM; section 3.3 does not.

### Evaluation outputs from this section

| Artifact | Paper | Status |
|---|---|---|
| Splat views | Section 3.2 training, section 3.3 inspection renders | Missing. No splat. |
| Masks | Section 3.3, every tenth view | Missing. SAM has not run. |
| Features | Section 3.3, OpenCLIP 512-d and DINOv2 ViT-S/14 | Missing. Do not substitute DINOv3. |
| Meshes | Section 3.3, SAM3D | Missing. Hunyuan, TRELLIS.2, and Pixal3D in stage 7 are not SAM3D. |

### SAM, GroundingDINO, OpenCLIP, DINOv2, LangSplat

- Input: posed frames from `Fruits.mp4`, one mask every tenth view.
- Output: mask PNGs, one OpenCLIP ViT-B-16 512-d vector, one DINOv2 ViT-S/14 vector, and a feature-on-Gaussian file.
- Depends on: posed images from section 3.2.
- Tools: https://github.com/facebookresearch/segment-anything , https://github.com/IDEA-Research/GroundingDINO , https://github.com/mlfoundations/open_clip , https://github.com/facebookresearch/dinov2 , https://github.com/minghanqin/LangSplat
- Done when: those four files exist for `Fruits.mp4`. All missing.
- Machine: Canadian box, after the splat.

### Gaussian association cuts

- Input: Gaussian centers from splatfacto and the SAM masks.
- Output: `objects/<name>/gaussians.json`.
- Tool: `sceneagent.association`, equation (1). No separate git URL.
- Depends on: a trained splat and the masks.
- `sceneagent.association` implements the published cuts. On 55073623 the cut kept 27548 of 84993 centers for the largest mask on `0001.jpg`. Artifact: `/workspace/fruits-features/gaussians.json`.
- Done when: that JSON lists centers that passed equation (1). That file does.
- Machine: Canadian box, after splatfacto.

### SAM3D meshes and LaMa

- Input: a SAM mask and the posed views.
- Output: a mesh file and an ICP residual file. LaMa's output is an inpainted patch.
- Tools: https://github.com/facebookresearch/sam-3d-objects and https://github.com/advimman/lama
- Depends on: masks, then Gaussians for the ICP.
- Done when: a mesh file and an ICP residual file exist. LaMa is optional and stays missing until a plane-completion render shows an inpainted patch.
- Machine: Canadian box. SAM3D weights are large; skip if free space would drop below the Isaac reserve.

## 3.4 Physical properties, parts, and articulation

Section 3.4 equation (2) picks a material class by a weighted vote. Unknown-class weights scale by 0.4. Ten classes, six body types, eight primitives. Mass and density are log targets. Soft body when deformability is at least 0.65. Young's modulus, Poisson's ratio, and the rendering material are class lookups. The lookup table is not in the paper, so those fields stay null.

The per-Gaussian MLP takes DINOv2, OpenCLIP, and 16 geometric features. It was trained on VLM labels from GSO, YCB, and ABO. Weights are not released. Do not train a stand-in and call it their prior.

PartSAM decomposes the object. CubePart completes hidden geometry. Articulate Anything proposes URDF joints. Agents review the motion. Figure 8 is the marker: masks, Gaussians, mesh, parts, tip under the cap, cap friction, then two Panda arms with Sharpa hands writing.

### Material vote

- Input: per-source class weights. Keyword weights are this package's, not a published table.
- Output: `physics_prior.json` from the phase-1 capture test.
- Tool: `sceneagent.physics`, equation (2).
- Depends on: nothing for the keyword vote. The MLP depends on unpublished weights and stays missing.
- Done when: the vote file exists. That file exists. The MLP row stays missing.
- Machine: developer machine for the vote. Canadian box once real features exist.

### VLM proposal

- Section 3.4 uses a VLM as one vote source and as the audit. Section 3.8 names Gemini 2.5 Flash for sister edits.
- SimFoundry already calls Gemini in stages 3, 5, 6, 11 and pipeline B.
- Input: one frame from `Fruits.mp4` and the existing Gemini key. The key is not written into the repo.
- Output: `ledger.json` entry for the material proposal, plus the class returned by Gemini 2.5 Flash.
- Tool: Gemini 2.5 Flash. No git URL is printed in the PDF.
- Depends on: a decoded frame. It does not depend on the unpublished MLP weights.
- Done when: the ledger records that call's class and the raw response is saved. Partial until that file exists.
- Machine: Canadian box or the developer machine. The key stays in `.env.local`.

### PartSAM, CubePart, Articulate Anything

- Tools: https://github.com/czvvd/PartSAM , https://github.com/Roblox/cube , https://github.com/vlongle/articulate-anything
- Depends on: a SAM3D mesh.
- Done when: a part mesh, a completed hidden region, and a URDF exist and the ledger records a simulation review. All missing.
- Machine: Canadian box, after Isaac Lab can load the URDF.

## 3.5 Fracture

Section 3.5: parchment uses an orthotropic thin sheet in NVIDIA Warp with cohesive fracture. The illustrated rollout fails 49 interfaces and covers 95% of the bowl. Ceramics use NVIDIA Blast (tensile strength 6 MPa, Young's modulus 6 GPa, fracture energy 14 J/m2, density 1850 kg/m3, inner wall 4.5 mm). These numbers are engineering estimates. Neither model is part of automatic export. Section 3.6 and figures 11 and 13 use MPM, FLIP, and Newton for milk, jelly, and watermelon.

- SimFoundry has rigid and articulated bodies only.
- Input: an authored parchment or ceramic vessel scene, not a field on the USD contract.
- Output: a Warp or Blast run log for that scene.
- Tool: https://github.com/NVIDIA/warp . Blast and Newton have no single URL printed in the PDF.
- Depends on: an Isaac Lab scene.
- Done when: one authored parchment or vessel example runs and the log matches the script, not the paper's 95% coverage unless this repo measures it. Missing.
- Machine: Canadian box, after Isaac Lab.

## 3.6 From 3D to 4D

Section 3.6 extends the scene to `S = (K, E, G, {d_m})` with equation (3). Solvers are MPM, FLIP, and fracture models. The deliverable is a build script that regenerates frames, meshes, and trajectories without reading the video. Section 4.2 scores dynamic IoU, equation (6), on 4DCodeBench B_09, Z_06, and Z_12. The 99.09% figure is not claimed here.

- SimFoundry has no 4D program.
- Input: one clip after a static scene exports.
- Output: per-frame meshes, persistent trajectories, and a separate dynamic-IoU file. Do not report the paper's 99.09%.
- Tool: the build script. MPM, FLIP, and 4DCodeBench have no git URL printed in the PDF.
- Depends on: sections 3.1 through 3.5.
- Done when: a script emits per-frame meshes and persistent trajectories for one clip, and a separate scorer writes dynamic IoU from those frames. Missing.
- Machine: Canadian box, after the static path.

## 3.7 Scene export and editing

Section 3.7 packages OpenUSD with rigid body, mass, collision, friction, restitution, and density. Deformables, fracture, and fluids do not ride along in that schema. Experiments use Isaac Lab. MuJoCo and Unreal were also verified by the authors. Figure 14 is the Three.js viewer: robot placement, background crop, asset insert, pose and physics edits.

### USD physics contract

- SimFoundry stages 13 and 14 write USD and OmniGibson JSON for that stack.
- Input: `package/usd_physics_contract.json` from the phase-1 pipeline.
- Output: a `.usd` or `.usda` with the rigid-body fields in that JSON. Phase 1 leaves `usd_written` false. Partial.
- Depends on: the contract JSON. Loading it depends on Isaac Lab.
- Done when: a `.usd` loads in Isaac Lab and the same fields are present.
- Status: done for `phase1_place.usda`. Command: `/workspace/load_usd.py` under the jammy loader. Exit 0. Artifact: `/workspace/usd-load-result.txt` (`opened=True`, prims `/World`, `/World/box`, `/World/bowl`). The phase-1 JSON still records `usd_written` false; the USDA is a separate writer.
- Tool: https://github.com/PixarAnimationStudios/OpenUSD
- Machine: Canadian box.

### Isaac Lab and Isaac Sim

- Official pair to install: Isaac Sim 5.0 or 5.1 with the Isaac Lab 2.x release that declares that Sim 5 version. Not Isaac Sim 4.x.
- Path: the Isaac Lab pip or binary instructions at https://github.com/isaac-sim/IsaacLab , headless.
- Input: the Canadian box with driver 580.x. Live free space on 1 Nov 2026 is 60 GB after the 5.1.0 unpack. That is enough to finish this Lab install. It is not enough for a second simulator download.
- Output: `/workspace/isaac-install.log` and versions in `tools/canadian_isaac_instance.json`.
- Done when: a headless process imports `isaaclab` and opens an empty stage, and the log records the version strings.
- Status: done for the empty-stage smoke. Isaac Sim `5.1.0-rc.19+release.26219.9c81211b.gl`, Isaac Lab v2.3.2, Python package `isaaclab` 0.54.2. Command: jammy `ld-linux` plus `/workspace/empty_stage.py`. Exit 0. Artifact: `/workspace/empty-stage-result.txt` (`EMPTY_STAGE_OK anon:0x389f960:World1.usd`). `python.sh` on the host glibc 2.31 still segfaults; the working loader is `/workspace/jammy/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2`.
- Machine: 55073623 only.

### MuJoCo clip

- Section 3.7 says exported scenes were loaded in MuJoCo.
- Input: the USD from the contract, or an MJCF converted from it.
- Output: a short headless clip or a load log, not a copied paper video.
- Tool: https://github.com/google-deepmind/mujoco
- Depends on: a real USD or MJCF file.
- Missing. Machine: Canadian box, after the USD exists.

### Unreal and the Three.js viewer

- Input: the same USD.
- Output: a person-confirmed Unreal load, and a Three.js view of a captured mesh in SimFoundry's light editor.
- Tool: Unreal has no git URL in the PDF. Three.js is https://github.com/mrdoob/three.js and is already the light editor.
- Depends on: a written USD for Unreal, and a captured mesh for the viewer.
- Done when: Unreal has been loaded by a person, and the light editor shows a captured mesh. Both missing as a reproduction. Partial for the existing editor.
- Machine: a person's machine for Unreal. Developer machine for Three.js, after a captured mesh exists.

### Sister grid page

- Input: `examples/phase1_place_scene.json` and the scripted grid counts.
- Output: the HTML grid from `sceneagent.report`.
- Tool: `sceneagent.report`. Figure 14's viewer is Three.js; the site's pi0.5 cells are not this file.
- Depends on: the phase-1 evaluation counts.
- Done when: the HTML exists. Partial: the cells are a scripted geometric check, lighting is not simulated, and the counts are not the site's pi0.5 cells.
- Machine: developer machine. Already produced for `examples/phase1_place_scene.json`.

## 3.8 Digital twins and sisters

Section 3.8's first method deforms the mesh within bounds and bakes noise, hue rotation, mottling, and speckle, then checks extent, volume, and grasp. The grasp cutoff is not published; the band in this package is ours. The second method uses Gemini 2.5 Flash, Gemini 3 Pro Image, and Meshy image-to-3D or Meshy-6. Do not label Hunyuan as Meshy.

Figure 7 also names scene sisters and task sisters. Section 3.10 randomizes placement.

- Input: a mesh extent and a task instruction for the model-free path. An image for the Gemini and Meshy path.
- Output: `sisters.json` for the model-free deformation. A generated asset for Meshy. Scene and task sister JSON for the template offsets.
- Tool: `sceneagent.sisters` for the model-free path. Gemini 2.5 Flash and Meshy have no git URL in the PDF. Do not label Hunyuan as Meshy.
- Depends on: a mesh for the deformation. The Meshy call also depends on an image.
- Done when: `sisters.json` exists. That file exists, so the model-free row is done. The generative Meshy row is missing. Scene and task sisters are templates and horizontal offsets only, so that row stays partial. Pipeline B remains the generative layout path.
- Machine: developer machine for the model-free file. Canadian box for any Gemini or Meshy call.

## 3.9 and 3.10, and the scores

Section 3.9 places one converted object into an artist scene.

- Input: a captured mesh and an artist scene in the light editor.
- Output: the scene file with that mesh inserted.
- Tool: SimFoundry's light editor (Three.js).
- Depends on: a captured mesh.
- Done when: the inserted mesh is in the saved scene. Missing.
- Machine: developer machine.

Section 3.10's scripted expert uses simulator state (approach, descend, grasp, lift, carry, lower, release, retreat) and keeps episodes whose object ends inside the target.

- Input: `examples/phase1_place_scene.json`.
- Output: the scripted grid counts.
- Tool: `sceneagent.evaluation`. An Isaac rollout is a later tool, Isaac Lab.
- Depends on: the phase-1 scene. An Isaac rollout depends on the USD load.
- Done when: an Isaac rollout keeps episodes whose object ends inside the target. Partial today: the geometric reach-and-fit check runs, and it is not an Isaac rollout.
- Machine: developer machine for the geometric check. 55073623 for the Isaac rollout.

Section 4.1 equations (4) and (5), tau = 1 cm, Hungarian match, unmatched F1 = 0.

- Input: predicted and reference point clouds. The paper's scene is GraspClutter6D `hard/000091`.
- Output: an F1 and Chamfer file for that scene.
- Tool: `sceneagent.metrics`. GraspClutter6D has no git URL printed in the PDF.
- Depends on: meshes from section 3.3.
- Done when: that scene is scored. Partial: the functions run on synthetic clouds. The GraspClutter scene has not been scored.
- Machine: Canadian box, after meshes exist.

The website's Wilson interval is implemented and used on counts this repo produces.

- Input: the sister-grid place counts this repo produces.
- Output: the interval in the report.
- Tool: `sceneagent.evaluation`.
- Depends on: those counts.
- Done when: the report quotes the interval. That row is done. It does not make the sister-grid cells a reproduction of figure 14.
- Machine: developer machine.

Section 4.4 fine-tunes pi0.5 with LoRA.

- Input: demonstrations from the scripted expert. The paper's table is not an input to copy.
- Output: a LoRA checkpoint and this repo's own success counts.
- Tool: https://github.com/Physical-Intelligence/openpi . The PDF does not print that URL.
- Depends on: Isaac Lab episodes.
- Done when: a LoRA run writes those counts. Missing. Do not copy the eggplant table.
- Machine: 55073623, after the inspector and the expert exist.

## Dependency order for the next run

1. Start nothing new until instance 55073623 is running and `df` shows at least 100 GB free. If it does not, stop before any Isaac download.
2. Install Isaac Sim 5.x and the matching Isaac Lab 2.x, headless. Smoke: import `isaaclab` and open an empty stage. Record versions.
3. Sync this tree to `/workspace/sceneagent-simfoundry` without wiping unrelated data.
4. If space remains, CUDA COLMAP and splatfacto on `Fruits.mp4`, then one rendered view.
5. SAM masks, OpenCLIP, DINOv2, association on the real centers, SAM3D mesh, USD, Isaac inspector.
6. Parts, joints, sisters that call Meshy, MuJoCo load, then 4D and fracture examples.

Step 1 is past the start. The live free space is 60 GB, so a second full simulator download is not planned. The open step is the glibc 2.35 import. Splatfacto waits until that import's disk use is known.
