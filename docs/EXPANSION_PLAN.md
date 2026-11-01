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

- Section 3.1's inspector reviews the export in Isaac Lab (`SCENEAGENT-INSPECT-ISAAC` in figure 7). Covered under section 3.7. It cannot start until Isaac Lab 2 imports on the Canadian box.

## 3.2 Input preparation

Section 3.2: RGB sequences use COLMAP for cameras and Nerfstudio splatfacto big for the 3DGS. The scene is `S = (G_bg, {o_k})` with `o_k = (P_k, M_k, T_k, theta_k, J_k)`. The same section also accepts existing 3DGS, photogrammetry, LiDAR, and generated scenes (World Labs Marble, NVIDIA Lyra, Tencent HY-World). Dynamic video follows section 3.6.

### COLMAP camera poses

- SimFoundry has no COLMAP stage. Depth is DA3 or FoundationStereo.
- Add `sceneagent.colmap_stage`: ffmpeg sample, `feature_extractor`, `sequential_matcher`, `mapper`, `model_analyzer`. The 1 fps sample and the 16-frame cap are this runner's, not the paper's. A banner that says "without CUDA" forces CPU SIFT.
- Input: `docs/assets/example_videos/Fruits.mp4` (1,733,022 bytes, 15.067 s).
- Output: `colmap/sparse/0` and the analyzer text.
- Tool: https://github.com/colmap/colmap
- Depends on: the mp4.
- Done when: `model_analyzer` prints registered images for a full-rate sequence and those cameras feed splatfacto. Partial today: on instance 58591955, COLMAP 3.7 without CUDA registered 15/15 frames, 674 points, mean reprojection error 0.892696 px. CUDA COLMAP was not started.
- Machine: next attempt is the Canadian box, only after Isaac if disk remains. Texas stays as the CPU record.

### splatfacto big

- SimFoundry stage 2c can train a background splat when `--bg-splat` is set. Default video runs do not.
- Add `ns-train splatfacto` on the COLMAP sparse model, then one rendered view of the trained splat.
- Input: COLMAP images and `sparse/0`.
- Output: a splat checkpoint and one rendered PNG.
- Tool: https://github.com/nerfstudio-project/nerfstudio
- Depends on: COLMAP.
- Done when: the PNG exists and the ledger accepts it. Missing today. Not started, because the Canadian box is exited and the API free-space figure is about 77 GB, and the Texas disk is 40 GB.
- Machine: 55073623 only, and only if free space remains after Isaac Sim 5.

### Photogrammetry, LiDAR, and existing 3DGS

- SimFoundry accepts video and ZED.
- Add a reader for an existing splat, a photogrammetry mesh, or a LiDAR asset, then the same object stages.
- Input: a file the caller already has. Polycam is a capture source on the site and in section 3.5's figure captions, not a git repository.
- Output: the same scene directory as a video run.
- Depends on: sections 3.3 onward.
- Done when: one non-video asset passes the ledger. Missing.
- Machine: Canadian box once the video path works.

### Example video metadata

- SimFoundry ships `docs/assets/example_videos`.
- `sceneagent.video` reads the mp4 header. Partial: duration is known; the splat view is not.
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

- Depends on: posed images from section 3.2.
- Tools: https://github.com/facebookresearch/segment-anything , https://github.com/IDEA-Research/GroundingDINO , https://github.com/mlfoundations/open_clip , https://github.com/facebookresearch/dinov2 , https://github.com/minghanqin/LangSplat
- Done when: mask PNGs, a 512-d crop vector, a DINOv2 vector, and a feature-on-Gaussian file exist for `Fruits.mp4`. All missing.
- Machine: Canadian box, after the splat.

### Gaussian association cuts

- `sceneagent.association` implements the published cuts. Partial: unit tests pass, and no Gaussian center from a trained splat has been labeled.
- Done when: `objects/<name>/gaussians.json` lists centers that passed equation (1).
- Machine: Canadian box, after splatfacto.

### SAM3D meshes and LaMa

- Tools: https://github.com/facebookresearch/sam-3d-objects and https://github.com/advimman/lama
- Depends on: masks, then Gaussians for the ICP.
- Done when: a mesh file and an ICP residual file exist. LaMa is optional and stays missing until a plane-completion render shows an inpainted patch.
- Machine: Canadian box. SAM3D weights are large; skip if free space would drop below the Isaac reserve.

## 3.4 Physical properties, parts, and articulation

Section 3.4 equation (2) picks a material class by a weighted vote. Unknown-class weights scale by 0.4. Ten classes, six body types, eight primitives. Mass and density are log targets. Soft body when deformability is at least 0.65. Young's modulus, Poisson's ratio, and the rendering material are class lookups. The lookup table is not in the paper, so those fields stay null.

The per-Gaussian MLP takes DINOv2, OpenCLIP, and 16 geometric features. It was trained on VLM labels from GSO, YCB, and ABO. Weights are not released. Do not train a stand-in and call it their prior.

PartSAM decomposes the object. CubePart completes hidden geometry. Articulate Anything proposes URDF joints. Agents review the motion. Figure 8 is the marker: masks, Gaussians, mesh, parts, tip under the cap, cap friction, then two Panda arms with Sharpa hands writing.

### Material vote

- `sceneagent.physics` implements equation (2) and the published cutoffs. Keyword weights are this package's, not a published table.
- Output: `physics_prior.json` from the phase-1 capture test.
- Done for the vote file. The MLP row stays missing.
- Machine: developer machine for the vote. Canadian box once real features exist.

### VLM proposal

- Section 3.4 uses a VLM as one vote source and as the audit. Section 3.8 names Gemini 2.5 Flash for sister edits.
- SimFoundry already calls Gemini in stages 3, 5, 6, 11 and pipeline B. Partial: no section 3.4 audit file has been written for `Fruits.mp4`.
- Machine: Canadian box, using the existing Gemini key, without printing it.

### PartSAM, CubePart, Articulate Anything

- Tools: https://github.com/czvvd/PartSAM , https://github.com/Roblox/cube , https://github.com/vlongle/articulate-anything
- Depends on: a SAM3D mesh.
- Done when: a part mesh, a completed hidden region, and a URDF exist and the ledger records a simulation review. All missing.
- Machine: Canadian box, after Isaac Lab can load the URDF.

## 3.5 Fracture

Section 3.5: parchment uses an orthotropic thin sheet in NVIDIA Warp with cohesive fracture. The illustrated rollout fails 49 interfaces and covers 95% of the bowl. Ceramics use NVIDIA Blast (tensile strength 6 MPa, Young's modulus 6 GPa, fracture energy 14 J/m2, density 1850 kg/m3, inner wall 4.5 mm). These numbers are engineering estimates. Neither model is part of automatic export. Section 3.6 and figures 11 and 13 use MPM, FLIP, and Newton for milk, jelly, and watermelon.

- SimFoundry has rigid and articulated bodies only.
- Add explicit example scenes later, not a field on the USD contract.
- Tool: https://github.com/NVIDIA/warp . Blast and Newton have no single URL printed in the PDF.
- Depends on: an Isaac Lab scene.
- Done when: one authored parchment or vessel example runs and the log matches the script, not the paper's 95% coverage unless this repo measures it. Missing.
- Machine: Canadian box, after Isaac Lab.

## 3.6 From 3D to 4D

Section 3.6 extends the scene to `S = (K, E, G, {d_m})` with equation (3). Solvers are MPM, FLIP, and fracture models. The deliverable is a build script that regenerates frames, meshes, and trajectories without reading the video. Section 4.2 scores dynamic IoU, equation (6), on 4DCodeBench B_09, Z_06, and Z_12. The 99.09% figure is not claimed here.

- SimFoundry has no 4D program.
- Add the build-script contract only after a static scene exports. Do not report a dynamic IoU.
- Depends on: sections 3.1 through 3.5.
- Done when: a script emits per-frame meshes and persistent trajectories for one clip, and a separate scorer writes dynamic IoU from those frames. Missing.
- Machine: Canadian box, after the static path.

## 3.7 Scene export and editing

Section 3.7 packages OpenUSD with rigid body, mass, collision, friction, restitution, and density. Deformables, fracture, and fluids do not ride along in that schema. Experiments use Isaac Lab. MuJoCo and Unreal were also verified by the authors. Figure 14 is the Three.js viewer: robot placement, background crop, asset insert, pose and physics edits.

### USD physics contract

- SimFoundry stages 13 and 14 write USD and OmniGibson JSON for that stack.
- Phase 1 writes `package/usd_physics_contract.json` with `usd_written` false. Partial.
- Done when: a `.usd` loads in Isaac Lab and the same fields are present.
- Tool: https://github.com/PixarAnimationStudios/OpenUSD
- Machine: Canadian box.

### Isaac Lab and Isaac Sim

- Official pair to install: Isaac Sim 5.0 or 5.1 with the Isaac Lab 2.x release that declares that Sim 5 version. Not Isaac Sim 4.x.
- Path: the Isaac Lab pip or binary instructions at https://github.com/isaac-sim/IsaacLab , headless.
- Input: the Canadian box with driver 580.x and at least 100 GB free.
- Output: `/workspace/isaac-install.log` and versions in `tools/canadian_isaac_instance.json`.
- Done when: a headless process imports `isaaclab` and opens an empty stage, and the log records the version strings.
- Status: missing. Blocked. Instance 55073623 is exited, and the API free-space figure is about 77 GB. No download was started.
- Machine: 55073623 only.

### MuJoCo clip

- Section 3.7 says exported scenes were loaded in MuJoCo.
- Output: a short headless clip or a load log, not a copied paper video.
- Tool: https://github.com/google-deepmind/mujoco
- Depends on: a real USD or MJCF file.
- Missing. Machine: Canadian box, after the USD exists.

### Unreal and the Three.js viewer

- Unreal is named in section 3.7 and is not a library to vendor. Missing until a person loads the USD.
- SimFoundry's light editor is already Three.js. Partial: extend it, do not replace it. Machine: developer machine, after a captured mesh exists.

### Sister grid page

- Figure 14 and the site show sister-by-lighting place counts. The site adds a 95% Wilson interval and nine light names.
- `sceneagent.report` writes an HTML grid. Partial: the cells are a scripted geometric check, lighting is not simulated, and the counts are not the site's pi0.5 cells.
- Machine: developer machine. Already produced for `examples/phase1_place_scene.json`.

## 3.8 Digital twins and sisters

Section 3.8's first method deforms the mesh within bounds and bakes noise, hue rotation, mottling, and speckle, then checks extent, volume, and grasp. The grasp cutoff is not published; the band in this package is ours. The second method uses Gemini 2.5 Flash, Gemini 3 Pro Image, and Meshy image-to-3D or Meshy-6. Do not label Hunyuan as Meshy.

Figure 7 also names scene sisters and task sisters. Section 3.10 randomizes placement.

- Model-free sisters: `sisters.json` from the phase-1 test. Done for that file. The generative Meshy row is missing.
- Scene and task sisters: templates and horizontal offsets only. Partial. Pipeline B remains the generative layout path.
- Machine: developer machine for the model-free file. Canadian box for any Gemini or Meshy call.

## 3.9 and 3.10, and the scores

Section 3.9 places one converted object into an artist scene. Missing until a captured mesh is inserted in the light editor.

Section 3.10's scripted expert uses simulator state (approach, descend, grasp, lift, carry, lower, release, retreat) and keeps episodes whose object ends inside the target. Partial: `sceneagent.evaluation` is a geometric reach-and-fit check, not an Isaac rollout.

Section 4.1 equations (4) and (5), tau = 1 cm, Hungarian match, unmatched F1 = 0. Partial: the functions run on synthetic clouds. GraspClutter6D `hard/000091` has not been scored. Missing until meshes exist.

The website's Wilson interval is implemented and used on counts this repo produces. That row is done. It does not make the sister-grid cells a reproduction of figure 14.

Section 4.4 fine-tunes pi0.5 with LoRA. The public pi0 code is https://github.com/Physical-Intelligence/openpi . The PDF does not print that URL. Missing. Do not copy the eggplant table.

## Dependency order for the next run

1. Start nothing new until instance 55073623 is running and `df` shows at least 100 GB free. If it does not, stop before any Isaac download.
2. Install Isaac Sim 5.x and the matching Isaac Lab 2.x, headless. Smoke: import `isaaclab` and open an empty stage. Record versions.
3. Sync this tree to `/workspace/sceneagent-simfoundry` without wiping unrelated data.
4. If space remains, CUDA COLMAP and splatfacto on `Fruits.mp4`, then one rendered view.
5. SAM masks, OpenCLIP, DINOv2, association on the real centers, SAM3D mesh, USD, Isaac inspector.
6. Parts, joints, sisters that call Meshy, MuJoCo load, then 4D and fracture examples.

The blocked step is 1. Steps 4 and later do not start while Isaac has no disk.
