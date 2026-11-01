# SceneAgent on SimFoundry — design

This document is the upgrade plan for a new tree at `sceneagent-simfoundry`. The tree is the SimFoundry fork (`sandy-t-shelvan/SimFoundry`, parent `NVlabs/SimFoundry`, commit `9e34ebe`) plus a `sceneagent` package. SceneAgent, the Harvard paper and site, is the target behavior. It is not a second codebase in this tree. No other application repository is imported.

The row-by-row map is [`GAP_MATRIX.md`](GAP_MATRIX.md), generated from `sceneagent/catalog.py`. Current counts: 6 done, 9 partial, 23 missing, 38 rows. Done means a published rule is implemented in this package. Partial means a sampled run or an interface exists and the rest of that block is still open. Missing means the named system has not been run.

This document does not treat the SceneAgent website's demo numbers as results of this repository.

## Sources

- Hollis, Fan, and Yang, *SceneAgent: An Agentic Pipeline from 3D and 4D Captures to Simulation Environments*, Harvard Computational Robotics Group, preprint 2026, arXiv:2610.34745v1 (26 Oct 2026). Local PDF read in full.
- Project site https://computationalrobotics.seas.harvard.edu/SceneAgent/ (read 31 Oct 2026): intro, pipeline, physics prior, agent graph, hybrid scenes, digital twins and sisters, policy training, evaluation, citation. The page is one scrolling article. Code is labeled coming soon. Videos on the page include the lab cover, the marker writing sequence, harness comparison renders, 4DCodeBench source/target clips (B_09, Z_06, Z_12), fracture and material clips (parchment, milk, jelly, watermelon, salmon), deformable captures (textile, hoodie, towels, hose, corduroy hat, extension cord), hybrid task clips (kitchen, warehouse, chemistry lab, data center, studio), and the real/sim wrist mosaic.
- Ranawaka et al., *SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation*, arXiv:2606.28276v4, plus the NVlabs/SimFoundry tree at commit `9e34ebe` ("Refinement (#7)"), including `README.md`, `AGENTS.md`, `docs/AGENT_INSTALL.md`, and `scripts/pipeline/README.md`.

SceneAgent's code, processed scenes, and example objects are not released. The site says they will ship with the paper. This repository reimplements the published phase-1 contract. It is not a drop-in of their unreleased code.

## What SceneAgent produces

SceneAgent converts 3D Gaussian splats, photogrammetry, LiDAR, generated 3DGS (World Labs Marble, NVIDIA Lyra, Tencent HY-World), and 4D video into an interactive simulation. The output the paper and site actually show is:

1. A scene `S = (G_bg, {o_k})` with a completed background splat and objects `o_k = (P_k, M_k, T_k, theta_k, J_k)`: Gaussians, mesh, rigid transform plus uniform scale, physical properties, and joints (section 3.2).
2. Per-object physics: ten material classes, six body types, eight collision primitives, and numerical priors (friction, rigidity in `[0, 1]`, mass and density as logarithms). Young's modulus, Poisson's ratio, and rendering parameters are class lookups. The lookup table is not in the paper. An object is a soft body when deformability is at least 0.65 (section 3.4).
3. Parts (PartSAM), hidden-geometry completion (CubePart), and URDF joints (Articulate Anything), reviewed in simulation. The marker example completes the tip under the cap and writes with two Panda arms and Sharpa hands (figure 8).
4. OpenUSD with rigid-body mass, collision, friction, restitution, and density, aimed at Isaac Lab, MuJoCo, and Unreal. Deformables, fracture, and fluids do not travel through that USD schema (section 3.7).
5. Digital sisters. The model-free path deforms the mesh within bounds and bakes noise, hue rotation, mottling, and speckle. The generative path uses Gemini 2.5 Flash, Gemini 3 Pro Image, and Meshy (section 3.8). The site also randomizes placement and lighting.
6. A Three.js editor for robot placement, background cropping, Omniverse insertion, pose and physics edits, sisters, and parts (figure 14).
7. A policy loop (figure 19 and the site's six steps): real-world failure review, twin plus sisters, policy evaluation, expert-controller revision, fine-tune on episodes that pass the success check, real-world re-evaluation. The demonstration factory is a scripted expert on ground-truth poses (approach, descend, grasp, lift, carry, lower, release, retreat) with observations composited over the splat (section 3.10).
8. An experiment matrix: object sister by lighting, place success over ten trials (figure 14). The site adds a 95% Wilson interval on each cell and names nine lights (Neutral, Strong warm, Strong cool, Very dim, Very bright, Hard left, Hard right, Backlit, Overhead spot) and sister rows such as box length ±5%, box cool, box fiber, and box print. Those cell counts are their pi0.5 runs. This repo does not copy them.
9. Reconstruction scores on GraspClutter6D `hard/000091` (20 YCB objects, 13 RGB views): F1 at 1 cm and symmetric Chamfer, Hungarian-matched, unmatched F1 = 0. Their best reported F1 is 60.75% (Codex with skills) against SimFoundry at 6.9% F1 and 4.29 cm Chamfer (figure 5, section 4.1). On three 4DCodeBench cases, Gemini 4 Argon's mean dynamic IoU goes from 0.86% to 99.09% (section 4.2). Preliminary pi0.5 transfer is an eggplant-into-basket task with the episode-count table in section 4.4 (for example 40 simulated episodes: 57/200 in sim, 24/50 in the real world; policy 3: 170/200 sim, 44/50 real). Those numbers were not re-run here.

Agents coordinate this with a plan, a manifest per step, and a ledger of accepted or rejected outputs (section 3.1). Figure 7 names the skills, including `SCENEAGENT-RUN-PIPELINE`, `SCENEAGENT-INGEST-DATA`, `SCENEAGENT-OBJECT-SISTERS`, `SCENEAGENT-POLICY-EVAL`, and `SCENEAGENT-ADAPT-VLA-POLICIES`.

The paper is explicit about limits that this design keeps: the physics MLP is trained on VLM labels, not measurements; fracture parameters are engineering estimates and are not part of automatic export; mass, friction, rigidity, and joints are not calibrated against robot interaction; the reconstruction study is one kitchen and one bin, each converted once per harness; real-world results are pick-and-place and still being evaluated.

## What SimFoundry already does

SimFoundry (commit `9e34ebe`) turns a short video into an OmniGibson scene.

| Pipeline | What the code does | Main outputs |
|---|---|---|
| A reconstruction | 14 stages from video or ZED through depth, support plane, decomposition, Hunyuan/TRELLIS/Pixal3D meshes, pose, optional articulation, physics compile, USD import, OmniGibson JSON | `Data/<scene>/s14_og/reconstructed_og_scene.json` |
| B augmentation | VLM/image digital cousins, cousin meshes, scene sampling, task YAML | `proposed_tasks/*.yaml`, cousin USD |
| C application | Smoke, policy eval, teleop, waypoint demos, replay in OmniGibson | `s15_eval/`, `s17_generated_demos/`, and related stage folders |

The light editor (`scripts/interactive/light_editor/`) is a Three.js tool for pose, scale, cameras, tasks, and a ground plane. It does not run Isaac. Saved scenes are JSON with the contract in `AGENTS.md` (meters, Z-up, XYZW quaternions). Full install wants about 250 GB, mamba, and `/usr/local/cuda-12.8`. VLM stages use Vertex or `GEMINI_API_KEY`.

The SimFoundry paper's own claims, which this design does not re-measure: across 12 scenes, zero-shot F1 of 0.81–0.92; across 7 tasks and 5 policies, mean Pearson correlation 0.911 between sim and real; cousin training gains of about 17% (object), 21% (scene), and 40% (task). SceneAgent's GraspClutter comparison uses a different protocol (20 occluded YCB objects, F1 at 1 cm against dataset meshes, SimFoundry stage 10 at native scale) and reports 6.9%. Both numbers can be true because they are different tests. This repo has not repeated either one.

## Gap

| | SimFoundry today | SceneAgent paper and site |
|---|---|---|
| Input | One video or ZED capture, one support plane | 3DGS, photogrammetry, LiDAR, generated worlds, 4D video |
| Agent loop | Fixed stage scripts. A human edits in the light editor | Orchestrator, manifests, ledger, simulator renders, repair |
| Physics | Stage 11 / articulation estimates for OmniGibson (mass, friction, joints) | Per-Gaussian prior, 10/6/8 taxonomy, deformability ≥ 0.65, VLM audit. Not claimed to be measured |
| Deform / fluid / fracture | Not in the A/B/C stages | MPM, FLIP, Warp, Newton, Blast. Fracture examples are hand-authored, not automatic export |
| Variants | Digital cousins via VLM images and mesh generation (pipeline B) | Digital sisters: model-free mesh and texture edits, plus Gemini/Meshy |
| Layouts and tasks | Scene sampling and task YAML with spatial predicates | Scene sisters around a task anchor; task sisters change the instruction; failure-driven sisters |
| Export | OmniGibson JSON and USD assets for that stack | USD / USDZ for Isaac Lab, MuJoCo, and Unreal |
| Policy | Teleop, waypoints, OmniGibson eval (pipeline C) | Scripted expert, sister × lighting matrix, LoRA fine-tune of a VLA (pi0.5 in section 4.4), real-robot loop |
| Metrics in the paper | F1 on their reconstruction scenes; Pearson sim-vs-real | F1 at 1 cm, Chamfer, dynamic IoU, Wilson intervals on the site |
| Editor | Light editor, no Isaac | Three.js editor plus an agent operator and an experiment grid |

SimFoundry stays the reconstruction and cousin engine. The upgrade is a new `sceneagent` package. It does not rewrite `scripts/pipeline/A_reconstruction`, `B_augmentation`, or `C_application`.

## Architecture

```
capture JSON  or  SimFoundry objects_info names
        |
        v
sceneagent.pipeline
  plan.json + per-step manifest + ledger.json
  physics vote  ->  objects/<name>/physics_prior.json
  model-free sisters, scene layouts, task instructions
  package/usd_physics_contract.json     (no .usd yet)
  evaluation/sister_grid.json + report.html
```

Later phases call SimFoundry stages and external models through the same ledger. They do not replace those stages.

Skill IDs match figure 7. Phase 1 implements the file, vote, mesh, and scripted-check skills. `SCENEAGENT-ALIGN-SCENE` is a gate only: four residuals are accepted only when the caller supplies thresholds, because the paper's thresholds are unpublished. ICP itself is not run.

## Phased SDLC

Stages used for every phase: requirements (this document plus the paper section), design (interfaces in `sceneagent`), implementation, test (`pytest tests/sceneagent` on CPU), release (merge to `main` after CI). Definition of done and landing rules are in `docs/SDLC.md`.

### Phase 1 — built in this tree

A CPU run that writes SceneAgent's file contract and the experiment-grid shape.

- Ingest a phase-1 capture or object names from a SimFoundry `objects_info.init_info` block. No USD load, no OmniGibson.
- Material vote with the 0.4 unknown-class scale. Keyword dictionary is ours; the paper says keyword priors exist and does not print them.
- Logs of caller-supplied mass and density. Class lookups for Young's modulus, Poisson's ratio, and rendering stay null.
- Soft body at deformability ≥ 0.65.
- Mask-grouping and front-surface tests with the published constants (section 3.3).
- Model-free sisters: axis scale, volume and extent, a labeled grasp/volume gate (the paper's numeric gate is unpublished; ±5% is the site's geometry axis), procedural texture parameters.
- Scene-sister offsets and three task instructions. Relation names and wording are phase-1 templates.
- USD physics contract JSON for the attributes in section 3.7. No `.usd` bytes.
- Ten-trial scripted place check and Wilson intervals. Lighting columns use the site's nine names and are marked not simulated.
- HTML report in the shape of figure 14.
- F1 and Chamfer functions for later scoring. They are not pointed at GraspClutter6D.

Run:

```bash
python -m sceneagent examples/phase1_place_scene.json --out runs/phase1_place
```

`runs/` is gitignored. The example's alignment thresholds are configuration for the gate, not the paper's missing numbers. The example's +40% length sister is there so the fit check can reject a mesh; the site's published grid uses ±5%, which this example also includes.

### Phase 2 — next, still no new benchmark claims

- Point the ingest at a finished SimFoundry `s14_og` scene and pipeline B task YAML, read-only.
- Optional VLM proposal through the existing Gemini path, cached the way SimFoundry `--cache-mode` caches model calls. The vote already accepts an extra proposal.
- Write sister specs next to B's cousin outputs without changing B's scripts.

### Phase 3 — alignment numbers

- When stage 8 emits median, p90, coverage, and silhouette, pass them through the gate.
- Choose thresholds on a held-out set and record them as our calibration. Do not present them as the paper's.

### Phase 4 — USD writer

- Emit USD rigid attributes from the contract, using the USD tools SimFoundry already vendors, and load-check outside CI.
- Leave deformables out of that file, matching section 3.7.

### Phase 5 — editor

- Add the experiment grid and sister list to the light editor. The phase-1 HTML report is the stand-in.

### Phase 6 — reconstruction skills on a GPU box

- COLMAP, splatfacto, SAM / GroundingDINO, OpenCLIP, DINOv2, SAM3D. This is the 250 GB class of install. Do it on a machine with the disk and CUDA the SimFoundry installer requires. The Texas text box from this pass is not that machine.

### Phase 7 — parts and joints

- PartSAM, CubePart, Articulate Anything, reviewed by the ledger. SimFoundry stage 9 remains available for the video path.

### Phase 8 — 4D, fluids, fracture

- Only as explicit example scenes. The paper says these solvers are not part of automatic export.

### Phase 9 — learned policies

- Scripted demonstrations in a simulator, then LoRA fine-tune. Report only runs we actually execute. The eggplant table in section 4.4 is theirs.

## What phase 1 deliberately does not do

- No COLMAP, 3DGS, SAM, SAM3D, PartSAM, CubePart, Articulate Anything, Meshy, Isaac, MuJoCo, Unreal, or OmniGibson.
- No trained physics MLP. The paper's network inputs are DINOv2, OpenCLIP, and 16 geometric features; the weights are unreleased.
- No invented friction, mass, modulus, or ICP thresholds.
- No GraspClutter6D, MessyKitchens, or 4DCodeBench scores.
- No pi0.5, GR00T, or real-robot loop.
- No edit to the running Vast pipeline on instance `58357498`.

## Risks

- Unpublished constants (ICP thresholds, modulus table, grasp cutoff, the nine numerical head names) will be wrong if we fill them in early. Phase 1 leaves them open.
- SceneAgent and SimFoundry F1 numbers are not comparable without repeating both protocols.
- A full SimFoundry install needs ~250 GB and `/usr/local/cuda-12.8`. Putting that on the small Texas box will fail the disk check in `docs/AGENT_INSTALL.md`.
- Generative sisters and VLM audits can drift. Cache them and keep the model-free sister path as the deterministic baseline.
- The site's lighting matrix will look uniform until a renderer exists. The report says lighting is not simulated rather than inventing per-light rates.
- Physics priors are starting points for randomization. Using them as ground truth would contradict section 3.4 and section 5.

## Honest scope

Phase 1 is the agent file contract, the published association and vote math, model-free sisters, and a scripted place grid with Wilson intervals. It is the slice that can be tested on CPU and extended without breaking A/B/C. It is not a reproduction of the SceneAgent website.

## Block map

The same rows are in docs/GAP_MATRIX.md.

Counts: 6 done, 9 partial, 23 missing, 38 rows.

| Block | Sub-block | Paper requirement | SimFoundry today | Status | Tool | Repo | Next action |
|---|---|---|---|---|---|---|---|
| 3.1 Agent orchestration | Plan, manifest, ledger | Section 3.1: agents pass a plan, a manifest per step, and a ledger of accepted or rejected files. | A/B/C runners write a pipeline manifest. They do not repair from simulator renders. | done | sceneagent.pipeline |  | Keep new stages on this ledger instead of editing A/B/C. |
| 3.1 Agent orchestration | Alignment acceptance gate | Section 3.1: accept a pose only when normalized median, p90, coverage, and silhouette all pass fixed thresholds. The numbers are not published. | Stage 8 estimates poses. It does not emit these four residuals. | partial | sceneagent.alignment |  | Feed real residuals in phase 3. Do not invent the thresholds. |
| 3.2 Input preparation | COLMAP camera poses | Section 3.2: RGB sequences use COLMAP for camera poses. | The A pipeline has no COLMAP stage. sceneagent.colmap_stage runs the COLMAP binary on a sampled RGB sequence. | partial | colmap | https://github.com/colmap/colmap | COLMAP 3.7 on the Texas box is built without CUDA, so SIFT ran on CPU. It registered 15/15 frames sampled at 1 fps from Fruits.mp4 (688 points, mean reprojection 0.901 px). The 16-frame cap is this runner's. Poses are not passed to splatfacto. |
| 3.2 Input preparation | splatfacto big | Section 3.2: Nerfstudio splatfacto big trains the 3DGS. | Optional stage 2c trains a background splat with nerfstudio when --bg-splat is set. Default video runs do not. | missing | ns-train splatfacto | https://github.com/nerfstudio-project/nerfstudio | Do not install the full splat stack on the 40 GB Texas disk. |
| 3.2 Input preparation | Photogrammetry and LiDAR | Section 3.2 also accepts photogrammetry, LiDAR, and existing 3DGS files. | Video and ZED only. | missing | photogrammetry or LiDAR asset |  | Accept an existing asset the way section 3.2 describes. The PDF names no importer repository. Polycam is a capture source on the project site, not code in this tree. |
| 3.2 Input preparation | Example video metadata | Section 3.2 starts from an RGB sequence. SimFoundry ships example videos the A pipeline already accepts. | docs/assets/example_videos, including PutMarkerInCup.mp4 and Fruits.mp4. | partial | sceneagent.video | https://github.com/NVlabs/SimFoundry | The mp4 header supplies duration. A 1 fps sample of Fruits.mp4 has COLMAP poses on the Texas box. Splats remain the splatfacto row. |
| 3.3 Object reconstruction | SAM masks | Section 3.3: SAM masks every tenth view by default. | Stage 5 uses its own decomposition stack, including SAM3 in the install docs, not this stride. | missing | segment_anything | https://github.com/facebookresearch/segment-anything | Wire masks into the association test after SAM is installed. |
| 3.3 Object reconstruction | GroundingDINO | The project website assigns semantic features with GroundingDINO and SAM. The paper text uses SAM plus CLIP and DINOv2 and does not name GroundingDINO in section 3.3. | Not a pipeline stage. | missing | groundingdino | https://github.com/IDEA-Research/GroundingDINO | Use it for the website's text prompts. Keep the paper's SAM stride as the section 3.3 path. |
| 3.3 Object reconstruction | OpenCLIP object vector | Section 3.3: OpenCLIP ViT-B-16 embeds each masked crop to 512 dimensions. It is a physics-prior input, not the Gaussian assignment. | Not a pipeline stage. | missing | open_clip | https://github.com/mlfoundations/open_clip | Embed crops only after masks exist. |
| 3.3 Object reconstruction | DINOv2 object vector | Section 3.3: DINOv2 ViT-S/14 is the other physics-prior feature. | The install requests facebook/dinov3 for other stages, which is not this ViT-S/14. | missing | dinov2 | https://github.com/facebookresearch/dinov2 | Do not substitute DINOv3 and call it the paper model. |
| 3.3 Object reconstruction | LangSplat-style codebook | Section 3.3 follows LangSplat's approach of attaching semantic features to Gaussians. It does not say the LangSplat repo is executed as-is. | No language-Gaussian codebook. | missing | LangSplat | https://github.com/minghanqin/LangSplat | Read the method. Do not vendor that research repo until the feature lift is specified. |
| 3.3 Object reconstruction | Gaussian association cuts | Section 3.3: centroid gate min(0.10, 0.6 e), containment 0.18 or Jaccard 0.08, front-surface eps_a 0.08 and eps_r 0.03, cell min(16, round(s/192)). | No equivalent cut. | done | sceneagent.association |  | Apply the cuts to real Gaussian centers once a splat exists. |
| 3.3 Object reconstruction | SAM3D meshes | Section 3.3: SAM3D generates object geometry. Trimmed similarity ICP starts from the SAM3D pose. | Stage 7 uses Hunyuan3D, TRELLIS.2, or Pixal3D, not SAM3D. | missing | SAM3D | https://github.com/facebookresearch/sam-3d-objects | Install on a large disk. Phase 1 box meshes are not SAM3D output. |
| 3.3 Object reconstruction | LaMa background infill | Section 3.3: optional LaMa texture inpainting after plane completion. | No LaMa stage. | missing | lama | https://github.com/advimman/lama | Optional after foreground Gaussians are removed. |
| 3.4 Physics prior | Material vote | Section 3.4 equation (2), with unknown-class weights scaled by 0.4. Ten classes, six body types, eight primitives. Soft body at deformability >= 0.65. Mass and density are log targets. | Stage 11 and the articulation physics file estimate mass and friction for OmniGibson. They do not use this vote. | done | sceneagent.physics |  | Add measured proposals when they exist. Leave modulus lookups null. |
| 3.4 Physics prior | Per-Gaussian MLP | Section 3.4: an MLP on DINOv2, OpenCLIP, and 16 geometric features, trained on VLM labels from GSO, YCB, and ABO. Weights are not released. | No such network. | missing | unreleased |  | Do not train a stand-in and call it their prior. |
| 3.4 Physics prior | VLM material proposal | Section 3.4: a VLM is one vote source and may audit the class. Sister edits in section 3.8 name Gemini 2.5 Flash. | Gemini is already the VLM for stages 3, 5, 6, 11 and pipeline B. | partial | google-genai |  | A video run may call Gemini 2.5 Flash on one frame. That call is not the full audit. |
| 3.4 Parts and articulation | PartSAM | Section 3.4: PartSAM decomposes each reconstructed object. | Stage 9 articulation has its own part segmentation. | missing | PartSAM | https://github.com/czvvd/PartSAM | Run only after a mesh exists. |
| 3.4 Parts and articulation | CubePart | Section 3.4: CubePart completes hidden geometry, such as a marker tip under the cap. | No hidden-part completion stage. | missing | CubePart | https://github.com/Roblox/cube | Do not fill missing geometry with a scaled box. |
| 3.4 Parts and articulation | Articulate Anything | Section 3.4: Articulate Anything proposes URDF joints. Agents review the motion. | Stage 9 can build articulations with --detect-articulation. It is not this VLM URDF system. | missing | Articulate Anything | https://github.com/vlongle/articulate-anything | Review joints in simulation before the ledger accepts them. |
| 3.5 Fracture | Warp, Blast, Newton | Section 3.5: parchment uses Warp cohesive fracture. Ceramics use NVIDIA Blast. Section 3.6 and the site use Newton for jelly and watermelon. These examples are not automatic export. | Rigid and articulated bodies only. | missing | warp / blast / newton | https://github.com/NVIDIA/warp | Author explicit examples later. Do not put fracture in the USD contract. |
| 3.6 4D captures | Dynamic IoU benchmark | Section 4.2 scores B_09, Z_06, and Z_12 from 4DCodeBench. Solvers named in section 3.6 are MPM and FLIP. | No 4D dynamic-scene program. | missing | 4DCodeBench |  | Do not report the paper's 99.09% dynamic IoU as our score. |
| 3.7 Export and editor | USD physics contract | Section 3.7: OpenUSD rigid body, mass, collision, friction, restitution, density. Deformables do not ride along in the file. | Stages 13 and 14 write USD and OmniGibson JSON for that stack. | partial | sceneagent package/usd_physics_contract.json | https://github.com/PixarAnimationStudios/OpenUSD | Phase 4 writes a .usd with the same fields. Phase 1 writes JSON only. |
| 3.7 Export and editor | Isaac Lab | Sections 3.1 and 3.7: the inspector reviews the scene in Isaac Lab. Experiments use Isaac Lab. | OmniGibson, not Isaac Lab. | missing | isaaclab | https://github.com/isaac-sim/IsaacLab | Inspector skill stays deferred until a scene loads there. |
| 3.7 Export and editor | MuJoCo | Section 3.7: exported scenes were also loaded in MuJoCo. | No MuJoCo export. | missing | mujoco | https://github.com/google-deepmind/mujoco | A load check comes after a real USD or MJCF file exists. |
| 3.7 Export and editor | Unreal Engine | Section 3.7 names Unreal as a third target. The engine is not a GitHub library we can vendor. | No Unreal export. | missing | Unreal Engine |  | Confirm a USD load in Unreal only after the file exists. |
| 3.7 Export and editor | Three.js viewer | Section 3.7 and figure 14: an interactive Three.js viewer places robots, crops the background, inserts assets, and edits pose and physics. | scripts/interactive/light_editor is already a Three.js editor for OmniGibson scene JSON. | partial | three.js | https://github.com/mrdoob/three.js | Extend the existing light editor. Do not replace it with a second viewer. |
| 3.7 Export and editor | Experiment grid page | Figure 14 shows sister-by-lighting place success. The site adds a 95% Wilson interval. | The light editor does not draw this matrix. | partial | sceneagent.report |  | The HTML grid is the figure 14 shape. It is not the Three.js scene editor. |
| 3.8 Digital sisters | Model-free mesh and texture | Section 3.8 first method: bounded mesh deformation plus noise, hue rotation, mottling, and speckle, checked for extent, volume, and grasp. | Pipeline B cousins are generated images and meshes, not this deformation. | done | sceneagent.sisters |  | The numeric grasp band is ours. The paper does not publish it. |
| 3.8 Digital sisters | Meshy generative sisters | Section 3.8 second method: Gemini 2.5 Flash, Gemini 3 Pro Image, and Meshy image-to-3D or Meshy-6 retexture. | Pipeline B is the cousin generator already in-tree, using Gemini and Hunyuan rather than Meshy. | missing | Meshy |  | No open repository matched Meshy-6. Do not substitute Hunyuan and label it Meshy. |
| 3.8 Digital sisters | Scene and task sisters | Figure 7 skills SCENEAGENT-SCENE-SISTERS and SCENEAGENT-TASK-SISTERS. Section 3.10 also randomizes placement. | Pipeline B samples scenes and writes task YAML with spatial predicates. | partial | sceneagent.sisters.scene_sisters |  | Templates and horizontal offsets only. Pipeline B remains the generative layout path. |
| 3.9 Hybrid scenes | Captured object in an artist scene | Section 3.9 places one converted object into an artist-authored environment. | The light editor can insert props. It does not build the site's kitchen, datacenter, or lab hybrids. | missing | light editor asset insert | https://github.com/NVlabs/SimFoundry | Use the existing editor after a captured mesh exists. Do not claim the hybrid videos were rebuilt. |
| 3.10 Policy | Scripted place check | Section 3.10: a scripted expert uses simulator state and keeps episodes whose object ends inside the target. | Pipeline C teleop, waypoints, and OmniGibson predicates. | partial | sceneagent.evaluation |  | Geometric reach-and-fit check only. It is not an Isaac or OmniGibson rollout. |
| 3.10 Policy | Wilson interval | The website reports a 95% Wilson interval on each ten-trial cell. Figure 14 reports the count. | No Wilson summary in pipeline C. | done | sceneagent.metrics.wilson_interval |  | Keep using it on counts this repo actually produces. |
| 3.10 Policy | pi0.5 fine-tune | Section 4.4 fine-tunes pi0.5 with LoRA on simulated demonstrations. | Pipeline C can eval a policy. It does not fine-tune pi0.5. | missing | pi0.5 | https://github.com/Physical-Intelligence/openpi | Do not copy the eggplant table into our results. |
| 4.1 Reconstruction metrics | F1 at 1 cm and Chamfer | Section 4.1 equations (4) and (5), tau = 1 cm, Hungarian match, unmatched F1 = 0. | The SimFoundry paper reports a different F1 protocol. | done | sceneagent.metrics |  | The functions are tested on synthetic clouds. GraspClutter6D was not scored. |
| 4.1 Reconstruction metrics | GraspClutter6D hard/000091 | Section 4.1 compares harnesses and SimFoundry on that bin scene. | Not run in this pass. | missing | GraspClutter6D |  | Score only after meshes exist. Their 60.75% and 6.9% figures stay theirs. |
| Official release | SceneAgent code | The PDF and the project site say the code, processed scenes, and example objects will be released with the paper. The PDF prints no git URL. | This tree is the SimFoundry fork plus the phase-1 package. It is not the Harvard release. | missing | unreleased |  | Keep building from the paper text. Do not substitute another repository for the unreleased code. |
