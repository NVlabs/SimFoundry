# SceneAgent on SimFoundry — design

This document is the upgrade plan for a new tree at `sceneagent-simfoundry`. The tree is the SimFoundry fork (`sandy-t-shelvan/SimFoundry`, parent `NVlabs/SimFoundry`, commit `9e34ebe`) plus a `sceneagent` package. SceneAgent, the Harvard paper and site, is the target behavior. It is not a second codebase in this tree. No other application repository is imported.

The row-by-row map is [`GAP_MATRIX.md`](GAP_MATRIX.md), generated from `sceneagent/catalog.py`. Current counts: 6 done, 7 partial, 24 missing, 37 rows. Done means a published rule is implemented in this package. Partial means the shape is here and the paper's tool is not. Missing means the named system has not been run.

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
