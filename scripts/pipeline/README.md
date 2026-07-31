# Pipeline Reference

SimFoundry is organized into three runnable sub-pipelines:

```bash
scripts/pipeline/A_reconstruction/run.sh
scripts/pipeline/B_augmentation/run.sh
scripts/pipeline/C_application/run.sh
```

The dispatcher forwards to the same scripts:

```bash
scripts/pipeline/run.sh A_reconstruction ...
scripts/pipeline/run.sh B_augmentation ...
scripts/pipeline/run.sh C_application ...
```

All runners accept `--scene-name`, `--root-dir`, `--include`, `--exclude`, `--dry-run`, and Hydra-style overrides after `--`.

## A Reconstruction

Builds an OmniGibson scene from video or ZED stereo capture.

### Example

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name pull_scene_2 \
  --video-fpath /path/to/video.mov \
  --max-vram-gb 24
```

Useful options:

- `--pipeline video|stereo|zed`: input mode; default is `video`.
- `--stream / --no-stream`: stream stages 5-8 together or run them one at a time.
- `--max-vram-gb N`: hard VRAM budget for streamed stages.
- `--detect-articulation`: run stage 8b for automatic articulated-object generation. Not shipped in this release — the flag is accepted but ignored with a warning.
- `--env-b1k simfoundry`: use this if OmniGibson is installed in the `simfoundry` env.

### Stages

| ID | Script | Env | Purpose | Key outputs |
|---|---|---|---|---|
| `1a` | `A_reconstruction/stages/1a_take_stereo_images.py` | `simfoundry` | Capture ZED stereo images. | `s1_zed/` |
| `1b` | `A_reconstruction/stages/1b_process_raw_video.py` | `simfoundry` | Convert and sample a video. | `s1_video/frames_*`, `input_video.mp4` |
| `2` | `A_reconstruction/stages/2_run_depth.py` | `da3` or `simfoundry` | Run the selected depth backend. | `s2_da/` or `s2_fs/` |
| `3` | `A_reconstruction/stages/3_segment_ground_plane.py` | `simfoundry` | Find the support plane. | `s3_ground/` |
| `4` | `A_reconstruction/stages/4_unify_world_frame.py` | `simfoundry` | Align the scene to a stable world frame. | `s4_frame/` |
| `5` | `A_reconstruction/stages/5_decompose_scene.py` | `simfoundry` | Detect objects and create object-removal crops. | `s5_scene/` |
| `6` | `A_reconstruction/stages/6_upsample_object_images.py` | `simfoundry` | Create cleaner object images for mesh generation. | `s6_upsample/` |
| `7` | `A_reconstruction/stages/7_generate_object_meshes.py` | `hunyuan` | Generate 3D meshes. | `s7_mesh/` |
| `8` | `A_reconstruction/stages/8_match_object_poses.py` | `simfoundry` | Estimate object poses. | `s8_pose/` |
| `8b` | `A_reconstruction/stages/8b_articulate_objects.py` | `simfoundry` | Optional automatic articulation. **Not shipped in this release.** | `s8b_articulate_objects/` |
| `9` | `A_reconstruction/stages/9_compile_scene.py` | `simfoundry` | Compile object metadata. | `s9_compile/` |
| `10` | `A_reconstruction/stages/10_make_objects_sim_ready.py` | `simfoundry` | Build sim-ready URDF/collision assets. | `s10_sim/` |
| `11` | `A_reconstruction/stages/11_stabilize_physics.py` | `simfoundry` | Settle objects in physics. | `s11_physics/` |
| `12` | `A_reconstruction/stages/12_import_usd.py` | `simfoundry` | Import assets into USD datasets. | dataset USD assets |
| `13` | `A_reconstruction/stages/13_create_og_scene.py` | `simfoundry` | Create the final OG scene JSON and preview. | `s13_og/reconstructed_og_scene.json`, `reconstructed_scene.png` |

### Inputs And Outputs

Inputs:

- Video mode: `--video-fpath /path/to/video.mov`
- ZED mode: connected ZED camera and `--pipeline zed`

Final outputs:

- `Data/<scene>/s13_og/reconstructed_og_scene.json`
- `Data/<scene>/s13_og/reconstructed_scene.png`
- `Data/<scene>/s13_og/settled_poses.json`

## B Augmentation

Generates digital cousin assets, samples scene variants, and proposes tasks for a reconstructed scene.

### Example

```bash
bash scripts/pipeline/B_augmentation/run.sh \
  --scene-name pull_scene_2 \
  --max-vram-gb 24 \
  -- prompt_cousin_structured.max_objects=2 \
       prompt_cousin_structured.max_generated_images_per_object=1
```

Run only part of B:

```bash
bash scripts/pipeline/B_augmentation/run.sh --phases object-cousins
bash scripts/pipeline/B_augmentation/run.sh --phases scene-variations,task-generation
bash scripts/pipeline/B_augmentation/run.sh --include-p2p
```

### Stages

| ID | Script | Env | Purpose | Key outputs |
|---|---|---|---|---|
| `1` | `B_augmentation/stages/1_prompt_object_cousins.py` | `simfoundry` | Ask a VLM/image model for object cousin images. | `prompt_cousin_structured/` |
| `2` | `B_augmentation/stages/2_generate_cousin_combinations.py` | `simfoundry` | Choose which cousins to use together. | `cousins_combination/combinations.json` |
| `3` | `B_augmentation/stages/3_generate_cousin_meshes.py` | `hunyuan` | Generate textured cousin meshes. | `cousin_generation/` |
| `4` | `B_augmentation/stages/4_make_cousins_sim_ready.py` | `simfoundry` | Convert cousin meshes to sim-ready URDFs. | `sim_cousins/` |
| `5` | `B_augmentation/stages/5_import_cousin_usd.py` | `simfoundry` | Import cousin URDFs as USD assets. | `usd_cousins/`, custom asset dataset entries |
| `6` | `B_augmentation/stages/6_sample_reconstructed_scene.py` | `simfoundry` | Swap cousins into the reconstructed scene and sample variants. | `s13_og/auto_generation/` |
| `7` | `B_augmentation/stages/7_propose_scene_tasks.py` | `simfoundry` | Propose simple task YAMLs for the scene. | `proposed_tasks/*.yaml` |
| `8` | `B_augmentation/stages/8_match_cousin_p2p.py` | `simfoundry` | Optional point-to-point correspondence between base and cousin meshes. | `cousin_p2p_match/` |

### Inputs And Outputs

B expects a completed A reconstruction, especially:

- `Data/<scene>/s6_upsample/`
- `Data/<scene>/s13_og/reconstructed_og_scene.json`
- imported dataset assets from A Stage 12

Important B config keys in `scripts/cfg/real2sim_cfg.yaml`:

- `prompt_cousin_structured.*`
- `generate_cousins_combination.*`
- `cousin_generation.*`
- `sim.*`
- `usd.*`
- `propose_scene_task.*`
- `cousin_p2p_match.*`

The wrapper intentionally sets small local defaults for cousin count and task count. Override them after `--` for larger runs.

## C Application

Loads a reconstructed scene for smoke tests, policy evaluation, teleoperation, annotation, demo generation, and replay.

### Examples

```bash
bash scripts/pipeline/C_application/run.sh --scene-name pull_scene_2 --mode smoke-random
bash scripts/pipeline/C_application/run.sh --scene-name pull_scene_2 --mode eval
bash scripts/pipeline/C_application/run.sh --scene-name pull_scene_2 --mode demo
```

Use a generated task YAML in smoke mode:

```bash
bash scripts/pipeline/C_application/run.sh \
  --scene-name pull_scene_2 \
  --mode smoke-random \
  -- application_smoke.task_config=/path/to/task.yaml
```

### Modes And Stages

| Mode / ID | Script | Purpose | Key outputs |
|---|---|---|---|
| `smoke` | `C_application/stages/0_smoke_random_actions.py` | Headless random-action load/step test. | `application_smoke/*.mp4` |
| `1` | `C_application/stages/1_eval_policy_og_scene.py` | Evaluate a policy in the OG scene. | `s15_eval/` |
| `2` | `C_application/stages/2_teleop_og_scene.py` | Collect teleop demonstrations. | `s14_teleop/` |
| `3` | `C_application/stages/3_annotate_src_demo.py` | Annotate source demos. | `s15_annotation/` |
| `3b` | `C_application/stages/3b_modify_annotations.py` | Adjust annotations. | `s15_annotation/` |
| `4` | `C_application/stages/4_extract_waypoints.py` | Extract object-centric waypoints. | `s16_waypoints/` |
| `5` | `C_application/stages/5_generate_demos.py` | Generate demos from waypoints. | `s17_generated_demos/` |
| `6` | `C_application/stages/6_replay_dataset.py` | Replay generated demos. | `s18_replay/` |

Mode mapping:

- `--mode smoke-random`: stage `smoke`
- `--mode eval`: stage `1`
- `--mode demo`: stages `2,3,3b,4,5,6`
- `--mode full`: stages `1,2,3,3b,4,5,6`

## Cache And Test Mode

A and B can cache raw remote model responses:

```bash
--cache-mode --model-cache-dir .cache/simfoundry/model_calls
```

Replay cached responses:

```bash
--test-mode --model-cache-dir .cache/simfoundry/model_calls
```

Use this for reproducible debugging and CI-like checks when remote model calls are expensive or unstable.
