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
  --video-fpath /path/to/video.mov
```

Example captures are in `docs/assets/example_videos/` (repo root), e.g.
`--video-fpath docs/assets/example_videos/Fruits.mp4`. Shooting your own? See
[Capturing an input video](#capturing-an-input-video).

Useful options:

- `--pipeline video|stereo|zed`: input mode; default is `video`.
- `--stream / --no-stream`: stream stages 5-8 together or run them one at a time.
- `--max-vram-frac F`: VRAM budget for streamed stages as a fraction of total GPU memory. Default `0.9`, so the same setting works across card sizes.
- `--max-vram-gb N`: opt-in absolute hard budget in GiB, overriding the fraction. Leave unset unless you need to pin it — with `hard_vram_cap` the budget counts *total* GPU usage, so a value too small for the card stalls stages.
- `--detect-articulation`: run stage 9 for automatic articulated-object generation. Requires the optional `articulate` environments.
- Stage 9 has two optional interactive checkpoints, enabled as Hydra overrides (after `--`):
  - `s9_articulate_objects.merge_interactive_correction=true` opens a browser UI to review and
    correct the mesh segmentation before joints are estimated.
  - `s9_articulate_objects.interactive_joint_refinement=true` opens a browser UI after
    articulation to refine each joint — type, axis, pivot, limits, and physics (per-joint
    damping/friction, per-part mass and surface friction) — with a live 3D motion preview.
    Saves rewrite `results/mobility.urdf` (the original is backed up) and user physics edits
    land in `physics_overrides.json`.

  Both UIs open in a browser tab and block the pipeline until you finish (or cancel), so use
  them for attended runs only. 
- `--env-nerfstudio NAME`: select the Nerfstudio environment used by stage 2c. Default: `nerfstudio_simfoundry`.
- `--env-b1k NAME`: env for the OmniGibson stages. Default: `simfoundry`; pass this only if OmniGibson lives in a separate env.

### Capturing an input video

Keep these in mind when taking an input video:

1. **One flat surface.** Place every object on the same flat surface (a desk, a counter).
   Keep other surfaces — a second table, shelves — out of frame: stage 3 fits a single
   support plane.
2. **Angle down at the surface.** Shoot at an oblique downward angle — not edge-on from
   the side, not straight top-down — so both the surface and the sides of the objects
   are visible.
3. **Translate, don't rotate.** Sweep the camera sideways around the scene. Panning in
   place gives the reconstruction no parallax.
4. **Keep every object visible in every frame.** An object that leaves the frame can come
   out missing or mis-posed.
5. **Close all articulated objects.** Make sure all objects with joints (trash cans, cabinets) are closed.

> See [`example_videos/`](../../docs/assets/example_videos)

<img src="../../docs/assets/capture_lateral_sweep.png" alt="Start, middle, and end frames of an example capture: one table, camera angled down, sweeping laterally" width="100%">

*Start, middle, and end of `docs/assets/example_videos/Fruits.mp4`: the
camera stays angled down at one table, translates laterally, and keeps every object in
frame throughout.*

### Stages

| ID | Script | Env | Purpose | Key outputs |
|---|---|---|---|---|
| `1a` | `A_reconstruction/stages/1a_take_stereo_images.py` | `simfoundry` | Capture ZED stereo images. | `s1_zed/` |
| `1b` | `A_reconstruction/stages/1b_process_raw_video.py` | `simfoundry` | Convert and sample a video. | `s1_video/frames_*`, `input_video.mp4` |
| `2c` | `A_reconstruction/stages/2c_train_bg_splat.py` | `nerfstudio_simfoundry` (`--env-nerfstudio`) | Train and export the background Gaussian splat. | `s2c_gs/` |
| `2` | `A_reconstruction/stages/2_run_depth.py` | `da3` or `simfoundry` | Run the selected depth backend. | `s2_da/` or `s2_fs/` |
| `3` | `A_reconstruction/stages/3_segment_ground_plane.py` | `simfoundry` | Pick the canonical frame and find the support plane. | `s3_ground/` |
| `4` | `A_reconstruction/stages/4_unify_world_frame.py` | `simfoundry` | Align the scene to a stable world frame. | `s4_frame/` |
| `5` | `A_reconstruction/stages/5_decompose_scene.py` | `simfoundry` | Detect objects and create object-removal crops. | `s5_scene/` |
| `6` | `A_reconstruction/stages/6_upsample_object_images.py` | `simfoundry` | Create cleaner object images for mesh generation. | `s6_upsample/` |
| `7` | `A_reconstruction/stages/7_generate_object_meshes.py` | `hunyuan` (`--env-mesh`) | Generate 3D meshes. | `s7_mesh/` |
| `8` | `A_reconstruction/stages/8_match_object_poses.py` | `simfoundry` | Estimate object poses. | `s8_pose/` |
| `9` | `A_reconstruction/stages/9_articulate_objects.py` | `simfoundry` | Optional automatic articulation (`--detect-articulation`). Also the source of articulated-object physics: the workflow estimates per-part mass/surface friction and per-joint damping/friction (`results/physics_properties.json` + URDF `<dynamics>`), which stage 11 consumes instead of estimating its own; user edits from the refinement UI (`physics_overrides.json`) take precedence. | `s9_articulate_objects/` |
| `10` | `A_reconstruction/stages/10_compile_scene.py` | `simfoundry` | Compile object metadata. | `s10_compile/` |
| `11` | `A_reconstruction/stages/11_make_objects_sim_ready.py` | `simfoundry` | Build sim-ready URDF/collision assets. | `s11_sim/` |
| `12` | `A_reconstruction/stages/12_stabilize_physics.py` | `simfoundry` | Settle objects in physics. | `s12_physics/` |
| `13` | `A_reconstruction/stages/13_import_usd.py` | `simfoundry` | Import assets into USD datasets. | dataset USD assets |
| `14` | `A_reconstruction/stages/14_create_og_scene.py` | `simfoundry` | Create the final OG scene JSON and preview. | `s14_og/reconstructed_og_scene.json`, `reconstructed_scene.png` |

### Mesh Generators

Stage 7 supports three image-to-3D backends, selected with `s7_mesh.shape_model` and
`s7_mesh.texture_model` (set both to the same value):

- **Hunyuan3D-2.1** (`hunyuan`, the default). Runs in the `hunyuan` env built by
  `install_everything.sh` and needs the RealESRGAN checkpoint from
  `download_checkpoints.sh`. On cards below ~29 GB VRAM, set `s7_mesh.low_vram=true`
  (CPU offloading, ~6 GB). This is the default `run.sh` invocation — nothing to pass.

- **TRELLIS.2** (`trellis2`). Opt-in: install it with
  `bash scripts/installation/install_trellis.sh` (or `install_simfoundry.sh --trellis`),
  which adds it to the `simfoundry` env — so stage 7 must also be pointed at that env.
  Model weights (`microsoft/TRELLIS.2-4B`) download automatically on the first run.

  ```bash
  bash scripts/pipeline/A_reconstruction/run.sh --scene-name <scene> --video-fpath <video> \
    --env-mesh simfoundry \
    -- s7_mesh.shape_model=trellis2 s7_mesh.texture_model=trellis2
  ```

- **Pixal3D** (`pixal3d`). Opt-in, pixel-aligned generation built on the TRELLIS.2
  backbone. It needs its own env (its pins conflict with the shared `simfoundry` env):
  build the TRELLIS.2 stack into a dedicated env first, then install Pixal3D on top —

  ```bash
  bash scripts/installation/install_trellis.sh --env-name pixal3d
  bash scripts/installation/install_pixal3d.sh --env-name pixal3d
  ```
  
  Then run stage 7 with:

  ```bash
  bash scripts/pipeline/A_reconstruction/run.sh --scene-name <scene> --video-fpath <video> \
    --env-mesh pixal3d \
    -- s7_mesh.shape_model=pixal3d s7_mesh.texture_model=pixal3d
  ```

All three write to the same layout (`s7_mesh/textured_mesh/<backend>/iter_N_mesh.glb`), so
downstream stages need no other changes. To compare backends on one capture, run each into
a separate `--scene-name`.

### Canonical Frame Selection

Stages 3-14 reconstruct the scene from a *single* frame of the capture: stage 3 fits the
support plane in it, stage 4 makes its camera the world frame, and stage 5 crops every object
out of it for stages 6-8. A bad frame therefore caps the quality of everything downstream — a
blurry frame produces blurry meshes, and a frame shot from far away leaves small objects at too
few pixels to reconstruct.

By default `s3_ground.img_idx: auto`, so stage 3 scores the candidate frames and picks one.

`frame_selection.mode` controls the last step:

- `heuristic`: scoring only, no remote calls.
- `hybrid` (default): the heuristic short-lists `vlm_top_k` frames and a VLM makes the final
  call.
- `vlm`: the VLM chooses among every frame that passes the gates.

Pin a frame instead  with an integer:

```bash
bash scripts/pipeline/A_reconstruction/run.sh --scene-name pull_scene_2 \
  -- s3_ground.img_idx=4
```

If every candidate is rejected, stage 3 fails with the per-frame reasons; loosen the gate it
names or pin a frame. Changing the selection after a run invalidates the `image_<idx>_*`
artifacts stages 4-14 wrote for the previous frame, so rerun from stage 3.

### Inputs And Outputs

Inputs:

- Video mode: `--video-fpath /path/to/video.mov`
- ZED mode: connected ZED camera and `--pipeline zed`

Final outputs:

- `Data/<scene>/s14_og/reconstructed_og_scene.json`
- `Data/<scene>/s14_og/reconstructed_scene.png`
- `Data/<scene>/s14_og/settled_poses.json`

## B Augmentation

Generates digital cousin assets, samples scene variants, and proposes tasks for a reconstructed scene.

### Example

```bash
bash scripts/pipeline/B_augmentation/run.sh \
  --scene-name pull_scene_2 \
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
| `3` | `B_augmentation/stages/3_generate_cousin_meshes.py` | `hunyuan` (`--env-mesh`) | Generate textured cousin meshes. | `cousin_generation/` |
| `4` | `B_augmentation/stages/4_make_cousins_sim_ready.py` | `simfoundry` | Convert cousin meshes to sim-ready URDFs. | `sim_cousins/` |
| `5` | `B_augmentation/stages/5_import_cousin_usd.py` | `simfoundry` | Import cousin URDFs as USD assets. | `usd_cousins/`, custom asset dataset entries |
| `6` | `B_augmentation/stages/6_sample_reconstructed_scene.py` | `simfoundry` | Swap cousins into the reconstructed scene and sample variants. | `s14_og/auto_generation/` |
| `7` | `B_augmentation/stages/7_propose_scene_tasks.py` | `simfoundry` | Propose simple task YAMLs for the scene. | `proposed_tasks/*.yaml` |
| `8` | `B_augmentation/stages/8_match_cousin_p2p.py` | `simfoundry` | Optional point-to-point correspondence between base and cousin meshes. | `cousin_p2p_match/` |

### Inputs And Outputs

B expects a completed A reconstruction, especially:

- `Data/<scene>/s6_upsample/`
- `Data/<scene>/s14_og/reconstructed_og_scene.json`
- imported dataset assets from A Stage 13

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

### Task Predicates

Task YAMLs (see `scripts/cfg/task/example.yaml` for a fully commented example) use two
predicate vocabularies:

**Spatial placement predicates** (`group_predicate_placement`, applied at each episode
reset; implemented in `simfoundry/utils/placement_utils.py`): `on_top`, `left_of`,
`right_of`, `behind`, `in_front_of`, `inside`, `near`, `between`, `inside_link`.
The horizontal predicates displace one axis by `gap` and sample the other; `near`
places at a random direction around the reference; `between` places on the segment
between two reference groups (`reference_groups: [a, b]`); `inside_link` places
within a named link's AABB (`link_name`, e.g. a shelf level or drawer).

```yaml
group_predicate_placement:
  cup:
    reference_group: plate
    predicates: [left_of, near]
    gap: {left_of: [0.03, 0.15], near: [0.02, 0.10]}
```

**Check predicates** (`init_predicates_*`, `goal_predicates_*`, `milestone_predicates`):
any OmniGibson object state (`OnTop`, `Touching`, ...) plus the SimFoundry special
states — `PlaceOnTop` (init-only AABB placement), `InsideAABB` (containment,
`volume_threshold`), `OnTopAABB` (`z_tolerance`, `xy_overlap_threshold`), `AboveAABB`
(`min_clearance`, optional `xy_overlap_threshold`), `Lifted` (unary, `min_height`
above the episode-start pose), and `IsGrasping` (milestones only). The AABB-based
states are recommended for reconstructed/custom assets, whose collision meshes often
break OmniGibson's sampling-based states.

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
