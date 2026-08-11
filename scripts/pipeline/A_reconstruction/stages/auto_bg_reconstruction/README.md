# Auto-Background Reconstruction

The auto-background flow adds a 3D Gaussian Splat background to an existing A reconstruction. It is useful when you want the final OmniGibson scene to include both reconstructed foreground objects and a realistic static background.

This is an optional add-on. Run the normal A reconstruction first.

## Quick Start

Step 1: run A reconstruction in splat-prep mode so foreground objects and background training share the same DA3 frame set.

```bash
OMNIGIBSON_HEADLESS=1 \
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name <scene> \
  --video-fpath /path/to/video.mov \
  --no-stream \
  -- s1_video.splat_prep=true \
     s1_video.n_subsampled_frames=400 \
     s1_video.target_w=672 \
     s1_video.target_h=384 \
     s5_scene.pda_geometric_backend=depth_pro \
     s10_sim.vlm_model=gemini-2.5-pro
```

Step 2: run the background add-on.

```bash
bash scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/run_auto_align.sh \
  <scene> /path/to/video.mov \
  --floor-category "desk, table, or counter"
```

The wrapper checks that the required A outputs exist and prints the expected Step 1 command if they do not.

## What It Runs

| Step | Script | Purpose | Main output |
|---|---|---|---|
| 1 | `1_generate_quadmask_for_void.py` | Build masks for foreground object removal. | `auto_bg/void/input/`, mask debug images |
| 2 | `2_run_void_pass1.py` | First VOID inpainting pass. | `auto_bg/void/pass1/` |
| 3 | `3_run_void_pass2.py` | Second VOID pass and cleaned frames. | `auto_bg/void/pass2/cleaned_frames/` |
| 4 | `4_build_seed_ply_from_void_da3.py` | Run DA3 on cleaned frames and build a seed point cloud. | `auto_bg/seed.ply` |
| 5 | `5_train_bg_splat.py` | Train a depth-supervised splat. | `auto_bg/splat/export/<scene>_bg.ply` |
| 6 | `6_bridge_bg_splat_to_og.py` | Align the splat to the OG world. | `<scene>_bg.ply.pose.json` |
| 7 | `7_build_og_scene_assets.py` | Build a loadable scene asset directory. | `assets/scenes/<scene>/` |

## Inputs

Required from the splat-prep A run:

- `Data/<scene>/s1_video/frames_subsampled_400/`
- `Data/<scene>/s1_video/input_video.mp4`
- `Data/<scene>/s2_da/da/exports/npz/results.npz`
- `Data/<scene>/s4_frame/image_0_cam2world.npy`
- `Data/<scene>/s13_og/reconstructed_og_scene.json`

Do not mix frames from one run with DA3 output from another run; the metric scale can differ.

## Outputs

```text
Data/<scene>/auto_bg/
  void/
  clean_frames -> void/pass2/cleaned_frames
  da3/void/da/exports/npz/results.npz
  seed.ply
  splat/export/<scene>_bg.ply
  splat/export/<scene>_bg.ply.pose.json

assets/scenes/<scene>/
  objects/
  <scene>_scene_state_auto_bg.json
```

## Setup Notes

Auto-background needs the standard SimFoundry setup plus:

- `void` environment for video inpainting
- `3dgrut` environment for PLY to USDZ conversion
- `nerfstudio_simfoundry` environment for splat training
- CUDA 12.x toolchain and a compatible host compiler for `gsplat`

Recommended pinned `nerfstudio_simfoundry` versions:

- Python 3.10
- `torch==2.1.2`
- `torchvision==0.16.2`
- `gsplat==1.4.0`
- `nerfstudio==1.1.5`
- `hydra-core>=1.3,<1.4` (required by the canonical Stage 2c entrypoint)

Create it manually:

```bash
mamba create -n nerfstudio_simfoundry python=3.10 -y
mamba run -n nerfstudio_simfoundry pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
mamba run -n nerfstudio_simfoundry pip install gsplat==1.4.0 nerfstudio==1.1.5 "hydra-core>=1.3,<1.4"
```

The depth-loss patch in `patches/splatfacto_depth_loss.patch` must be applied to the installed `nerfstudio` package if you use depth-supervised training.

## Practical Defaults

The defaults are tuned for a 24 GiB GPU:

- 400 frames
- 672x384 splat-prep frames
- DA3 resolution 448
- `splatfacto-big`
- 80k splat iterations
- depth loss enabled

Typical Step 2 wall time on a 24 GiB GPU is around two hours, plus the canonical A reconstruction.

## Troubleshooting

- Blurry splat: confirm `splatfacto-big` and camera optimizer `SO3xR3` are enabled.
- Floaters above a flat surface: confirm depth loss is enabled and try increasing `depth_loss_mult`.
- Ghost objects remain after VOID: increase guidance scale or choose a more accurate `--floor-category`.
- DA3 scale drift: refilm with smoother camera motion or increase frame overlap.
- `gsplat` compile errors: verify CUDA 12.x, a compatible compiler, and no unsupported old GPU architecture in `TORCH_CUDA_ARCH_LIST`.
