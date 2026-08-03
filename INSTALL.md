# Installation

This guide covers the standard SimFoundry setup: environments, checkpoints, service logins, and optional components.

## Requirements

- Linux with an NVIDIA GPU
- CUDA-compatible driver
- Mamba or Conda with `mamba`
- `ffmpeg`
- Git submodules enabled
- Hugging Face account for gated models such as SAM3
- Google Cloud project with the Vertex AI API and billing enabled — the pipeline's VLM stages (reconstruction, articulation, and B augmentation) run on Vertex AI (Gemini). Authenticate with `gcloud auth application-default login`
- ZED SDK only if you plan to use ZED capture

Recommended VRAM:

- 24 GiB works for the standard video pipeline. The streaming budget defaults to a fraction of
  total GPU memory (`stream_subseq.max_vram_frac`, 0.9), so no flag is needed; pass
  `--max-vram-gb N` only to pin an absolute cap.
- More VRAM can improve throughput for streamed reconstruction and high-resolution background runs.

## 1. Clone And Prepare Submodules

```bash
git submodule update --init --recursive
```

If a dependency is not a submodule on your checkout, use the matching install script in `scripts/installation/`.

## 2. Install Environments

The easiest path builds every pipeline conda env in one shot:

```bash
bash scripts/installation/install_everything.sh
```

This installs:

| Env | Purpose | Script |
|---|---|---|
| `simfoundry` | Main pipeline, VLM calls, image processing, OmniGibson tools. | `install_simfoundry.sh` |
| `hunyuan` | Hunyuan3D mesh generation. | `install_hunyuan.sh` |
| `any6d` | Pose estimation dependencies. | `install_any6d.sh` |
| `da3` | Depth Anything 3 inference. | `install_da3.sh` |
| `void` | VOID inpainting (auto-background). | `install_void.sh` |
| `nerfstudio_simfoundry` | Background splat train/export (auto-background). | `install_nerfstudio.sh` |
| `3dgrut` | PLY → USDZ (auto-background). | `install_3dgrut.sh` |

Build a subset with `--only`, e.g. just the core reconstruction envs:

```bash
bash scripts/installation/install_everything.sh --only "simfoundry hunyuan any6d da3"
```

Or install individually when debugging or customizing names:

```bash
cd scripts/installation
bash install_simfoundry.sh --project-root ../.. --env-name simfoundry --default
bash install_hunyuan.sh --project-root ../.. --env-name hunyuan --default
bash install_any6d.sh --project-root ../.. --env-name any6d --default
bash install_da3.sh --project-root ../.. --env-name da3 --default
```

Optional environments:

| Env | Purpose | Script |
|---|---|---|
| `3dgrut` | Convert Gaussian splats to USDZ for auto-background scenes. | `install_3dgrut.sh` |
| `articulate` | Articulation generation dependencies (stage 8b). | `install_articulate.sh` |
| `openpi` | OpenPI policy evaluation. | `install_openpi.sh` |

## 3. Log In To Services

The pipeline's VLM stages (reconstruction 3/5/6/10 and B augmentation) run on
**Google Cloud Vertex AI**. First, setup a [gcloud project](https://console.cloud.google.com/welcome/new) and then enable [Vertex AI](https://docs.vectorize.io/build-deploy/external-service-setup/how-to/google-vertex-ai/create-a-gcp-service-account-for-google-vertex-ai/).
Then, authenticate and set your project:

```bash
gcloud auth application-default login
```

Set the Google Cloud project in `scripts/cfg/real2sim_cfg.yaml` (or `export GCLOUD_PROJECT`):

```yaml
gcloud_project: <your-project-id>
```

Make sure the Gemini model IDs referenced in the configs are enabled in your project and region.

Log in to the other services (Hugging Face is required for gated model weights such as SAM3 and VOID):

```bash
bash scripts/installation/login_services.sh
```

Non-interactive login reads keys from a file:

```bash
cp scripts/installation/api_keys.template.txt scripts/installation/api_keys.txt
# Fill in scripts/installation/api_keys.txt (at minimum HF_TOKEN and GCLOUD_PROJECT). It is ignored by git.
bash scripts/installation/login_services.sh --default
```

Minimum service setup for the main (A reconstruction) pipeline:

```bash
export GCLOUD_PROJECT=<your-gcp-project>
gcloud auth application-default login
huggingface-cli login
```

## 4. Download Checkpoints

```bash
bash scripts/installation/download_checkpoints.sh --default
```

The script downloads model files used by FoundationStereo/FoundationPose, SAM2/SAM3-related tools, DepthPro, Hunyuan, VOID, and related dependencies where applicable.

If downloads are unreliable, provide a local fallback root:

```bash
bash scripts/installation/download_checkpoints.sh \
  --default \
  --checkpoint-fallback-root /path/to/known-good/repo-copy
```

### Robot assets

`install_simfoundry.sh` provisions OmniGibson robot assets in two steps, into
`deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets/`:

1. OmniGibson's own public download (`franka_panda`, `sky.jpg`, and the other stock robots).
2. The SimFoundry asset bundle from
   [`og_cdc_assets`](https://github.com/cremebrule/og_cdc_assets), which adds the
   `franka_robotiq` end effector used by most task configs. Its `models/` tree is merged over
   the public assets without overwriting them.

Override the source or pin a revision if needed:

```bash
OG_SIMFOUNDRY_ASSETS_REPO=git@github.com:cremebrule/og_cdc_assets.git \
OG_SIMFOUNDRY_ASSETS_COMMIT=<sha> \
  bash scripts/installation/install_simfoundry.sh --project-root ../.. --env-name simfoundry --default
```

If you already have a checkout containing the assets, point at it instead:

```bash
bash scripts/installation/install_simfoundry.sh \
  --project-root ../.. --env-name simfoundry --default \
  --robot-asset-fallback-root /path/to/repo-with-assets
```

## 5. Verify The Install

Basic environment checks:

```bash
mamba run -n simfoundry python -c "import torch, hydra, simfoundry; print('simfoundry ok')"
mamba run -n any6d    python -c "import torch, simfoundry; print('any6d ok')"
mamba run -n da3      python -c "import torch, simfoundry; print('da3 ok')"
mamba run -n hunyuan  python -c "import torch, simfoundry; print('hunyuan ok')"
```

Each should print a path inside *this* checkout. All four environments install the
`simfoundry` package editable, so a mismatch means a stale editable install is shadowing it.

Dry-run the pipeline wrappers (prints the stage plan; executes nothing):

```bash
bash scripts/pipeline/A_reconstruction/run.sh --dry-run --include 1b,2
bash scripts/pipeline/B_augmentation/run.sh --dry-run --include 1
bash scripts/pipeline/C_application/run.sh --dry-run --mode smoke-random
```

Run the test suite:

```bash
pip install -r requirements_dev.txt
mamba run -n simfoundry python -m pytest -q
```

A four-file subset needs no runtime dependencies at all, so it works before any environment
is built:

```bash
pytest tests/test_subpipeline_layout.py tests/test_resource_scheduler.py \
       tests/test_pipeline_reporting.py tests/test_pipeline_orchestrator.py
```

## 6. Run A Small Smoke Test

Use an existing video and scene name:

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name smoke_scene \
  --video-fpath /path/to/video.mov \
  --include 1b,2,3
```

Once a full A reconstruction exists, test scene loading:

```bash
bash scripts/pipeline/C_application/run.sh \
  --scene-name smoke_scene \
  --mode smoke-random
```

## Auto-Background Extras

The auto-background flow adds a 3D Gaussian Splat background to an existing reconstruction. It needs additional tooling:

- `void`
- `3dgrut`
- `nerfstudio_simfoundry`
- CUDA 12.x toolchain for `gsplat`

See [scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/README.md](scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/README.md).

## Articulation Dependencies

Articulation is optional. If these environments are not installed, `--detect-articulation` is
ignored with a warning and the rest of the reconstruction pipeline runs normally.

Articulation (stage 8b, `--detect-articulation`) is an optional component installed by:

```bash
bash scripts/installation/install_articulate.sh --default
```

It clones the SimFoundry articulate-anything fork from public GitHub and builds one
conda environment per segmentation backend: `articulate-anything-{hunyuan,partfield}`.

Requirements specific to articulation:

- **Source** — the fork is cloned from
  [`nadunRanawaka1/articulate-anything-sf`](https://github.com/nadunRanawaka1/articulate-anything-sf).
  The default segmentation backends (`Hunyuan3D-Part`, `PartField`) are fetched from their public
  upstreams and patched at install time (see `deps/articulate-anything/patches/`).
- **Git LFS** — the repos store large assets (embeddings, meshes) in Git LFS. The install
  script installs `git-lfs` automatically, but it must be present before cloning.
- **CUDA 12.8** at `/usr/local/cuda-12.8` (flash-attn / spconv build against it).
- **Vertex AI (Gemini)** — the articulation VLM calls (classifier + s2/s4/s5) run on Google Vertex AI;
  set `gcloud_project` (see section 3) and authenticate with `gcloud auth application-default login`.

Optional environment overrides (repo-relative defaults are used if unset):

| Variable | Purpose | Default |
|---|---|---|
| `GCLOUD_PROJECT` | GCP project for the Vertex AI (Gemini) VLM calls. | unset (set it, or `gcloud_project` in the config) |

P3-SAM weights auto-download on first use; SAM2 and PartField checkpoints are fetched by the
install script.

## Teleoperation Dependencies

Teleoperation (`scripts/pipeline/C_application` stages 2 / 2b) additionally requires
**TeleMoMa**, which SimFoundry does **not** install.

[TeleMoMa](https://github.com/UT-Austin-RobIn/telemoma) ships no license file and is
therefore all-rights-reserved. SimFoundry does not install, distribute, mirror, or
cache it, and grants no rights to it. The teleop stages import it lazily and raise an
actionable error if it is absent.

If you have separately established your own right to use TeleMoMa, install it yourself:

```bash
pip install --no-deps telemoma==0.3.0
```

The remaining teleop dependencies are installed normally from `requirements_teleop.txt`.

## Licence Terms Accepted During Installation

**The installer accepts NVIDIA and third-party terms on your behalf.** Running the
install scripts constitutes your acceptance of the following. Review them before
installing; if you do not accept them, do not run these scripts.

| Accepted by | Flag / variable | What it accepts |
|---|---|---|
| `install_simfoundry.sh:306`, `install_openpi.sh:78` | `--accept-nvidia-eula` | [NVIDIA Omniverse License Agreement](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html) — covers Isaac Sim, the Omniverse Kit runtime, the Kit USD libraries (`pxr`), and NuRec |
| `install_simfoundry.sh:306`, `install_openpi.sh:78` | `--accept-dataset-tos` | BEHAVIOR-1K / OmniGibson dataset terms |
| `install_simfoundry.sh:306`, `install_openpi.sh:78` | `--accept-conda-tos` | Anaconda / conda channel Terms of Service |
| `install_simfoundry.sh:509`, `reparent_usd_joints.py:21` | `OMNI_KIT_ACCEPT_EULA=YES` | NVIDIA Omniverse Kit EULA, set so Kit can start headless |

SimFoundry's own Apache 2.0 licence does **not** cover any of the above. The NVIDIA
platform components are listed in
[THIRD_PARTY_LICENSES.md §5a](THIRD_PARTY_LICENSES.md).

Separately, `scripts/installation/login_services.sh` can perform a `docker login` to
NVIDIA NGC (`nvcr.io`) using an API key you supply in `api_keys.txt`. That step is
optional, is skipped when no key is provided, and is governed by the
[NGC Terms of Use](https://ngc.nvidia.com/legal/terms).

## Optional Component Boundaries

SimFoundry's own source code is Apache 2.0. Several optional components it can fetch
are **not** — they are non-commercial, research-only, source-available, or unlicensed,
and their model weights frequently carry terms separate from their source code. None of
them are distributed in this repository or its release artifacts.

Components requiring your own review before use include SAM 3, Any6D, Hunyuan3D-2.1,
Hunyuan3D-Part, PartField, FoundationPose, FoundationStereo, nvdiffrast, cuRobo,
Depth Pro, VOID/CogVideoX weights, OpenPI/Gemma weights, CoTracker (CC-BY-NC-4.0),
and TeleMoMa (all-rights-reserved, user-supplied).

For each of these, the **component disclosure matrix** in
[THIRD_PARTY_LICENSES.md §6](THIRD_PARTY_LICENSES.md#6-component-disclosure-matrix--restricted-and-optional-components)
records whether it is required or optional, how it is acquired, the exact pinned
version, the separate terms covering its source and its model weights, and the
restriction that applies. Sections 1–5 of the same file give the per-component
license, copyright holder, and license link.

## Notes

- `api_keys.txt`, `Data/`, `deps/`, `reports/`, and local caches are ignored by git.
- Most scripts infer the repo root automatically; avoid hard-coding absolute paths in config unless the data really lives outside the repo.
- Use `--env-b1k simfoundry` on pipeline commands if OmniGibson is installed in the `simfoundry` environment rather than a separate `b1k` environment.
