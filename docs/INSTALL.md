# Installation

This guide covers the standard SimFoundry setup: environments, checkpoints, service logins, and optional components.

## Requirements

- Linux with an NVIDIA GPU
- CUDA-compatible driver
- Mamba or Conda with `mamba`
- `ffmpeg`
- ~250 GB of free disk space for a full install (conda envs ≈ 100 GB, `deps/` ≈ 82 GB of
  which the VOID model alone is 41 GB, Hugging Face cache ≈ 12 GB, plus checkpoints)
- Hugging Face account for gated models such as SAM3, and DINOv3 for the optional `pixal3d` mesh backend
- Google Cloud project with the Vertex AI API and billing enabled — the pipeline's VLM stages (reconstruction, articulation, and B augmentation) run on Vertex AI (Gemini). Authenticate with `gcloud auth application-default login`
- ZED SDK only if you plan to use ZED capture

VRAM:

- **16 GiB is the minimum.** Stage 5's model stack peaks at ~14 GiB, which fits a 16 GiB
  card's budget (the budget defaults to 90% of total GPU memory via
  `stream_subseq.max_vram_frac`); on smaller cards the stage is rejected before it starts.
- On cards below ~29 GiB, pass `s7_mesh.low_vram=true` — the default (`false`) needs
  ~29 GiB for mesh shape generation at stage 7, while `true` drops it to ~6 GiB via CPU
  offloading. 24 GiB works for the standard video pipeline with that flag.
- More VRAM improves throughput for streamed reconstruction and high-resolution background
  runs. Pass `--max-vram-gb N` only to pin an absolute cap.
- The optional articulation stage (9) needs a minimum of 18 GiB — see
  [Articulation Dependencies](#articulation-dependencies).

## 1. Clone The Repository

This repo uses **Git LFS** for binary assets (PNGs, GIFs) in `docs/`. Install it before cloning:

```bash
# Install Git LFS (once per machine)
git lfs install

# Then clone normally — LFS files are fetched automatically
git clone https://github.com/NVlabs/SimFoundry.git
```

If you already cloned without LFS, fetch the assets with:

```bash
git lfs pull
```

The repository has no git submodules. All dependencies are cloned into `deps/` by the
install scripts in `scripts/installation/`.

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
| `articulate` | Articulation generation dependencies (stage 9). | `install_articulate.sh` |

Optional mesh-generation backends. Pass the env they live in to the pipeline with
`--env-mesh NAME`, and select the backend with `s7_mesh.shape_model` / `s7_mesh.texture_model`.
The two installers behave differently:

- `install_trellis.sh` installs **into** an existing env (default `simfoundry`) and pins shared
  packages there; if that env is missing it offers to create it via `install_simfoundry.sh --trellis`.
- `install_pixal3d.sh` defaults to a dedicated **`pixal3d`** env, requires it to already exist
  (build it with `install_trellis.sh --env-name pixal3d`), and refuses `simfoundry` outright
  unless `--allow-shared-env` is passed.

| Backend | Purpose | Script |
|---|---|---|
| `trellis2` | TRELLIS.2 mesh generation, an alternative to the default `hunyuan`. | `install_trellis.sh` |
| `pixal3d` | Pixal3D pixel-aligned mesh generation. Builds on the TRELLIS.2 stack, so run `install_trellis.sh` into the same env first. Pulls gated DINOv3 weights at runtime. | `install_pixal3d.sh` |

`pixal3d` generates shape and texture in a single pipeline, so it must be set as **both**
`shape_model` and `texture_model`. Its requirements conflict with the `simfoundry` env's pins,
so install it into a dedicated env:

```bash
cd scripts/installation
bash install_trellis.sh --project-root ../.. --env-name pixal3d
bash install_pixal3d.sh --project-root ../.. --env-name pixal3d
```

Then run stage 7 with it — but note that **stages 5-8 streaming is enabled by default and
relaunches stage 7 once per object**, which reloads Pixal3D, four DINOv3 encoders, MoGe-2 and
NAF every time. Until stage 7 gains a persistent worker, use `--include 7` or `--no-stream`:

```bash
# stage 7 only
scripts/pipeline/A_reconstruction/run.sh --env-mesh pixal3d --include 7 \
    s7_mesh.shape_model=pixal3d s7_mesh.texture_model=pixal3d s7_mesh.low_vram=true

# full pipeline, streaming off
scripts/pipeline/A_reconstruction/run.sh --env-mesh pixal3d --no-stream \
    s7_mesh.shape_model=pixal3d s7_mesh.texture_model=pixal3d s7_mesh.low_vram=true
```

`s7_mesh.low_vram=true` is **required on a 24 GiB card and is part of the supported profile**,
not an optimization. It selects resolution 1024. `s7_mesh.low_vram` defaults to `false`, which
selects resolution **1536 — measured to OOM on an RTX 4090** (24 GiB). Only override it on a
card with materially more memory.

Backend options go through `s7_mesh.generation_kwargs` (and `cousin_generation.generation_kwargs`).
Because that dict starts empty, adding a key from the CLI needs Hydra's `+` prefix:

```bash
scripts/pipeline/A_reconstruction/run.sh --env-mesh pixal3d --include 7 \
    s7_mesh.shape_model=pixal3d s7_mesh.texture_model=pixal3d \
    +s7_mesh.generation_kwargs.resolution=1024 \
    +s7_mesh.generation_kwargs.fov=0.2
```

(Set them in the YAML instead and plain `key=value` overrides work.) Keys the selected backend
cannot accept are dropped with a warning. `seed` defaults to the global `seed`.

Caveats, in rough order of how likely they are to bite:

- **Validated on exactly one object.** `trashcan_closed` completed stages 7 → 13 plus 9
  articulation on an RTX 4090 at `low_vram=true` / resolution 1024 (stage 7 ≈ 133 s; 281k verts,
  4096² PBR). Multi-object scenes, resolution 1536, and cousin generation are **unexercised** —
  treat `pixal3d` as experimental outside that envelope.
- **`--no-stream` (or `--include 7`) is a requirement, not a tuning tip.** Streaming relaunches
  stage 7 per object and reloads ~18 GiB of models each time.
- **Orientation is unvalidated.** Pixal3D generates in the input view's frame rather than a
  canonical one, so stage 8 pose matching needs checking before relying on it for a full scene.
- **Cousin generation (stage B/3) is experimental with this backend.** B only rescales the cousin
  and then reuses the original object's stage-8 pose, so a view-aligned mesh is not registered to
  it. Prefer `hunyuan`/`trellis2` for cousins until registration exists or same-view input is
  enforced.
- **RGBA inputs are required.** The alpha channel must already isolate the object (stage 6
  produces exactly this), because SimFoundry does not load Pixal3D's gated, non-commercial
  background-removal model. Other inputs are rejected with an explanatory error.
- **VRAM.** Measured **14.2 GiB peak** with `s7_mesh.low_vram=true` (resolution 1024) on an
  RTX 4090 — the supported profile. The `low_vram=false` default (resolution 1536) was also
  tested on the same card and **OOMs**, so 24 GiB users must set `low_vram=true`.
  Update `stream_subseq.stage_vram_gb[7]` if you use streaming anyway.

The `hunyuan` backend is installed by `install_hunyuan.sh` as part of the standard setup above.
The `direct3d` and `trellis` backend names are also registered but have no install script; you
must set them up yourself.

OpenPI policy evaluation needs no separate environment: the `openpi-client` package is
installed into the `simfoundry` env by `install_simfoundry.sh`.

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
hf auth login
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

A four-file subset needs no runtime dependencies beyond `pytest` itself, so it works before
any environment is built (a stock conda `base` does not ship `pytest` — install it first
with `pip install pytest` or `pip install -r requirements_dev.txt`):

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

Stage 2c runs directly in `nerfstudio_simfoundry` and requires Hydra there. The
current installer includes it; update an environment created by an older checkout with:

```bash
mamba run -n nerfstudio_simfoundry pip install "hydra-core>=1.3,<1.4"
```

See [scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/README.md](../scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/README.md).

## Articulation Dependencies

Articulation is optional. If these environments are not installed, `--detect-articulation` is
ignored with a warning and the rest of the reconstruction pipeline runs normally.

Articulation (stage 9, `--detect-articulation`) is an optional component installed by:

```bash
bash scripts/installation/install_articulate.sh --default
```

It clones the SimFoundry articulate-anything fork from public GitHub and builds one
conda environment per segmentation backend: `articulate-anything-{hunyuan,partfield}`.

Requirements specific to articulation:

- **VRAM** — articulation needs a minimum of 18 GiB of GPU memory. The 16 GiB minimum for
  the standard pipeline does not cover stage 9: the segmentation and part-decomposition
  models allocate outside the pipeline's VRAM scheduler.
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
| `install_simfoundry.sh` | `--accept-nvidia-eula` | [NVIDIA Omniverse License Agreement](https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html) — covers Isaac Sim, the Omniverse Kit runtime, the Kit USD libraries (`pxr`), and NuRec |
| `install_simfoundry.sh` | `--accept-dataset-tos` | BEHAVIOR-1K / OmniGibson dataset terms |
| `install_simfoundry.sh` | `--accept-conda-tos` | Anaconda / conda channel Terms of Service |
| `install_simfoundry.sh`, `reparent_usd_joints.py` | `OMNI_KIT_ACCEPT_EULA=YES` | NVIDIA Omniverse Kit EULA, set so Kit can start headless |

SimFoundry's own Apache 2.0 licence does **not** cover any of the above. The NVIDIA
platform components are listed in
[THIRD_PARTY_LICENSES.md §5a](../THIRD_PARTY_LICENSES.md).

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

In particular, the Tencent Hunyuan 3D 2.1 Community License grants rights only within
a Territory that **excludes the European Union, the United Kingdom, and South Korea**
— users in those places receive no rights from that license. A copy of the Agreement
ships at
[`third_party_notices/Hunyuan3D-2.1-LICENSE.txt`](../third_party_notices/Hunyuan3D-2.1-LICENSE.txt).

The optional `pixal3d` backend adds **DINOv3** (Meta, custom gated licence — you must accept its
terms before installing) and **NAF**, whose pinned checkout vendors DINOv3-licensed source that
executes on every generation. A machine or image with `pixal3d` installed therefore carries
DINOv3 terms; do not describe such a build as Apache-only or fully open source.

For each of these, the **component disclosure matrix** in
[THIRD_PARTY_LICENSES.md §6](../THIRD_PARTY_LICENSES.md#6-component-disclosure-matrix--restricted-and-optional-components)
records whether it is required or optional, how it is acquired, the exact pinned
version, the separate terms covering its source and its model weights, and the
restriction that applies. Sections 1–5 of the same file give the per-component
license, copyright holder, and license link.

## Notes

- `api_keys.txt`, `Data/`, `deps/`, `reports/`, and local caches are ignored by git.
- Most scripts infer the repo root automatically; avoid hard-coding absolute paths in config unless the data really lives outside the repo.
- OmniGibson stages run in the `simfoundry` environment by default. Pass `--env-b1k NAME` on pipeline commands only if you keep OmniGibson in a separate environment.
