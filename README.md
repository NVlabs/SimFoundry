<div align="center">

  <img src="docs/banner.png" alt="SimFoundry — Modular and Automated Scene Generation for Policy Learning and Evaluation" width="100%">

</div>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-76B900.svg)](LICENSE)
[![Project Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://research-staging.nvidia.com/labs/gear/simfoundry/)
[![Paper](https://img.shields.io/badge/arXiv-2606.28276-b31b1b.svg)](https://arxiv.org/abs/2606.28276)

</div>

---

# SimFoundry

SimFoundry builds simulation-ready scenes from real videos. The pipeline can reconstruct a scene, generate object-level "digital cousin" variations, and run evaluation or data-collection workflows on the resulting scene. The entire system is modular and allows users to pick which models to use according to their requirements and budget.

## Table of Contents

- [News](#news)
- [Quick Start](#quick-start)
- [Main Entrypoints](#main-entrypoints)
- [Common Examples](#common-examples)
- [Outputs](#outputs)
- [Related Docs](#related-docs)
- [License](#license)

## Release Tracker

| Date | Update |
|------|--------|
| **2026-08-13** | 🚀 Initial open-source release. V0 rigid-body and articulation generation |
| **Coming Soon** | Automated Background Generation |
| **Coming Soon** | Robotics Data Generation, Training and Evaluation |  

## Quick Start

1. Build the conda environments (this step will take a long time):

```bash
bash scripts/installation/install_everything.sh
```

2. Set up service access — the pipeline's VLM stages run on **Google Cloud Vertex AI (Gemini)**. First, setup a [gcloud project](https://console.cloud.google.com/welcome/new) and then enable [Vertex AI](https://docs.vectorize.io/build-deploy/external-service-setup/how-to/google-vertex-ai/create-a-gcp-service-account-for-google-vertex-ai/). A Hugging Face account is also required. Log in, then request access to these gated models (approval can take a while):

   - [facebook/sam3](https://huggingface.co/facebook/sam3)
   - [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
   - [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
   - Optional: [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)

```bash
export GCLOUD_PROJECT=<your-gcp-project>   # or: bash scripts/installation/login_services.sh
gcloud auth application-default login
huggingface-cli login
```

**Note:** if you would prefer to run the pipeline with a Gemini API key, generate one through [AI Studio](https://aistudio.google.com/api-keys) and then run `export GEMINI_API_KEY=<Your Key>`.

3. Download model checkpoints (needs the Hugging Face login and model access from step 2):

```bash
bash scripts/installation/download_checkpoints.sh --default
```

> Already logged in to Hugging Face? You can fold step 3 into step 1 with
> `bash scripts/installation/install_everything.sh --checkpoints`.

4. (Optional) Install the articulation pipeline:

```bash
bash scripts/installation/install_articulate.sh
```

All VLM stages — reconstruction (stages 3, 5, 6, 10) and the B augmentation pipeline — run on **Google Vertex AI (Gemini)**. Set `gcloud_project` in `scripts/cfg/real2sim_cfg.yaml` (or `export GCLOUD_PROJECT`) and authenticate with `login_services.sh` or `gcloud auth application-default login`. Make sure the Gemini model IDs referenced in the configs are enabled in your GCP project and region.

More installation detail: [INSTALL.md](docs/INSTALL.md)

## Main Entrypoints

The current pipeline entrypoints are:

```bash
scripts/pipeline/A_reconstruction/run.sh
scripts/pipeline/B_augmentation/run.sh
scripts/pipeline/C_application/run.sh
```

You can also use the dispatcher:

```bash
scripts/pipeline/run.sh A_reconstruction --help
scripts/pipeline/run.sh B_augmentation --help
scripts/pipeline/run.sh C_application --help
```

Pipeline reference: [scripts/pipeline/README.md](scripts/pipeline/README.md)

## Common Examples

Reconstruct a scene from video:

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name pull_scene_2 \
  --video-fpath /path/to/video.mov
```

The streamed stages budget VRAM as a fraction of the card (90% by default), so this works
unchanged on a 24 GiB or a 96 GiB GPU. Add `--max-vram-gb N` only to pin an absolute cap.

Note: all VLM stages run on Google Cloud (Vertex AI Gemini), so `gcloud_project` must be set — including for the B augmentation pipeline below.

Automatic articulation is available via `--detect-articulation` (stage 8b). It needs the optional
`articulate` environments — see [INSTALL.md](docs/INSTALL.md). If they are not installed, the flag is
ignored with a warning and the rest of the pipeline runs normally.

Generate bounded digital cousins, scene variants, and task proposals:

```bash
bash scripts/pipeline/B_augmentation/run.sh \
  --scene-name pull_scene_2 \
  -- prompt_cousin_structured.max_objects=2 \
       prompt_cousin_structured.max_generated_images_per_object=1
```

Smoke-test the reconstructed scene in OmniGibson:

```bash
bash scripts/pipeline/C_application/run.sh \
  --scene-name pull_scene_2 \
  --mode smoke-random
```

Run C with a generated task config:

```bash
bash scripts/pipeline/C_application/run.sh \
  --scene-name pull_scene_2 \
  --mode smoke-random \
  -- application_smoke.task_config=/path/to/task.yaml
```

## Outputs

Pipeline data is written under:

```text
Data/<scene_name>/
```

Important outputs include:

- `s13_og/reconstructed_og_scene.json`: final OmniGibson scene
- `s13_og/reconstructed_scene.png`: scene preview
- `prompt_cousin_structured/`: cousin image proposals
- `sim_cousins/` and `usd_cousins/`: simulation-ready cousin assets
- `proposed_tasks/`: generated task YAMLs
- `application_smoke/`: C pipeline smoke-test videos

## Related Docs

- [INSTALL.md](docs/INSTALL.md): installation and service setup
- [scripts/pipeline/README.md](scripts/pipeline/README.md): stage-by-stage pipeline reference
- [Auto-background README](scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/README.md): optional 3D Gaussian Splat background flow

## License

NVIDIA-owned SimFoundry source code is licensed under the
[Apache License 2.0](LICENSE).

Portions of SimFoundry are derived from the
[ACDC / digital-cousins](https://github.com/cremebrule/digital-cousins) project,
Copyright (c) 2024 the ACDC authors, also licensed under Apache 2.0. Files
containing derived code carry an attribution note in their header.

SimFoundry can optionally download or integrate third-party source code,
models, datasets, and SDKs governed by separate terms. The Apache 2.0
license does not apply to those materials. Several optional components are
non-commercial, research-only, or otherwise restricted.

See:

- [Third-Party Licenses](THIRD_PARTY_LICENSES.md)
- [Third-Party Notices](THIRD_PARTY_NOTICES.md)
- [Patch Provenance](PATCH_PROVENANCE.md)
- [Installation and optional component boundaries](docs/INSTALL.md)
