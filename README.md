# SimFoundry

SimFoundry (previously Controllable Digital Cousins) builds simulation-ready OmniGibson scenes from real video or ZED captures. The pipeline can reconstruct a scene, generate object-level "digital cousin" variations, and run evaluation or data-collection workflows on the resulting scene.

## Quick Start

1. Build the conda environments (this step will take a long time):

```bash
bash scripts/installation/install_everything.sh
```

2. Set up service access — the pipeline's VLM stages run on **Google Cloud Vertex AI (Gemini)**, and Hugging Face is required before downloading the gated VOID weights:

```bash
export GCLOUD_PROJECT=<your-gcp-project>   # or: bash scripts/installation/login_services.sh --gcloud
gcloud auth application-default login
huggingface-cli login
```

3. Download model checkpoints (needs the Hugging Face login from step 2):

```bash
bash scripts/installation/download_checkpoints.sh --default
```

> Already logged in to Hugging Face? You can fold step 3 into step 1 with
> `bash scripts/installation/install_everything.sh --checkpoints`.

All VLM stages — reconstruction (stages 3, 5, 6, 10), articulation (`--detect-articulation` / stage 8b), and the B augmentation pipeline — run on **Google Vertex AI (Gemini)**. Set `gcloud_project` in `scripts/cfg/real2sim_cfg.yaml` (or `export GCLOUD_PROJECT`) and authenticate with `login_services.sh --gcloud` or `gcloud auth application-default login`. Make sure the Gemini model IDs referenced in the configs are enabled in your GCP project and region.

More installation detail: [INSTALL.md](INSTALL.md)

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

Reconstruct a scene from video with a 24 GiB VRAM budget and automatic articulation:

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name pull_scene_2 \
  --video-fpath /path/to/video.mov \
  --max-vram-gb 24 \
  --detect-articulation
```

Note: all VLM stages run on Google Cloud (Vertex AI Gemini), so `gcloud_project` must be set — including for `--detect-articulation` (stage 8b) and the B augmentation pipeline below.

Generate bounded digital cousins, scene variants, and task proposals:

```bash
bash scripts/pipeline/B_augmentation/run.sh \
  --scene-name pull_scene_2 \
  --max-vram-gb 24 \
  -- prompt_cdc_structured.max_objects=2 \
       prompt_cdc_structured.max_generated_images_per_object=1
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
- `prompt_cdc_structured/`: cousin image proposals
- `sim_cousins/` and `usd_cousins/`: simulation-ready cousin assets
- `proposed_tasks/`: generated task YAMLs
- `application_smoke/`: C pipeline smoke-test videos

## Related Docs

- [INSTALL.md](INSTALL.md): installation and service setup
- [scripts/pipeline/README.md](scripts/pipeline/README.md): stage-by-stage pipeline reference
- [Auto-background README](scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/README.md): optional 3D Gaussian Splat background flow
