<div align="center">

  <img src="docs/pull_figure.png" alt="SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation" width="100%">

</div>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-76B900.svg)](LICENSE)
[![Project Website](https://img.shields.io/badge/Project-Website-blue.svg)](https://research.nvidia.com/labs/gear/simfoundry/)
[![Paper](https://img.shields.io/badge/arXiv-2606.28276-b31b1b.svg)](https://arxiv.org/abs/2606.28276)

</div>

---

# SimFoundry

SimFoundry turns a short real-world video into a physics-ready simulation scene in under an hour, with no manual annotation required. Point it at a tabletop, and it automatically segments every object, reconstructs geometry, generates textured 3D meshes, and compiles the result into an OmniGibson scene complete with physical parameters, digital cousin variations, and task proposals.

Unlike prior scene reconstruction approaches, SimFoundry is fully modular: each stage is an independently swappable component. As foundation models improve, the SimFoundry pipeline improves with them.

## News

| Date | Update |
|------|--------|
| **2026-08-14** | 🚀 Initial open-source release: V0 rigid-body and articulation generation |
| **2026-08-26** | 📦 Example scenes and assets published — see [Example Scenes & Assets](#example-scenes--assets) |
| **Coming Soon** | Automated background generation |
| **Coming Soon** | Robotics data generation, training, and evaluation |

## Table of Contents

- [Quick Start](#quick-start)
- [Common Examples](#common-examples)
- [Pipeline Overview](#pipeline-overview)
- [Scene Gallery](#scene-gallery)
- [Digital Cousins](#digital-cousins)
- [Scene Editors](#scene-editors)
- [Sim-to-Real Policy Training](#sim-to-real-policy-training)
- [Example Scenes & Assets](#example-scenes--assets)
- [Outputs](#outputs)
- [What's Included](#whats-included)
- [Documentation](#documentation)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)

## Requirements

- Linux with an NVIDIA GPU and CUDA
- Mamba or Conda with `mamba`
- `ffmpeg`
- ~250 GB of free disk space for a full install
- Hugging Face account
- Google Cloud project or Gemini API key

## Quick Start

**1.** Build the conda environments (takes a while) or ask your agent to do with [AGENT_INSTALL.md](docs/AGENT_INSTALL.md):

```bash
bash scripts/installation/install_everything.sh
```

**2.** Set up service access. Request access to these gated Hugging Face models (approval can take time):

- [facebook/sam3](https://huggingface.co/facebook/sam3)
- [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
- Optional: [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)

VLM stages run on **Google Cloud Vertex AI (Gemini)**. Set up a [gcloud project](https://console.cloud.google.com/welcome/new) with [Vertex AI enabled](https://docs.vectorize.io/build-deploy/external-service-setup/how-to/google-vertex-ai/create-a-gcp-service-account-for-google-vertex-ai/), then authenticate:

```bash
export GCLOUD_PROJECT=<your-gcp-project>
gcloud auth application-default login
hf auth login
```

> **No GCP project?** Generate a Gemini API key at [AI Studio](https://aistudio.google.com/api-keys) and run `export GEMINI_API_KEY=<your-key>` instead.

Alternatively, run the interactive login helper which covers all services at once:

```bash
bash scripts/installation/login_services.sh
```

**3.** Download model checkpoints:

```bash
bash scripts/installation/download_checkpoints.sh --default
```

> Already logged in to Hugging Face? Fold this into step 1 with `bash scripts/installation/install_everything.sh --checkpoints`.

**4.** (Optional) Install the articulation pipeline:

```bash
bash scripts/installation/install_articulate.sh
```


Full installation details: [INSTALL.md](docs/INSTALL.md)

## Common Examples

Reconstruct a scene from video (example inputs in [`docs/assets/example_videos/`](docs/assets/example_videos),
capture tips in [the pipeline README](scripts/pipeline/README.md#capturing-an-input-video)):

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name my_scene \
  --video-fpath /path/to/video.mov
```

The streamed stages budget VRAM as a fraction of the card (90% by default), so this works
unchanged on a 24 GiB or a 96 GiB GPU. Add `--max-vram-gb N` only to pin an absolute cap.
On a 24 GiB card, also pass `-- s7_mesh.low_vram=true` — the default needs ~29 GiB for mesh
shape generation at stage 7.

Enable automatic articulation decomposition (requires the optional `articulate` environments — see [INSTALL.md](docs/INSTALL.md)):

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name my_scene \
  --video-fpath /path/to/video.mov \
  --detect-articulation
```

Generate digital cousins, scene variants, and task proposals:

```bash
bash scripts/pipeline/B_augmentation/run.sh \
  --scene-name my_scene \
  -- prompt_cousin_structured.max_objects=2 \
       prompt_cousin_structured.max_generated_images_per_object=1
```

Smoke-test the reconstructed scene in OmniGibson:

```bash
bash scripts/pipeline/C_application/run.sh \
  --scene-name my_scene \
  --mode smoke-random
```

You can also use the unified dispatcher:

```bash
scripts/pipeline/run.sh A_reconstruction --help
scripts/pipeline/run.sh B_augmentation --help
scripts/pipeline/run.sh C_application --help
```

## Pipeline Overview

<div align="center">
  <img src="docs/pipeline_figure.png" alt="SimFoundry Pipeline" width="100%">
</div>

SimFoundry extracts per-object relevant information (segmentation masks, depth, etc.), generates 3D visual meshes via 2D-to-3D generation models, and compiles the final output scene by annotating relevant physical parameters and sanity checking the overall scene configuration in a physics simulator. SimFoundry additionally supports diverse simulated augmentations of objects, scenes, and tasks. SimFoundry's modular design ensures that as individual foundation models improve, the SimFoundry pipeline improves with them.

SimFoundry is organized into three modular pipelines:

| Pipeline | Description |
|---|---|
| **A: Reconstruction** | Reconstructs a simulation-ready scene from a real video across 13 stages: video processing, depth estimation, ground segmentation, object decomposition, mesh generation, pose estimation, physics compilation, and USD/OmniGibson export. |
| **B: Augmentation** | Generates digital cousin variations of the reconstructed objects spanning geometry, topology, and visual appearance, and proposes manipulation tasks for each scene. |
| **C: Application** | Loads the scene into OmniGibson for robot policy evaluation, teleoperation data collection, and pipeline smoke testing. |

## Scene Gallery

Each object in the simulation column was generated fully automatically from a single 2D crop using Hunyuan3D, compiled into a physics-ready simulation scene. 
> Try it for yourself with [Pipeline A](scripts/pipeline/README.md#a-reconstruction).

<table>
  <tr>
    <th align="center">Real World</th>
    <th align="center">SimFoundry Reconstruction</th>
  </tr>
  <tr>
    <td align="center"><img src="docs/gallery/bathroom_1_real.png" width="360" alt="Bathroom — Real"><br><sub>Bathroom</sub></td>
    <td align="center"><img src="docs/gallery/bathroom_1_seq.gif" width="360" alt="Bathroom — SimFoundry"><br><sub>Bathroom Digital Twin</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/gallery/dining_1_real.png" width="360" alt="Dining Room — Real"><br><sub>Dining Room</sub></td>
    <td align="center"><img src="docs/gallery/dining_1_seq.gif" width="360" alt="Dining Room — SimFoundry"><br><sub>Dining Room Digital Twin</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/gallery/home_coffee_4_real.png" width="360" alt="Home Coffee Table — Real"><br><sub>Home Coffee Table</sub></td>
    <td align="center"><img src="docs/gallery/home_coffee_4_seq.gif" width="360" alt="Home Coffee Table — SimFoundry"><br><sub>Home Coffee Table Digital Twin</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/gallery/kitchen_2_real.png" width="360" alt="Kitchen — Real"><br><sub>Kitchen</sub></td>
    <td align="center"><img src="docs/gallery/kitchen_2_seq.gif" width="360" alt="Kitchen — SimFoundry"><br><sub>Kitchen Digital Twin</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/gallery/outdoor_1_real.png" width="360" alt="Outdoor — Real"><br><sub>Outdoor</sub></td>
    <td align="center"><img src="docs/gallery/outdoor_1_seq.gif" width="360" alt="Outdoor — SimFoundry"><br><sub>Outdoor Digital Twin</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/gallery/toys_1_real.png" width="360" alt="Toys — Real"><br><sub>Toys</sub></td>
    <td align="center"><img src="docs/gallery/toys_1_seq.gif" width="360" alt="Toys — SimFoundry"><br><sub>Toys Digital Twin</sub></td>
  </tr>
</table>

## Digital Cousins

Given a reconstructed scene, SimFoundry uses a VLM to propose geometry, topology, and appearance variations of each object, then generates the resulting 3D assets automatically. 

> This functionality is powered by [Pipeline B](scripts/pipeline/README.md#b-augmentation).


<table>
  <tr>
    <th align="center">Dining Room</th>
    <th align="center">Home Coffee</th>
    <th align="center">Toys</th>
  </tr>
  <tr>
    <td align="center" valign="top"><b>Digital Twin</b><br><img src="docs/gallery/dining_1_twin.png" width="240" alt="Dining Room — Digital Twin"></td>
    <td align="center" valign="top"><b>Digital Twin</b><br><img src="docs/gallery/home_coffee_4_twin.png" width="240" alt="Home Coffee — Digital Twin"></td>
    <td align="center" valign="top"><b>Digital Twin</b><br><img src="docs/gallery/toys_1_twin.png" width="240" alt="Toys — Digital Twin"></td>
  </tr>
  <tr>
    <td align="center" valign="top"><b>Digital Cousins</b><br><img src="docs/gallery/dining_1_cousin_static.png" width="240" alt="Dining Room — Digital Cousins"></td>
    <td align="center" valign="top"><b>Digital Cousins</b><br><img src="docs/gallery/home_coffee_4_cousin_static.png" width="240" alt="Home Coffee — Digital Cousins"></td>
    <td align="center" valign="top"><b>Digital Cousins</b><br><img src="docs/gallery/toys_1_cousin_static.png" width="240" alt="Toys — Digital Cousins"></td>
  </tr>
</table>

## Scene Editors

Two editors work with OmniGibson scene JSON — either a scene the editor saved
(`*_scene_state_*.json`) or Pipeline A's own output (`s13_og/reconstructed_og_scene.json`):

- **Light editor** — a browser-based scene editor (no OmniGibson required) for composing
  scenes, placing and scaling props, editing camera rigs, associating tasks, and exporting a
  runnable bundle:

  ```bash
  bash scripts/installation/install_light_editor.sh
  python scripts/interactive/light_editor/server.py --scene <scene_state.json>
  ```

  Open <http://localhost:8770>, drag a prop (or press `M`/`R` to move/rotate it), then click
  **Save scene JSON**. Full guide: [docs/INSTRUCTIONS_SCENE_EDITOR.md](docs/INSTRUCTIONS_SCENE_EDITOR.md).
- **Interactive scene editor** — the OmniGibson-based editor
  (`scripts/interactive/interactive_scene_editor.py`) for physics-accurate adjustments inside
  the simulator. Start it with `bash scripts/interactive/run_editor.sh`.

## Sim-to-Real Policy Training

Policies trained entirely on SimFoundry data transfer zero-shot to real-world tasks. The table below shows simulation evaluation, real-world evaluation, and generalization to unseen digital cousin objects for two robot platforms:

> [!IMPORTANT]
> **Not yet released.** Data generation and policy training code is **not included** in this repository; the training / data-generation pipeline will ship in a future release.

<table>
  <tr>
    <th align="center"></th>
    <th align="center">Simulated Evaluation</th>
    <th align="center">Real World Evaluation</th>
    <th align="center">Real World Evaluation (Unseen Objects)</th>
  </tr>
  <tr>
    <td align="center"><b>DROID</b></td>
    <td align="center"><img src="docs/gallery/droid_sim.gif" width="220" alt="DROID Simulated Evaluation"></td>
    <td align="center"><img src="docs/gallery/droid_real.gif" width="220" alt="DROID Real Evaluation"></td>
    <td align="center"><img src="docs/gallery/droid_real_cousin.gif" width="220" alt="DROID Real Evaluation with Unseen Objects"></td>
  </tr>
  <tr>
    <td align="center"><b>YAM (Bimanual)</b></td>
    <td align="center"><img src="docs/gallery/yam_sim.gif" width="220" alt="YAM Simulated Evaluation"></td>
    <td align="center"><img src="docs/gallery/yam_real_twin.gif" width="220" alt="YAM Real Evaluation"></td>
    <td align="center"><img src="docs/gallery/yam_real_cousin.gif" width="220" alt="YAM Real Evaluation with Unseen Objects"></td>
  </tr>
</table>

## Example Scenes & Assets

Example scenes are published on Hugging Face:
[nadunRanawaka1/simfoundry-assets](https://huggingface.co/datasets/nadunRanawaka1/simfoundry-assets).

```bash
hf download nadunRanawaka1/simfoundry-assets --repo-type dataset --local-dir assets
```

Open one in the [light editor](#scene-editors):

```bash
python scripts/interactive/light_editor/server.py \
  --scene assets/scenes/DROID/droid_desk_serve_fruits/droid_desk_serve_fruits_scene_state_latest.json
```

## Outputs

Pipeline data is written under `Data/<scene_name>/`. Key outputs:

| Path | Description |
|---|---|
| `s13_og/reconstructed_og_scene.json` | Final OmniGibson scene descriptor |
| `s13_og/reconstructed_scene.png` | Scene preview image |
| `prompt_cousin_structured/` | Digital cousin image proposals |
| `sim_cousins/` and `usd_cousins/` | Simulation-ready cousin assets |
| `proposed_tasks/` | Generated task YAMLs |
| `application_smoke/` | C pipeline smoke-test videos |

## What's Included

| Component | Description |
|---|---|
| `scripts/pipeline/A_reconstruction/` | 13-stage real-to-sim reconstruction pipeline |
| `scripts/pipeline/B_augmentation/` | Digital cousin generation and task proposal |
| `scripts/pipeline/C_application/` | OmniGibson scene loading, teleoperation, and evaluation |
| `scripts/interactive/` | Browser light editor and OmniGibson scene editor |
| `scripts/installation/` | Environment and checkpoint installers |
| `scripts/cfg/` | Hydra config files for all pipeline stages |
| `simfoundry/` | Core Python library (models, utils, pipeline orchestration) |

## Documentation

- [INSTALL.md](docs/INSTALL.md) — full installation and service setup guide
- [scripts/pipeline/README.md](scripts/pipeline/README.md) — stage-by-stage pipeline reference
- [Auto-background README](scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/README.md) — optional 3D Gaussian Splat background reconstruction
- [docs/INSTRUCTIONS_SCENE_EDITOR.md](docs/INSTRUCTIONS_SCENE_EDITOR.md) — light editor user guide
- [AGENTS.md](AGENTS.md) — project guide for coding agents

## Citation

If you find SimFoundry useful in your research, please cite:

```bibtex
@article{ranawaka2026simfoundry,
  title   = {SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation},
  author  = {Ranawaka, Nadun and Wong, Josiah and Pai, Wei-Lin and Chu, Wei-Teng and
             Dai, Tianyuan and Moghani, Masoud and Yin, Hang and Jiang, Yunfan and
             Durbano, Wesley and Huynh, Brandon and Fang, Yu and Xu, Danfei and
             Zhang, Ruohan and {Fei-Fei}, Li and Fan, Linxi and Wen, Bowen and
             Mandlekar, Ajay and Zhu, Yuke},
  journal = {arXiv preprint arXiv:2606.28276},
  year    = {2026},
}
```

## Acknowledgments

SimFoundry builds on a number of excellent open-source projects. We thank the teams behind [OmniGibson](https://github.com/StanfordVL/OmniGibson), [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K), [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2), [Depth Anything 3](https://github.com/bytedance-seed/Depth-Anything-3), [SAM3](https://github.com/facebookresearch/sam3), [FoundationPose](https://github.com/NVlabs/FoundationPose), and the [digital-cousins](https://github.com/cremebrule/digital-cousins) project, on which portions of this codebase are based.

## License

NVIDIA-owned SimFoundry source code is licensed under the [Apache License 2.0](LICENSE).

Portions of SimFoundry are derived from the [ACDC / digital-cousins](https://github.com/cremebrule/digital-cousins) project, Copyright (c) 2024 the ACDC authors, also licensed under Apache 2.0. Files containing derived code carry an attribution note in their header.

SimFoundry can optionally download or integrate third-party source code, models, datasets, and SDKs governed by separate terms. The Apache 2.0 license does not apply to those materials. Several optional components are non-commercial, research-only, or otherwise restricted.

See [Third-Party Licenses](THIRD_PARTY_LICENSES.md), [Third-Party Notices](THIRD_PARTY_NOTICES.md), [Patch Provenance](PATCH_PROVENANCE.md), and [INSTALL.md](docs/INSTALL.md) for component boundaries.

## Contact

For questions or support, reach out to Nadun Ranawaka at [nadun.ranawaka@gatech.edu](mailto:nadun.ranawaka@gatech.edu) or Ajay Mandlekar at [amandlekar@nvidia.com](mailto:amandlekar@nvidia.com).
