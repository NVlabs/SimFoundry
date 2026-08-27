# Third-Party Licenses

NVIDIA-owned SimFoundry source code is released under the
[Apache License 2.0](LICENSE). SimFoundry installs, downloads, or depends on the
third-party components listed below. Each component remains under its own license
and copyright; nothing in this file changes those terms, and **the Apache 2.0
license does not apply to those materials**.

**Third-party projects are not vendored in this repository.** They are fetched into
the ignored local `deps/` directory during installation, or installed from PyPI, and
remain governed by their upstream terms.

Two exceptions: the nine patch files under [`patches/`](patches/) carry fragments of
upstream source in their context and removed lines — three under non-OSS or
non-commercial terms (see [PATCH_PROVENANCE.md](PATCH_PROVENANCE.md)) — and the
three.js modules vendored under `scripts/interactive/light_editor/web/vendor/`
(item 0f below, MIT).

Scope and method:
- The lists cover components **SimFoundry uses directly** — either its own code
  (`simfoundry/`, `scripts/`) imports/invokes them, they are fetched into
  `deps/` by an installation script, or they are declared in the project's own
  `requirements*.txt` / installation scripts.
- Licenses were reconciled against the authoritative upstream repository for each
  project.
- Several components are **non-commercial, research-only, or otherwise
  restricted**, and several model weights carry terms separate from their source
  code. See [INSTALL.md](docs/INSTALL.md) for the optional-component boundaries.
- **License links** are commit-pinned (`/blob/<sha>/`) for components fetched at a
  pinned commit. PyPI packages link to the default branch, since no single commit
  applies — the governing terms are those of the version `requirements*.txt` resolves.
---

## 0. Upstream project SimFoundry is derived from

Portions of SimFoundry's `simfoundry/` package are derived from the ACDC /
digital-cousins project and are present in this repository as modified NVIDIA code.
Derived files carry an attribution note in their header.

| No. | Component | License | Copyright | License Link |
|-----|-----------|---------|-----------|--------------|
| 0 | ACDC / digital-cousins (upstream of SimFoundry) | Apache-2.0 | Copyright (c) 2024 the ACDC authors (Stanford Vision and Learning Lab) | https://github.com/cremebrule/digital-cousins/blob/5a6d120fa1e3808779cfdf887b2169cbe73c3678/LICENSE |
| 0e | behavior-1k/omnigibson-robot-assets (OmniGibson robot assets, incl. `franka_robotiq`) | MIT | Copyright (c) Stanford Vision and Learning Lab (BEHAVIOR-1K) | https://huggingface.co/datasets/behavior-1k/omnigibson-robot-assets |
| 0g | yamlab (YAM robot arm USD, `yam.usd`) | MIT | Copyright (c) 2026 Tianyuan Dai | https://github.com/ARISE-Initiative/yamlab/blob/ec0455d2b4ce35f21fc126418ea5e74ac567133d/LICENSE |

### 0a. Additional adapted sources

Individual files adapted from or vendored from other projects, each carrying an
attribution note in its header or an accompanying provenance README. Upstream copyright
is reproduced where the upstream declares one. The full MIT license texts for the
vendored urdfpy, OmniGibson, and three.js portions (items 0a, 0d, and 0f) are
reproduced in [§7](#7-full-license-texts-for-vendored-code).

| No. | Component | Adapted into | License | Copyright | License Link |
|-----|-----------|--------------|---------|-----------|--------------|
| 0a | urdfpy | `simfoundry/utils/urdfpy_utils.py` | **MIT** | Copyright (c) 2019 Matthew Matl | https://github.com/mmatl/urdfpy/blob/5466842899b33bd549e8f9e2a9a987bd5e37373b/LICENSE |
| 0b | MolmoSpaces | `simfoundry/utils/data_gen_utils.py` | Apache-2.0 | Copyright 2026 Allen Institute for AI | https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/LICENSE |
| 0d | OmniGibson (BEHAVIOR-1K) | `simfoundry/utils/asset_conversion_utils.py` | **MIT** | Copyright (c) 2023 Stanford Vision and Learning Group | https://github.com/StanfordVL/BEHAVIOR-1K/blob/d89aae4e0e9a1de3cf8285cb9669c11d8c8bb864/OmniGibson/LICENSE |
| 0f | three.js r160 (incl. OrbitControls, TransformControls, GLTFLoader, BufferGeometryUtils) | `scripts/interactive/light_editor/web/vendor/` | **MIT** | Copyright © 2010-2023 three.js authors | https://github.com/mrdoob/three.js/blob/r160/LICENSE |

## 1. Third-party projects fetched into `deps/` at install time (not distributed)

| No. | Component | License | Copyright | License Link |
|-----|-----------|---------|-----------|--------------|
| 1 | OmniGibson (BEHAVIOR-1K) | MIT | Copyright (c) 2023 Stanford Vision and Learning Group | https://github.com/StanfordVL/BEHAVIOR-1K/blob/d89aae4e0e9a1de3cf8285cb9669c11d8c8bb864/OmniGibson/LICENSE |
| 2 | BDDL — BEHAVIOR Domain Definition Language | MIT | Copyright (c) 2021 Stanford Vision and Learning Lab | https://github.com/StanfordVL/bddl/blob/master/LICENSE |
| 4 | DINOv2 | Apache-2.0 (code) | Copyright (c) Meta Platforms, Inc. and affiliates | https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/LICENSE |
| 5 | SAM 3 — Segment Anything Model 3 | SAM License (Meta, custom — source-available, **non-OSS**) | Copyright Meta Platforms, Inc. and affiliates | https://github.com/facebookresearch/sam3/blob/46957e47805eaa273f4aa7bbbd25a88bca9108ce/LICENSE |
| 6 | Depth-Anything-3 | Apache-2.0 | Copyright 2025 The Depth Anything 3 Team (ByteDance) | https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/LICENSE |
| 7 | Prior-Depth-Anything | Apache-2.0 | Copyright the Prior-Depth-Anything authors (SpatialVision) | https://github.com/SpatialVision/Prior-Depth-Anything/blob/8c029cbca669443fe0bbf8dcefb5f91ad531084d/LICENSE |
| 8 | Depth Pro (ml-depth-pro) | Apple Sample Code License (custom permissive) | Copyright (C) 2024 Apple Inc. | https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/LICENSE |
| 9 | VOID model | Apache-2.0 (code); **CogVideoX License** (weights) | Copyright Netflix, Inc. | https://github.com/netflix/void-model/blob/e3914f8f551dd4b880661991fd6b28cd1699a97a/LICENSE |
| 10 | Any6D | Custom **non-commercial / academic-only** — full text preserved at [`third_party_notices/Any6D-LICENSE.txt`](third_party_notices/Any6D-LICENSE.txt) | Copyright (c) 2025 Taeyeop Lee | https://github.com/taeyeopl/Any6D/blob/80eb4866a1c96ecb18be18836aba4f4bd6e80e9e/LICENSE |
| 11 | Hunyuan3D-2.1 | Tencent Hunyuan 3D 2.1 Community License (**non-OSS**; **licensed Territory excludes the EU, UK, and South Korea**) — full text preserved at [`third_party_notices/Hunyuan3D-2.1-LICENSE.txt`](third_party_notices/Hunyuan3D-2.1-LICENSE.txt) | Copyright (C) 2025 Tencent | https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/LICENSE |
| 12 | articulate-anything | MIT | Copyright (c) 2024 Long Le and the Articulate Anything authors| https://github.com/vlongle/articulate-anything/blob/main/LICENSE |
| 13 | openpi (openpi-client) | Apache-2.0 (client); **Gemma Terms** (some weights) | Copyright Physical Intelligence | https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/LICENSE |

### 1a. Articulation-stage backends (fetched at install into `deps/articulate-anything/deps/`)

articulate-anything itself (item 12) is MIT, but the articulation feature
(stage 8b) invokes the following sub-projects —

| No. | Component | License | Copyright | License Link |
|-----|-----------|---------|-----------|--------------|
| 12a | CoTracker | **CC-BY-NC-4.0** (Attribution-NonCommercial) | Copyright (c) Meta Platforms, Inc. and affiliates | https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md |
| 12c | PartField (NVIDIA-origin) | NVIDIA License (**non-commercial** for third parties) | Copyright (c) NVIDIA Corporation & affiliates | https://github.com/nv-tlabs/PartField/blob/main/LICENSE |
| 12d | Hunyuan3D-Part — incl. P3-SAM, X-Part | Tencent Hunyuan 3D-Part Community License (**non-OSS**) | Copyright (C) 2025 Tencent | https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/blob/main/LICENSE |


## 2. Third-party projects installed at build time (git clone / pip)

| No. | Component | License | Copyright | License Link |
|-----|-----------|---------|-----------|--------------|
| 14 | OpenAI CLIP | MIT | Copyright (c) 2021 OpenAI | https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/LICENSE |
| 15 | SAM 2 — Segment Anything 2 | Apache-2.0 | Copyright (c) Meta Platforms, Inc. and affiliates | https://github.com/facebookresearch/sam2/blob/main/LICENSE |
| 16 | Nerfstudio | Apache-2.0 | Copyright 2022 The Nerfstudio Team | https://github.com/nerfstudio-project/nerfstudio/blob/main/LICENSE |
| 17 | LeRobot | Apache-2.0 | Copyright The HuggingFace Inc. team | https://github.com/huggingface/lerobot/blob/577cd10974b84bea1f06b6472eb9e5e74e07f77a/LICENSE |
| 18 | PyTorch3D | BSD-3-Clause | Copyright (c) Meta Platforms, Inc. and affiliates | https://github.com/facebookresearch/pytorch3d/blob/75ebeeaea0908c5527e7b1e305fbc7681382db47/LICENSE |
| 19 | TRELLIS.2 | MIT | Copyright (c) Microsoft Corporation | https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/LICENSE |
| 20 | CoACD | MIT | Copyright (c) 2022 Xinyue Wei and contributors | https://github.com/SarahWeiii/CoACD/blob/main/LICENSE |
| 21 | xFormers | BSD-3-Clause | Copyright (c) Facebook, Inc. and its affiliates | https://github.com/facebookresearch/xformers/blob/4cf69f0967128217f1798de70b3e4477de138570/LICENSE |
| 21a | Real-ESRGAN (`RealESRGAN_x4plus.pth` weight, fetched for Hunyuan3D-2.1) | BSD-3-Clause | Copyright (c) 2021 Xintao Wang | https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE |

## 3. Direct third-party Python libraries (PyPI)

| No. | Package | License | Copyright | License Link |
|-----|---------|---------|-----------|--------------|
| 22 | NumPy | BSD-3-Clause | Copyright (c) 2005-present, NumPy Developers | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| 23 | PyTorch (torch, torchvision) | BSD-3-Clause | Copyright (c) 2016-present, PyTorch contributors | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| 24 | Pillow (PIL) | MIT-CMU (HPND) | Copyright (c) 2010-present Jeffrey A. Clark and contributors | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| 25 | opencv-python (cv2) | Apache-2.0 (OpenCV) / MIT (wrapper) | Copyright (c) OpenCV team; wrapper (c) Olli-Pekka Heinisuo | https://github.com/opencv/opencv/blob/master/LICENSE |
| 26 | scikit-image | BSD-3-Clause | Copyright (c) 2009-present, the scikit-image team | https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt |
| 27 | SciPy | BSD-3-Clause | Copyright (c) 2001-present, SciPy Developers | https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| 28 | Open3D | MIT | Copyright (c) 2018-present www.open3d.org | https://github.com/isl-org/Open3D/blob/main/LICENSE |
| 29 | trimesh | MIT | Copyright (c) 2019 Michael Dawson-Haggerty | https://github.com/mikedh/trimesh/blob/main/LICENSE.md |
| 30 | Shapely | BSD-3-Clause | Copyright (c) 2007, Sean C. Gillies; Shapely contributors | https://github.com/shapely/shapely/blob/main/LICENSE.txt |
| 31 | Matplotlib | Matplotlib License (PSF-based, BSD-style) | Copyright (c) 2012– Matplotlib Development Team | https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE |
| 32 | imageio | BSD-2-Clause | Copyright (c) 2014–, imageio developers | https://github.com/imageio/imageio/blob/master/LICENSE |
| 33 | h5py | BSD-3-Clause | Copyright (c) 2008 Andrew Collette and contributors | https://github.com/h5py/h5py/blob/master/LICENSE |
| 34 | NetworkX | BSD-3-Clause | Copyright (c) 2004–2024, NetworkX Developers | https://github.com/networkx/networkx/blob/main/LICENSE.txt |
| 35 | lxml | BSD-3-Clause | Copyright (c) 2004 Infrae | https://github.com/lxml/lxml/blob/master/LICENSE.txt |
| 36 | PyYAML | MIT | Copyright (c) 2017–2021 Ingy döt Net; (c) 2006–2016 Kirill Simonov | https://github.com/yaml/pyyaml/blob/main/LICENSE |
| 37 | tqdm | MPL-2.0 AND MIT | Copyright (c) 2013 noamraph; MPL portions (c) 2015–2024 Casper da Costa-Luis | https://github.com/tqdm/tqdm/blob/master/LICENCE |
| 38 | requests | Apache-2.0 | Copyright 2019 Kenneth Reitz | https://github.com/psf/requests/blob/main/LICENSE |
| 39 | click | BSD-3-Clause | Copyright 2014 Pallets | https://github.com/pallets/click/blob/main/LICENSE.txt |
| 40 | tyro | MIT | Copyright (c) 2023 Brent Yi | https://github.com/brentyi/tyro/blob/main/LICENSE |
| 41 | msgpack (msgpack-python) | Apache-2.0 | Copyright (C) 2008–2011 INADA Naoki | https://github.com/msgpack/msgpack-python/blob/main/COPYING |
| 42 | PyZMQ | BSD-3-Clause | Copyright (c) 2009–2012, Brian Granger, Min Ragan-Kelley, PyZMQ developers | https://github.com/zeromq/pyzmq/blob/main/LICENSE.md |
| 43 | Hydra (hydra-core) | MIT | Copyright (c) Facebook, Inc. and its affiliates | https://github.com/facebookresearch/hydra/blob/main/LICENSE |
| 44 | OmegaConf | BSD-3-Clause | Copyright (c) 2018, Omry Yadan | https://github.com/omry/omegaconf/blob/master/LICENSE |
| 45 | Transformers | Apache-2.0 | Copyright 2018– The HuggingFace Inc. team | https://github.com/huggingface/transformers/blob/main/LICENSE |
| 46 | Diffusers | Apache-2.0 | Copyright 2018– The HuggingFace Inc. team | https://github.com/huggingface/diffusers/blob/main/LICENSE |
| 47 | Accelerate | Apache-2.0 | Copyright 2021– The HuggingFace Inc. team | https://github.com/huggingface/accelerate/blob/main/LICENSE |
| 48 | Sentence-Transformers | Apache-2.0 | Copyright 2019 Nils Reimers (UKPLab) / Hugging Face | https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE |
| 49 | SentencePiece | Apache-2.0 | Copyright Google Inc. | https://github.com/google/sentencepiece/blob/master/LICENSE |
| 50 | einops | MIT | Copyright (c) 2018 Alex Rogozhnikov | https://github.com/arogozhnikov/einops/blob/main/LICENSE |
| 51 | FAISS | MIT | Copyright (c) Meta Platforms, Inc. and affiliates | https://github.com/facebookresearch/faiss/blob/main/LICENSE |
| 52 | supervision | MIT | Copyright (c) 2022 Roboflow | https://github.com/roboflow/supervision/blob/develop/LICENSE.md |
| 53 | PyMeshLab | **GPL-3.0-only** | Copyright (c) CNR-ISTI Visual Computing Lab | https://github.com/cnr-isti-vclab/PyMeshLab/blob/main/LICENSE |
| 54 | plyfile | **GPL-3.0-or-later** | Copyright (C) Darsh Ranjan and plyfile authors | https://github.com/dranjan/python-plyfile/blob/master/COPYING |
| 55 | PyBullet (bullet3) | Zlib | Copyright (c) 2003–2021 Erwin Coumans / Bullet contributors | https://github.com/bulletphysics/bullet3/blob/master/LICENSE.txt |
| 56 | probreg | MIT | Copyright (c) 2019 neka-nat | https://github.com/neka-nat/probreg/blob/master/LICENSE |
| 57 | pyglet | BSD-3-Clause | Copyright (c) 2006–2008 Alex Holkner / pyglet contributors | https://github.com/pyglet/pyglet/blob/master/LICENSE |
| 58 | torch-cluster (pytorch_cluster) | MIT | Copyright (c) 2020 Matthias Fey | https://github.com/rusty1s/pytorch_cluster/blob/master/LICENSE |
| 59 | gdown | MIT | Copyright (c) 2015 Kentaro Wada | https://github.com/wkentaro/gdown/blob/main/LICENSE |
| 60 | transformations | BSD-3-Clause | Copyright (c) 2006–2024 Christoph Gohlke | https://github.com/cgohlke/transformations/blob/master/LICENSE |
| 61 | decord | Apache-2.0 | Copyright (c) DMLC / decord contributors | https://github.com/dmlc/decord/blob/master/LICENSE |
| 62 | CVXPY | Apache-2.0 | Copyright (c) The CVXPY authors | https://github.com/cvxpy/cvxpy/blob/master/LICENSE |
| 63 | embreex | Apache-2.0 | Copyright (c) trimesh; wraps Intel Embree (Apache-2.0) | https://github.com/trimesh/embreex/blob/main/LICENSE.md |
| 64 | PyAV (av) | BSD-3-Clause | Copyright (c) 2017 Mike Boers / PyAV authors | https://github.com/PyAV-Org/PyAV/blob/main/LICENSE.txt |
| 65 | rembg | MIT | Copyright (c) 2020 Daniel Gatis | https://github.com/danielgatis/rembg/blob/main/LICENSE.txt |
| 66 | google-cloud-aiplatform (Vertex AI SDK) | Apache-2.0 | Copyright Google LLC | https://github.com/googleapis/python-aiplatform/blob/main/LICENSE |
| 67 | google-genai (Google Gen AI SDK) | Apache-2.0 | Copyright Google LLC | https://github.com/googleapis/python-genai/blob/main/LICENSE |
| 68 | openai (OpenAI Python library) | Apache-2.0 | Copyright OpenAI | https://github.com/openai/openai-python/blob/main/LICENSE |
| 69 | `pxr` USD Python bindings — **supplied by Isaac Sim, not PyPI**. Resolves to the Omniverse Kit extension `omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311` under `site-packages/isaacsim/extscache/`. No `usd-core` package is installed from PyPI. | **NVIDIA Omniverse License Agreement** (the Kit build; code sets `OMNI_KIT_ACCEPT_EULA=YES`). The upstream OpenUSD project is separately under the Modified Apache-2.0 "Tomorrow Open Source Technology License 1.0" (Copyright Pixar Animation Studios), but that is **not** the build loaded at runtime. | Copyright (c) NVIDIA CORPORATION & AFFILIATES (Kit build); Copyright Pixar Animation Studios (upstream OpenUSD) | https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html |
| 70 | packaging | Apache-2.0 OR BSD-2-Clause | Copyright (c) Donald Stufft and contributors | https://github.com/pypa/packaging/blob/main/LICENSE |
| 71 | coverage.py | Apache-2.0 | Copyright 2004 Ned Batchelder | https://github.com/nedbat/coveragepy/blob/master/LICENSE.txt |
| 72 | setuptools | MIT | Copyright (c) Python Packaging Authority (PyPA) | https://github.com/pypa/setuptools/blob/main/LICENSE |

## 4. Optional teleoperation / capture dependencies

Installed only for teleop / ZED-capture workflows (`requirements_teleop.txt`,
`--zed`).

| No. | Component | License | Copyright | License Link |
|-----|-----------|---------|-----------|--------------|
| 73 | pyzed — ZED SDK Python API | MIT (bindings); **proprietary ZED SDK** required at runtime | Copyright (c) 2018 Stereolabs | https://github.com/stereolabs/zed-python-api/blob/master/LICENSE |
| 74 | TeleMoMa | **No license file (all rights reserved)** — **not installed by SimFoundry; user-supplied only** | The University of Texas at Austin (RobIn Lab) | https://github.com/UT-Austin-RobIn/telemoma |
| 75 | MediaPipe | Apache-2.0 | Copyright The MediaPipe Authors (Google LLC) | https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE |
| 76 | pyspacemouse | MIT | Copyright (c) Jakub Andrýsek | https://github.com/JakubAndrysek/PySpaceMouse/blob/master/LICENSE |
| 77 | hidapi (cython-hidapi) | BSD-3-Clause | Copyright (c) Gary Bishop, Pavol Rusnak, contributors | https://github.com/trezor/cython-hidapi/blob/master/LICENSE.txt |

## 5. NVIDIA-origin components (for completeness — not third-party)

These originate from NVIDIA and are therefore **not third-party**, but several use the
**NVIDIA Source Code License (non-commercial)**, which is *not* the Apache-2.0 license
under which SimFoundry itself is released.

| No. | Component | License | Copyright | License Link |
|-----|-----------|---------|-----------|--------------|
| 78 | 3DGRUT (3D Gaussian Ray Tracing) | Apache-2.0 | Copyright NVIDIA Corporation & affiliates | https://github.com/nv-tlabs/3dgrut/blob/a37ef721012dea0f29c0fcfff2d525023b4e854a/LICENSE |
| 79 | FoundationPose | NVIDIA Source Code License (**non-commercial**) | Copyright (c) 2022–Present, NVIDIA Corporation & affiliates | https://github.com/NVlabs/FoundationPose/blob/e3d597b8c6b851d053094ebd6fa240191c5238f8/LICENSE |
| 80 | FoundationStereo | NVIDIA Source Code License (**non-commercial**) | Copyright (c) 2024–Present, NVIDIA Corporation & affiliates | https://github.com/NVlabs/FoundationStereo/blob/6e8806816b533e4d13ddbb95ffa907b797060a62/LICENSE |
| 81 | nvdiffrast | NVIDIA Source Code License (1-Way Commercial, **non-commercial** for third parties) | Copyright (c) 2020, NVIDIA Corporation | https://github.com/NVlabs/nvdiffrast/blob/253ac4fcea7de5f396371124af597e6cc957bfae/LICENSE.txt |
| 82 | cuRobo (reached through OmniGibson; invoked directly by `simfoundry/utils/data_gen_utils.py`) | Apache-2.0 | Copyright (c) NVIDIA CORPORATION & AFFILIATES | https://github.com/NVlabs/curobo/blob/main/LICENSE |

### 5a. NVIDIA proprietary platform software

SimFoundry runs on NVIDIA proprietary platform software. It is **not open source** and
**not third-party**, is not distributed by SimFoundry, and is governed by NVIDIA
license terms rather than SimFoundry's Apache 2.0 license. It is listed here because
SimFoundry's code calls it directly.

| No. | Component | Used by SimFoundry for | License terms | Link |
|-----|-----------|------------------------|---------------|------|
| 82 | Isaac Sim | Physics materials, debug draw, USD stage access — `lazy.isaacsim.*` in `simfoundry/utils/og_utils.py`, `scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py`, `scripts/interactive/interactive_scene_editor.py` | NVIDIA Omniverse License Agreement | https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html |
| 83 | Omniverse Kit runtime | In-viewport overlay UI — `lazy.omni.ui`, `lazy.omni.appwindow` in `simfoundry/utils/og_utils.py` | NVIDIA Omniverse License Agreement | https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html |
| 84 | Omniverse Kit USD libraries (`pxr`) | USD authoring and joint reparenting — see item 69 | NVIDIA Omniverse License Agreement | https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html |
| 85 | NuRec (Omniverse neural reconstruction / GS compositor) | Rendering Gaussian-splat backgrounds as USDZ volumes in Isaac Sim — `scripts/interactive/interactive_scene_editor.py` | NVIDIA Omniverse License Agreement | https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html |
| 86 | NGC container registry (`nvcr.io`) | Optional `docker login` for NGC-hosted images — `scripts/installation/login_services.sh`, key supplied via `api_keys.template.txt` | NVIDIA NGC Terms of Use | https://ngc.nvidia.com/legal/terms |

Isaac Sim, the Kit runtime, the Kit USD libraries and NuRec are obtained together as
part of the Isaac Sim / OmniGibson installation; SimFoundry neither redistributes nor
mirrors them. NGC access is optional and only used to pull container images.

## 6. Component disclosure matrix — restricted and optional components

This section records, for every component with terms that differ from SimFoundry's
Apache 2.0 license, the facts an evaluator needs before installing it.

**Distribution status — applies to every row below.** None are distributed by
SimFoundry: no source archive, release artifact, container, cache, model bundle, or
NVIDIA mirror. Each is fetched from its own upstream, by the user, at install time.

**The Apache 2.0 boundary.** SimFoundry's license covers only NVIDIA-authored code —
`simfoundry/`, `scripts/`, `tests/`, and the repo's own config and docs. It does
not extend to any component below, their model weights, or any dataset or SDK required.

| Component | Required? | Acquisition | Exact version | Source terms | Weights terms | Key restriction |
|---|---|---|---|---|---|---|
| SAM 3 | Required (`simfoundry`) | `git clone` — `install_simfoundry.sh` | `46957e47…` | SAM License (Meta) | Gated Hugging Face download | **Non-OSS**, source-available; HF login required |
| behavior-1k/omnigibson-robot-assets (`franka_robotiq` subtree) | Required (`simfoundry`) | `huggingface_hub.snapshot_download` — `install_simfoundry.sh` | unpinned (`main`) | MIT | n/a — assets only | Public HF dataset; only `models/franka/franka_robotiq/**` is fetched, into `deps/BEHAVIOR-1K/datasets/omnigibson-robot-assets/` |
| Any6D | Required (`any6d`) | `git clone` — `install_any6d.sh` | `80eb4866…` | Custom academic-only | n/a | **Non-commercial / academic use only** |
| Hunyuan3D-2.1 | Required (`hunyuan`) | `git clone` — `install_hunyuan.sh` | `82920d64…` | Tencent Hunyuan 3D 2.1 Community License | Same, plus Real-ESRGAN weight (BSD-3-Clause) | **Non-OSS** community license; **no grant in the EU, UK, or South Korea** |
| Hunyuan3D-Part (P3-SAM, X-Part) | Optional (`articulate`) | Fetched by articulate-anything | Tencent Hunyuan 3D-Part Community License | P3-SAM weights auto-download on first use | **Non-OSS** community license |
| PartField | Optional (`articulate`) | Fetched by articulate-anything | NVIDIA License | Checkpoint via install script | **Non-commercial for third parties** |
| FoundationPose | Required (`simfoundry`) | `git clone` — `install_simfoundry.sh` | `e3d597b8…` | NVIDIA Source Code License | Google Drive folders, **unversioned** | **Non-commercial** |
| FoundationStereo | Required (`simfoundry`) | `git clone` — `install_simfoundry.sh` | `6e880681…` | NVIDIA Source Code License | Google Drive folder, **unversioned** | **Non-commercial** |
| nvdiffrast | Required (`simfoundry`) | `git clone` tag `v0.4.0` | `253ac4fc…` | NVIDIA Source Code License (1-Way Commercial) | n/a | **Non-commercial for third parties** |
| cuRobo | Required (via OmniGibson) | Transitive — BEHAVIOR-1K install  | Apache-2.0 | n/a | None |
| Depth Pro | Required (`simfoundry`) | `git clone` — `install_simfoundry.sh` | `9efe5c1d…` | Apple Sample Code License | `depth_pro.pt` from Apple CDN | Apple sample-code terms |
| VOID / CogVideoX weights | Required (`void`) | `git fetch` — `install_void.sh` | `e3914f8f…` | Apache-2.0 (code) | **CogVideoX License**, gated HF | Weights are **not** Apache-2.0; HF login required |
| OpenPI / Gemma weights | Installed with `simfoundry` (client only) | `git clone` — `install_simfoundry.sh` | `15a9616a…` | Apache-2.0 (client) | **Gemma Terms of Use** | Weights governed by Gemma Terms |
| CoTracker | Optional (`articulate`) | Fetched by articulate-anything | **CC-BY-NC-4.0** | Same | **Non-commercial** |
| TeleMoMa | **User-supplied** | Not installed by SimFoundry | User's choice (`0.3.0` known-good) | **No license file — all rights reserved** | n/a | **No rights granted by upstream.** Not installed, distributed, or mirrored |
| TRELLIS.2 | Optional (`--trellis`) | `git clone` — `install_simfoundry.sh` | `75fbf018…` | MIT | n/a | None |
| pyzed / ZED SDK | Optional (`--zed`) | User installs the ZED SDK | User's SDK version | MIT (bindings) | n/a | **Proprietary ZED SDK** required at runtime, under Stereolabs terms |

### Components requiring your acceptance or approval before use

- **Gated downloads** — SAM 3 and the VOID/CogVideoX weights require a Hugging Face
  account and `huggingface-cli login`. You accept the model terms at that point.
- **Non-commercial components** — Any6D, FoundationPose, FoundationStereo, nvdiffrast,
  PartField, and CoTracker restrict use to non-commercial or research purposes.
  Commercial use requires separate licensing from each upstream.
- **Territory-restricted component** — the Tencent Hunyuan 3D 2.1 Community License
  grants rights only within a Territory that excludes the European Union, the United
  Kingdom, and South Korea; users in those places receive no rights from that license.
  A copy of the Agreement ships at
  [`third_party_notices/Hunyuan3D-2.1-LICENSE.txt`](third_party_notices/Hunyuan3D-2.1-LICENSE.txt).
- **No-license component** — TeleMoMa grants no rights at all. SimFoundry does not
  install, distribute, or mirror it; establish your own basis for using it, or do not
  use the teleoperation workflow.

## 7. Full license texts for vendored code

The MIT License requires that its copyright and permission notice be included with
all copies or substantial portions of the software. The notices below are reproduced
verbatim from the commit-pinned upstream `LICENSE` files for the three components
vendored in this repository (items 0a, 0d, and 0f above).

### urdfpy — vendored in `simfoundry/utils/urdfpy_utils.py`

```
MIT License

Copyright (c) 2019 Matthew Matl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The upstream license texts for the source fragments redistributed in
`patches/Any6D.patch` and `patches/Hunyuan3D-2.1.patch` are preserved byte-identical
under [`third_party_notices/`](third_party_notices/); see
[PATCH_PROVENANCE.md](PATCH_PROVENANCE.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the recorded SHA-256 checksums.

### OmniGibson (BEHAVIOR-1K) — vendored in `simfoundry/utils/asset_conversion_utils.py`

```
MIT License

Copyright (c) 2023 Stanford Vision and Learning Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### three.js — vendored in `scripts/interactive/light_editor/web/vendor/`

```
The MIT License

Copyright © 2010-2023 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
