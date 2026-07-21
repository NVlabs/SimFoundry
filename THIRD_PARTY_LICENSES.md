# Third-Party Licenses

SimFoundry is released by NVIDIA under the [Apache License 2.0](LICENSE). It
bundles, installs, or depends on the third-party components listed below. Each
component remains under its own license and copyright; nothing in this file
changes those terms.

Scope and method:
- The lists cover components **SimFoundry uses directly** — either its own code
  (`digital_cousins/`, `scripts/`) imports/invokes them, they are vendored under
  `deps/`, or they are declared in the project's own `requirements*.txt` /
  installation scripts.
- Licenses were read from each vendored `deps/<project>/LICENSE` where present
  and reconciled against the authoritative upstream repository.
---

## 1. Bundled third-party projects (source vendored under `deps/`)

| No. | Component | License | Copyright | License Link |
|-----|-----------|---------|-----------|--------------|
| 1 | OmniGibson (BEHAVIOR-1K) | MIT | Copyright (c) 2023 Stanford Vision and Learning Group | https://github.com/StanfordVL/OmniGibson/blob/main/LICENSE |
| 2 | BDDL — BEHAVIOR Domain Definition Language | MIT | Copyright (c) 2021 Stanford Vision and Learning Lab | https://github.com/StanfordVL/bddl/blob/master/LICENSE |
| 3 | JoyLo / GELLO teleop | MIT | Copyright (c) 2023 Philipp Wu | https://github.com/wuphilipp/gello/blob/main/LICENSE |
| 4 | DINOv2 | Apache-2.0 (code) | Copyright (c) Meta Platforms, Inc. and affiliates | https://github.com/facebookresearch/dinov2/blob/main/LICENSE |
| 5 | SAM 3 — Segment Anything Model 3 | SAM License (Meta, custom — source-available, **non-OSS**) | Copyright Meta Platforms, Inc. and affiliates | https://github.com/facebookresearch/sam3/blob/main/LICENSE |
| 6 | Depth-Anything-3 | Apache-2.0 | Copyright 2025 The Depth Anything 3 Team (ByteDance) | https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/LICENSE |
| 7 | Prior-Depth-Anything | Apache-2.0 | Copyright the Prior-Depth-Anything authors (SpatialVision) | https://github.com/SpatialVision/Prior-Depth-Anything/blob/main/LICENSE |
| 8 | Depth Pro (ml-depth-pro) | Apple Sample Code License (custom permissive) | Copyright (C) 2024 Apple Inc. | https://github.com/apple/ml-depth-pro/blob/main/LICENSE |
| 9 | VOID model | Apache-2.0 (code); **CogVideoX License** (weights) | Copyright Netflix, Inc. | https://github.com/netflix/void-model/blob/main/LICENSE |
| 10 | Any6D | Custom **non-commercial / academic-only** | Copyright (c) 2025 Taeyeop Lee | https://github.com/taeyeopl/Any6D/blob/main/LICENSE |
| 11 | Hunyuan3D-2.1 | Tencent Hunyuan 3D 2.1 Community License (**non-OSS**) | Copyright (C) 2025 Tencent | https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE |
| 12 | articulate-anything | MIT | Copyright (c) 2024 Long Le and the Articulate Anything authors| https://github.com/vlongle/articulate-anything/blob/main/LICENSE |
| 13 | openpi (openpi-client) | Apache-2.0 (client); **Gemma Terms** (some weights) | Copyright Physical Intelligence | https://github.com/Physical-Intelligence/openpi/blob/main/LICENSE |

### 1a. Articulation-stage backends (fetched at install into `deps/articulate-anything/deps/`)

articulate-anything itself (item 12) is MIT, but the articulation feature
(stage 8b) invokes the following sub-projects —

| No. | Component | License | Copyright | License Link |
|-----|-----------|---------|-----------|--------------|
| 12a | CoTracker | **CC-BY-NC-4.0** (Attribution-NonCommercial) | Copyright (c) Meta Platforms, Inc. and affiliates | https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md |
| 12b | samesh (Segment Any Mesh) | **No license provided** — optional dependency | Copyright samesh authors (gtangg12) — no license granted | https://github.com/gtangg12/samesh |
| 12c | PartField (NVIDIA-origin) | NVIDIA License (**non-commercial** for third parties) | Copyright (c) NVIDIA Corporation & affiliates | https://github.com/nv-tlabs/PartField/blob/main/LICENSE |
| 12d | Hunyuan3D-Part — incl. P3-SAM, X-Part | Tencent Hunyuan 3D-Part Community License (**non-OSS**) | Copyright (C) 2025 Tencent | https://github.com/Tencent-Hunyuan/Hunyuan3D-Part/blob/main/LICENSE |


## 2. Third-party projects installed at build time (git clone / pip)

| No. | Component | License | Copyright | License Link |
|-----|-----------|---------|-----------|--------------|
| 14 | OpenAI CLIP | MIT | Copyright (c) 2021 OpenAI | https://github.com/openai/CLIP/blob/main/LICENSE |
| 15 | SAM 2 — Segment Anything 2 | Apache-2.0 | Copyright (c) Meta Platforms, Inc. and affiliates | https://github.com/facebookresearch/sam2/blob/main/LICENSE |
| 16 | Nerfstudio | Apache-2.0 | Copyright 2022 The Nerfstudio Team | https://github.com/nerfstudio-project/nerfstudio/blob/main/LICENSE |
| 17 | LeRobot | Apache-2.0 | Copyright The HuggingFace Inc. team | https://github.com/huggingface/lerobot/blob/main/LICENSE |
| 18 | PyTorch3D | BSD-3-Clause | Copyright (c) Meta Platforms, Inc. and affiliates | https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE |
| 19 | TRELLIS.2 | MIT | Copyright (c) Microsoft Corporation | https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE |
| 20 | CoACD | MIT | Copyright (c) 2022 Xinyue Wei and contributors | https://github.com/SarahWeiii/CoACD/blob/main/LICENSE |
| 21 | xFormers | BSD-3-Clause | Copyright (c) Facebook, Inc. and its affiliates | https://github.com/facebookresearch/xformers/blob/main/LICENSE |

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
| 69 | OpenUSD / pxr | Modified Apache-2.0 (Tomorrow Open Source Technology License 1.0) | Copyright Pixar Animation Studios | https://github.com/PixarAnimationStudios/OpenUSD/blob/release/LICENSE.txt |
| 70 | packaging | Apache-2.0 OR BSD-2-Clause | Copyright (c) Donald Stufft and contributors | https://github.com/pypa/packaging/blob/main/LICENSE |
| 71 | coverage.py | Apache-2.0 | Copyright 2004 Ned Batchelder | https://github.com/nedbat/coveragepy/blob/master/LICENSE.txt |
| 72 | setuptools | MIT | Copyright (c) Python Packaging Authority (PyPA) | https://github.com/pypa/setuptools/blob/main/LICENSE |

## 4. Optional teleoperation / capture dependencies

Installed only for teleop / ZED-capture workflows (`requirements_teleop.txt`,
`--zed`, `--joylo`).

| No. | Component | License | Copyright | License Link |
|-----|-----------|---------|-----------|--------------|
| 73 | pyzed — ZED SDK Python API | MIT (bindings); **proprietary ZED SDK** required at runtime | Copyright (c) 2018 Stereolabs | https://github.com/stereolabs/zed-python-api/blob/master/LICENSE |
| 74 | TeleMoMa | **No license file (all rights reserved)** | The University of Texas at Austin (RobIn Lab) | https://github.com/UT-Austin-RobIn/telemoma |
| 75 | MediaPipe | Apache-2.0 | Copyright The MediaPipe Authors (Google LLC) | https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE |
| 76 | pyspacemouse | MIT | Copyright (c) Jakub Andrýsek | https://github.com/JakubAndrysek/PySpaceMouse/blob/master/LICENSE |
| 77 | hidapi (cython-hidapi) | Tri-licensed: BSD-3-Clause / GPL-3.0 / custom (your choice) | Copyright (c) Gary Bishop, Pavol Rusnak, contributors | https://github.com/trezor/cython-hidapi/blob/master/LICENSE.txt |

## 5. NVIDIA-origin bundled components (for completeness — not third-party)

These originate from NVIDIA and are therefore **not third-party**, but they are
bundled/installed and carry their own licenses. **Note:** several use the
**NVIDIA Source Code License (non-commercial)**, which is *not* the Apache-2.0
license under which SimFoundry itself is released.

| No. | Component | License | Copyright | License Link |
|-----|-----------|---------|-----------|--------------|
| 78 | 3DGRUT (3D Gaussian Ray Tracing) | Apache-2.0 | Copyright NVIDIA Corporation & affiliates | https://github.com/nv-tlabs/3dgrut/blob/main/LICENSE |
| 79 | FoundationPose | NVIDIA Source Code License (**non-commercial**) | Copyright (c) 2022–Present, NVIDIA Corporation & affiliates | https://github.com/NVlabs/FoundationPose/blob/main/LICENSE |
| 80 | FoundationStereo | NVIDIA Source Code License (**non-commercial**) | Copyright (c) 2024–Present, NVIDIA Corporation & affiliates | https://github.com/NVlabs/FoundationStereo/blob/master/LICENSE |
| 81 | nvdiffrast | NVIDIA Source Code License (1-Way Commercial, **non-commercial** for third parties) | Copyright (c) 2020, NVIDIA Corporation | https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt |
