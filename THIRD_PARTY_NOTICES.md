# Third-Party Notices

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

SimFoundry incorporates, installs, or depends on the third-party components
listed below. Each component remains licensed under its own terms and copyright.
Transitive dependencies (libraries required only because one of these components
needs them) are not listed individually. Full details — license links and
copyright holders — are in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).


## Bundled third-party projects (source vendored under `deps/`)

- OmniGibson — BEHAVIOR-1K (MIT)
- BDDL — BEHAVIOR Domain Definition Language (MIT)
- JoyLo / GELLO teleop (MIT)
- DINOv2 (Apache-2.0) — model-weight add-ons carry non-commercial licenses
- SAM 3 — Segment Anything Model 3 (SAM License, Meta — non-OSS)
- Depth-Anything-3 (Apache-2.0)
- Prior-Depth-Anything (Apache-2.0)
- Depth Pro / ml-depth-pro (Apple Sample Code License)
- VOID model (Apache-2.0 code; CogVideoX License weights)
- Any6D (custom non-commercial / academic-only)
- Hunyuan3D-2.1 (Tencent Hunyuan 3D 2.1 Community License — non-OSS)
- articulate-anything (MIT) —
  - CoTracker — Meta (CC-BY-NC-4.0)
  - samesh — Segment Any Mesh (no license provided by upstream)
  - PartField — NVIDIA License, non-commercial (NVIDIA-origin)
  - Hunyuan3D-Part, incl. P3-SAM & X-Part — Tencent Hunyuan 3D-Part Community License (non-OSS)
  - Renderers: pyrender (MIT, default); Blender (GPL-2.0-or-later) optional — downloaded from
    blender.org and invoked as a separate process (not redistributed by SimFoundry).
- openpi / openpi-client (Apache-2.0; Gemma Terms for some weights)

## Third-party projects installed at build time (git clone / pip)

- OpenAI CLIP (MIT)
- SAM 2 — Segment Anything 2 (Apache-2.0)
- Nerfstudio (Apache-2.0)
- LeRobot (Apache-2.0)
- PyTorch3D (BSD-3-Clause)
- TRELLIS.2 (MIT)
- CoACD (MIT)
- xFormers (BSD-3-Clause)

## Direct third-party Python libraries

- NumPy (BSD-3-Clause)
- PyTorch — torch, torchvision (BSD-3-Clause)
- Pillow (MIT-CMU / HPND)
- opencv-python (Apache-2.0 / MIT)
- scikit-image (BSD-3-Clause)
- SciPy (BSD-3-Clause)
- Open3D (MIT)
- trimesh (MIT)
- Shapely (BSD-3-Clause)
- Matplotlib (Matplotlib License, PSF-based)
- imageio (BSD-2-Clause)
- h5py (BSD-3-Clause)
- NetworkX (BSD-3-Clause)
- lxml (BSD-3-Clause)
- PyYAML (MIT)
- tqdm (MPL-2.0 AND MIT)
- requests (Apache-2.0)
- click (BSD-3-Clause)
- tyro (MIT)
- msgpack (Apache-2.0)
- PyZMQ (BSD-3-Clause)
- Hydra / hydra-core (MIT)
- OmegaConf (BSD-3-Clause)
- Transformers (Apache-2.0)
- Diffusers (Apache-2.0)
- Accelerate (Apache-2.0)
- Sentence-Transformers (Apache-2.0)
- SentencePiece (Apache-2.0)
- einops (MIT)
- FAISS (MIT)
- supervision (MIT)
- PyMeshLab (GPL-3.0-only)
- plyfile (GPL-3.0-or-later)
- PyBullet (Zlib)
- probreg (MIT)
- pyglet (BSD-3-Clause)
- torch-cluster (MIT)
- gdown (MIT)
- transformations (BSD-3-Clause)
- decord (Apache-2.0)
- CVXPY (Apache-2.0)
- embreex (Apache-2.0)
- PyAV (BSD-3-Clause)
- rembg (MIT)
- google-cloud-aiplatform (Apache-2.0)
- google-genai (Apache-2.0)
- openai (Apache-2.0)
- OpenUSD / pxr (Modified Apache-2.0 / Tomorrow Open Source Technology License)
- packaging (Apache-2.0 OR BSD-2-Clause)
- coverage.py (Apache-2.0)
- setuptools (MIT)

## Optional teleoperation / capture dependencies

- pyzed — ZED SDK Python API (MIT bindings; proprietary ZED SDK)
- TeleMoMa (no license file — all rights reserved)
- MediaPipe (Apache-2.0)
- pyspacemouse (MIT)
- hidapi / cython-hidapi (BSD-3-Clause / GPL-3.0)

## NVIDIA-origin bundled components (for completeness — not third-party)

These come from NVIDIA and are not third-party, but several are under the
**NVIDIA Source Code License (non-commercial)**, not SimFoundry's Apache-2.0:

- 3DGRUT — 3D Gaussian Ray Tracing (Apache-2.0)
- FoundationPose (NVIDIA Source Code License — non-commercial)
- FoundationStereo (NVIDIA Source Code License — non-commercial)
- nvdiffrast (NVIDIA Source Code License, 1-Way Commercial — non-commercial)

---

For full license identifiers, copyright holders, and links, see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
