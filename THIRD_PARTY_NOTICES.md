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

SimFoundry installs, downloads, or depends on the third-party components listed
below. Each component remains licensed under its own terms and copyright, and
SimFoundry's Apache 2.0 license does not apply to them. Transitive dependencies
(libraries required only because one of these components needs them) are not
listed individually. Full details — license links and copyright holders — are in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

**Third-party projects are not vendored in this repository.** They are fetched into
the ignored local `deps/` directory during installation, or installed from PyPI, and
remain governed by their upstream terms.

The one exception is [`patches/`](patches/): nine unified diffs applied to third-party
projects at install time, each carrying fragments of upstream source in its context
and removed lines. Those fragments remain under their upstream licenses. See
[PATCH_PROVENANCE.md](PATCH_PROVENANCE.md).


## Preserved upstream notices

Where an upstream project ships its own notice file and SimFoundry distributes a patch
against that project's source, the upstream notice is reproduced verbatim under
[`third_party_notices/`](third_party_notices/).

| Upstream | Reviewed commit | Preserved as |
|---|---|---|
| Hunyuan3D-2.1 | `82920d643c0dc2f7bfd7255f45f62d386edfe60c` | [`third_party_notices/Hunyuan3D-2.1-NOTICE.txt`](third_party_notices/Hunyuan3D-2.1-NOTICE.txt) |

The preserved file is byte-identical to the upstream original (SHA-256
`ffccf6b539a82e6084d14ff064dadd22d33384d6164b07c0c5a3141810df0350`) and must not be
edited. No other upstream patched by SimFoundry ships a notice file at its pinned
base commit.


## Preserved upstream license texts

`patches/Any6D.patch` and `patches/Hunyuan3D-2.1.patch` carry fragments of upstream
source whose licenses condition redistribution on the license terms accompanying the
code (Any6D) or on recipients receiving a copy of the license agreement (Tencent
Hunyuan 3D 2.1 Community License §3(a)). Both license texts are therefore reproduced
verbatim, byte-identical to the upstream originals at the pinned base commits:

| Upstream | Reviewed commit | Preserved as | SHA-256 |
|---|---|---|---|
| Any6D | `80eb4866a1c96ecb18be18836aba4f4bd6e80e9e` | [`third_party_notices/Any6D-LICENSE.txt`](third_party_notices/Any6D-LICENSE.txt) | `3caf5f91185e0e78c97191e8a3300461a9ac84866edb46a2178b3694b220d297` |
| Hunyuan3D-2.1 | `82920d643c0dc2f7bfd7255f45f62d386edfe60c` | [`third_party_notices/Hunyuan3D-2.1-LICENSE.txt`](third_party_notices/Hunyuan3D-2.1-LICENSE.txt) | `b79ac5e11ce063b6c6570dbe9686a45a03ba08bd248aa6aa82fb342a23a81c0c` |

These files must not be edited. Note that the Tencent Hunyuan 3D 2.1 Community
License grants rights only for a Territory that excludes the European Union, the
United Kingdom, and South Korea.


## Upstream project SimFoundry is derived from

- ACDC / digital-cousins (Apache-2.0) — Copyright (c) 2024 the ACDC authors
  (Stanford Vision and Learning Lab). Portions of `simfoundry/` are derived
  from this project; derived files carry an attribution note in their header.


## Vendored third-party code distributed with this repository

- three.js r160, incl. OrbitControls, TransformControls, GLTFLoader, and
  BufferGeometryUtils (MIT) — Copyright © 2010-2023 three.js authors. Vendored
  under `scripts/interactive/light_editor/web/vendor/` so the browser editor does
  not depend on a CDN; the full license text is reproduced in
  [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) §7.


## Third-party projects fetched into `deps/` at install time (not distributed)

- OmniGibson — BEHAVIOR-1K (MIT)
- BDDL — BEHAVIOR Domain Definition Language (MIT)
- yamlab — YAM robot arm USD, `yam.usd` (MIT)
- DINOv2 (Apache-2.0) — model-weight add-ons carry non-commercial licenses
- SAM 3 — Segment Anything Model 3 (SAM License, Meta — non-OSS)
- Depth-Anything-3 (Apache-2.0)
- Prior-Depth-Anything (Apache-2.0)
- Depth Pro / ml-depth-pro (Apple Sample Code License)
- VOID model (Apache-2.0 code; CogVideoX License weights)
- Any6D (custom non-commercial / academic-only)
- Hunyuan3D-2.1 (Tencent Hunyuan 3D 2.1 Community License — non-OSS; licensed
  Territory excludes the EU, UK, and South Korea)
- articulate-anything (MIT) —
  - CoTracker — Meta (CC-BY-NC-4.0)
  - PartField — NVIDIA License, non-commercial (NVIDIA-origin)
  - Hunyuan3D-Part, incl. P3-SAM & X-Part — Tencent Hunyuan 3D-Part Community License (non-OSS)
  - Renderers: pyrender (MIT, default); Blender (GPL-2.0-or-later) optional — downloaded from
    blender.org and invoked as a separate process (not redistributed by SimFoundry).
- openpi / openpi-client (Apache-2.0; Gemma Terms for some weights)
- Pixal3D (MIT) — optional `pixal3d` mesh backend. `install_pixal3d.sh` additionally fetches the
  following at install time (into `deps/pixal3d-weights/` and torch's hub cache) unless
  `--skip-weights` is passed:
  - DINOv3 ViT-L/16 weights — Meta (DINOv3 License, non-OSS, **gated**: manual approval)
  - BRIA RMBG-2.0 — BRIA AI (`license: other`, CC-BY-NC-4.0 terms — **non-commercial**, **gated**).
    Selected by Pixal3D's published `pipeline.json`. SimFoundry substitutes a stub so these
    weights are never downloaded, and requires RGBA inputs instead; see
    `simfoundry/models/mesh_generator.py`.
  - MoGe / MoGe-2 — Microsoft (MIT), camera FOV estimation
  - NAF — Neighborhood Attention Filtering, valeo.ai (Apache-2.0 for its own code). The pinned
    checkout additionally vendors `src/layers/rope.py` under the **DINOv3 License Agreement**
    (Meta), which is imported and executed on every generation. Cloned into torch's hub cache by
    `install_pixal3d.sh`.
  - NATTEN — Neighborhood Attention Extension (MIT), built from source at install
  - utils3d (MIT) — installed from a pinned, SHA256-verified GitHub release wheel, not PyPI

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
- packaging (Apache-2.0 OR BSD-2-Clause)
- coverage.py (Apache-2.0)
- setuptools (MIT)

## Optional teleoperation / capture dependencies

- pyzed — ZED SDK Python API (MIT bindings; proprietary ZED SDK)
- TeleMoMa (no license file — all rights reserved; not installed by SimFoundry,
  user-supplied only)
- MediaPipe (Apache-2.0)
- pyspacemouse (MIT)
- hidapi / cython-hidapi (BSD-3-Clause / GPL-3.0)

## NVIDIA proprietary platform software (not open source)

SimFoundry runs on NVIDIA proprietary platform software, governed by NVIDIA licence
terms and not by SimFoundry's Apache-2.0 licence. Not distributed by SimFoundry.

- Isaac Sim (NVIDIA Omniverse License Agreement)
- Omniverse Kit runtime — `omni.ui`, `omni.appwindow` (NVIDIA Omniverse License Agreement)
- Omniverse Kit USD libraries — `pxr` (NVIDIA Omniverse License Agreement)
- NuRec neural-reconstruction compositor (NVIDIA Omniverse License Agreement)
- NGC container registry, `nvcr.io` — optional (NVIDIA NGC Terms of Use)

The installer accepts these terms on your behalf via `--accept-nvidia-eula` and
`OMNI_KIT_ACCEPT_EULA=YES`; see [INSTALL.md](docs/INSTALL.md).

## NVIDIA-origin open components (for completeness — not third-party)

These come from NVIDIA and are not third-party. Like everything else in this
document they are fetched or installed at build time, not distributed in this
repository. Several are under the **NVIDIA Source Code License
(non-commercial)**, not SimFoundry's Apache-2.0:

- 3DGRUT — 3D Gaussian Ray Tracing (Apache-2.0)
- FoundationPose (NVIDIA Source Code License — non-commercial)
- FoundationStereo (NVIDIA Source Code License — non-commercial)
- nvdiffrast (NVIDIA Source Code License, 1-Way Commercial — non-commercial)

---

For full license identifiers, copyright holders, and links, see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
