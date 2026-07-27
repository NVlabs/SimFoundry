# Patch Provenance

SimFoundry distributes eight patch files under [`patches/`](patches/). Each is a
unified diff that an installation script applies to a third-party project after it
is fetched into the ignored local `deps/` directory.

**These patch files are the one place where SimFoundry does redistribute
third-party source code.** A unified diff carries upstream code in its context
lines (` `) and removed lines (`-`), so each patch reproduces a fragment of the
project it targets. That code remains under its upstream license — it is **not**
covered by SimFoundry's Apache 2.0 license, and a patch is not Apache-2.0 merely
because NVIDIA authored the diff.

Three of the eight carry fragments of source that is **not open source**:

| Patch | Upstream license of the carried source |
|---|---|
| `Any6D.patch` | Custom **non-commercial / academic-only** |
| `FoundationPose.patch` | **NVIDIA Source Code License** (non-commercial) |
| `Hunyuan3D-2.1.patch` | **Tencent Hunyuan 3D 2.1 Community License** (non-OSS) |

## Summary

All base commits below are the exact revisions the installation scripts pin, so
each patch is recorded against the source it is actually applied to.

| Patch | Upstream | Base commit | Upstream license | Owner | Applied by | `git apply --check` |
|---|---|---|---|---|---|---|
| `3dgrut.patch` | [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut) | `a37ef721012dea0f29c0fcfff2d525023b4e854a` | Apache-2.0 | SimFoundry team | `install_3dgrut.sh` | ✅ clean |
| `Any6D.patch` | [taeyeopl/Any6D](https://github.com/taeyeopl/Any6D) | `80eb4866a1c96ecb18be18836aba4f4bd6e80e9e` | **Non-commercial / academic-only** | SimFoundry team | `install_any6d.sh` | ✅ clean |
| `FoundationPose.patch` | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) | `e3d597b8c6b851d053094ebd6fa240191c5238f8` | **NVIDIA Source Code License (non-commercial)** | SimFoundry team | `install_cdc.sh` | ✅ clean |
| `Hunyuan3D-2.1.patch` | [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | `82920d643c0dc2f7bfd7255f45f62d386edfe60c` | **Tencent Hunyuan 3D 2.1 Community License (non-OSS)** | SimFoundry team | `install_hunyuan.sh` | ✅ clean |
| `ml-depth-pro.patch` | [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) | `9efe5c1def37a26c5367a71df664b18e1306c708` | Apple Sample Code License | SimFoundry team | `install_cdc.sh` | ✅ clean |
| `PriorDepthAnything.patch` | [SpatialVision/Prior-Depth-Anything](https://github.com/SpatialVision/Prior-Depth-Anything) | `8c029cbca669443fe0bbf8dcefb5f91ad531084d` | Apache-2.0 | SimFoundry team | `install_cdc.sh` | ✅ clean |
| `splatfacto_depth_loss.patch` | [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio) | `6b60855003011b2ca23c2fe3f8e2ca6314c69924` (tag `v1.1.5`) | Apache-2.0 | SimFoundry team | `install_nerfstudio.sh` | ✅ clean |
| `void-model.patch` | [netflix/void-model](https://github.com/netflix/void-model) | `e3914f8f551dd4b880661991fd6b28cd1699a97a` | Apache-2.0 | SimFoundry team | `install_void.sh` | ✅ clean |

Validation was performed on 2026-07-27 by fetching each upstream at the base commit
above and running:

```bash
git -C <upstream-checkout> apply --check patches/<patch>
```

All eight applied cleanly (nine at the time of validation; `lerobot.patch` has since been removed). Re-run this whenever a base commit is bumped.

## Upstream source carried per patch

"Upstream lines" counts context plus removed lines — the upstream code physically
present in the patch file. "Added" lines are NVIDIA-authored.

| Patch | Context | Removed | Upstream lines | Added (NVIDIA) | Files touched |
|---|---|---|---|---|---|
| `Any6D.patch` | 112 | 96 | **208** | 130 | 4 |
| `FoundationPose.patch` | 43 | 68 | **111** | 68 | 3 |
| `PriorDepthAnything.patch` | 32 | 27 | **59** | 28 | 5 |
| `Hunyuan3D-2.1.patch` | 22 | 3 | **25** | 10 | 3 |
| `splatfacto_depth_loss.patch` | 13 | 0 | **13** | 46 | 1 |
| `ml-depth-pro.patch` | 9 | 2 | **11** | 2 | 1 |
| `3dgrut.patch` | 6 | 1 | **7** | 4 | 1 |
| `void-model.patch` | 6 | 0 | **6** | 13 | 1 |

## Per-patch detail

### `3dgrut.patch`
- **Target**: `threedgrut/export/scripts/ply_to_usd.py`
- **Purpose**: Sets `export_cameras=False` for the PLY-only export path, which has no
  camera dataset. A background Gaussian-splat volume for OmniGibson/NuRec needs only
  geometry; without this the PLY→USDZ step fails with
  `ValueError: export_cameras=True requires a dataset`.
- **Upstream NOTICE**: none at base commit.

### `Any6D.patch`
- **Targets**: `estimater.py`, `foundationpose/Utils.py`,
  `foundationpose/bundlesdf/mycuda/common.cu`, `requirements.txt`
- **Purpose**: Rotation-grid tuning for pose hypotheses, `trimesh.visual.material`
  import fix, CUDA kernel dtype dispatch, dependency bounds.
- **Note**: Any6D vendors its own copy of FoundationPose, so this patch modifies
  **FoundationPose source nested inside Any6D**. Two upstream licenses apply to one
  patch file — Any6D's academic-only terms and the NVIDIA Source Code License.
- **Upstream NOTICE**: none at base commit.

### `FoundationPose.patch`
- **Targets**: `bundlesdf/mycuda/common.cu`, `bundlesdf/mycuda/setup.py`, `requirements.txt`
- **Purpose**: Wraps CUDA kernels in `AT_DISPATCH_FLOATING_TYPES` for dtype dispatch,
  adjusts the build setup, and bounds dependencies.
- **Upstream NOTICE**: none at base commit.

### `Hunyuan3D-2.1.patch`
- **Targets**: `hy3dpaint/textureGenPipeline.py`, `hy3dpaint/utils/simplify_mesh_utils.py`,
  `hy3dshape/hy3dshape/pipelines.py`
- **Purpose**: Points the texture pipeline at the local `hunyuanpaintpbr` path and
  switches mesh simplification to quadric decimation with an explicit face-count target.
- **Upstream NOTICE**: `Notice.txt` (121 lines) exists at the base commit and **is
  preserved verbatim** at
  [`third_party_notices/Hunyuan3D-2.1-NOTICE.txt`](third_party_notices/Hunyuan3D-2.1-NOTICE.txt)
  (SHA-256 `ffccf6b5…0350`), referenced from
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### `ml-depth-pro.patch`
- **Target**: `pyproject.toml`
- **Purpose**: Sets the Python target to 3.10 (`pythonVersion`, `target-version`) to
  match the `cdc` environment.
- **Upstream NOTICE**: none at base commit.

### `PriorDepthAnything.patch`
- **Targets**: `prior_depth_anything/__init__.py`, `depth_completion.py`, `plugin.py`,
  `requirements.txt`, `setup.py`
- **Purpose**: Adds a `coarse_only` mode and `args` plumbing through the completion
  path, plus packaging adjustments.
- **Upstream NOTICE**: none at base commit.

### `splatfacto_depth_loss.patch`
- **Target**: `nerfstudio/models/splatfacto.py`
- **Purpose**: Adds env-var-gated L1 depth supervision. With `NERFSTUDIO_DEPTH_LOSS=1`,
  splatfacto reads per-frame depth maps from `NERFSTUDIO_DEPTH_DIR` and adds
  `NERFSTUDIO_DEPTH_LOSS_MULT * |pred_depth - gt_depth|` to the loss. Documented in
  [`auto_bg_reconstruction/README.md`](scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/README.md) §4.2.
- **Note**: This is the only patch carrying an explicit `Subject:`/`Target:` header.
- **Upstream NOTICE**: none at base commit.

### `void-model.patch`
- **Target**: `inference/cogvideox_fun/make_warped_noise.py`
- **Purpose**: Prevents the `rp` dependency from invoking `sudo pip` for its
  auto-installs, which would otherwise raise an interactive password prompt in the
  middle of an unattended VOID pass.
- **Upstream NOTICE**: none at base commit.

## Open items

1. **Distribution rights for the three restricted patches.** Confirm that the Any6D
   academic-only license, the NVIDIA Source Code License (FoundationPose), and the
   Tencent Hunyuan 3D 2.1 Community License each permit NVIDIA to redistribute the
   source fragments carried in the corresponding patch files. If any does not, that
   patch must be replaced with a scripted transformation that carries no upstream
   source, or removed.
   <!-- TODO(SimFoundry): OSRB/Legal determination required. -->
