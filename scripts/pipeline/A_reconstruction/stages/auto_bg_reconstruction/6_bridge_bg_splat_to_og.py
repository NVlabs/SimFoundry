# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compute the rigid transform from a DA3-trained splat's world frame to the OG
world used by the main pipeline (stage 4), and emit it as a pose sidecar next
to the splat — *without* rotating the PLY.

Why no per-gaussian rotation? nerfstudio's PLY stores quats in f32; rotating
every gaussian's quat accumulates roundoff during the per-element re-
normalization, and the SH coefficients attached to each gaussian are *not*
re-rotated, so view-dependent shading drifts. Applying the transform as a
single rigid prim pose at OG load time preserves the trained state bit-for-bit.

The composite transform is:

    Same-scene (default):
        M_splat→og = step4_to_og @ image_<N>_cam2world @ ext_src[N]

    Cross-scene (--bridge-mode umeyama):
        Fit Sim(3) from source-DA3 to target-DA3 camera centers over the
        common src frames, then:
            M_splat→og = step4_to_og @ image_<N>_cam2world @ ext_tgt[N] @ Sim3

Convention note: DA3 extrinsics are world2cam (OpenCV / COLMAP), so the
trailing `ext[N]` is *not* inverted — `ext[N] @ p_world = p_camN` is exactly
the map we want at the anchor.

Inputs:
  --in-ply       trained splat in DA3 world
  --out-ply      same path as --in-ply in the canonical recipe — the splat
                 PLY is rewritten in place (sidecar JSON written next to it).
                 Pass a different path only if you want a verbatim copy + the
                 sidecar elsewhere.
  --src-npz      DA3 NPZ that the splat was trained against (defines that
                 DA3 world). Defaults to Data/<scene>/s2_da/da/exports/npz/
                 results.npz.
  --bridge-mode  single (same-scene; default) | umeyama (cross-scene) | auto
  --target-scene-name  scene whose s4_frame defines OG world (default: --scene-name)

Outputs:
  <out_ply>            byte-identical copy of <in-ply> (skipped when
                       --out-ply == --in-ply; the canonical case).
  <out_ply>.pose.json  {pos, ori_xyzw, scale, M_src_to_og} for build_og_scene_assets.py

Reads config from scripts/cfg/auto_bg.yaml (Hydra), section `s6_bridge`.
Run from the simfoundry env (Hydra override syntax):
  mamba run -n simfoundry python \\
      scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/6_bridge_bg_splat_to_og.py \\
      scene_name=quillen_table
Per-stage values can be overridden directly, e.g. `s6_bridge.bridge_mode=umeyama`.
"""
import json
import logging
import shutil
import sys
from pathlib import Path

import hydra
import numpy as np
from scipy.spatial.transform import Rotation

from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir
from simfoundry.utils.transform_utils import camera_centers_from_world2cam, umeyama_alignment

# cd to scripts/cfg so the cfg's `${root_dir}` (= ../../Data) resolves to repo/Data,
# matching the sibling A_reconstruction stages.
bootstrap_hydra_workdir(__file__)

from simfoundry import CFG_DIR  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[5]


def _src_frames_for_scene(scene: str) -> list[int]:
    """Return the 1-based src frame indices for each row of a scene's s2_da extrinsics,
    parsed from the filenames in s1_video/frames_subsampled_<N>/.
    """
    s1_dir = REPO_ROOT / "Data" / scene / "s1_video"
    cands = sorted(p for p in s1_dir.iterdir() if p.is_dir() and p.name.startswith("frames_subsampled_"))
    if not cands:
        sys.exit(f"No frames_subsampled_<N> dir in {s1_dir}")
    sub = cands[-1]   # use the largest if multiple (most recent run)
    frames = sorted(sub.glob("frame_*.png"))
    return [int(p.stem.split("_")[1]) for p in frames]


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bridge_bg_splat_to_og")


@hydra.main(config_name="auto_bg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sec = cfg.s6_bridge

    if not cfg.scene_name:
        sys.exit("scene_name is required (no scene_name set in cfg either)")

    scene = cfg.scene_name
    target_scene = sec.target_scene_name or scene
    s4_dir = REPO_ROOT / "Data" / target_scene / "s4_frame"
    s2_npz = Path(sec.src_npz).resolve() if sec.src_npz else \
             REPO_ROOT / "Data" / scene / "s2_da" / "da" / "exports" / "npz" / "results.npz"
    logger.info("Source DA3 NPZ (splat's world frame): %s", s2_npz)
    if not Path(s2_npz).exists():
        sys.exit(f"missing source DA3 npz (run stage 2b (2b_run_da.py), or pass s6_bridge.src_npz=): {s2_npz}")

    # Step4 anchor: use TARGET scene's image_<N>_cam2world.npy (defines the OG world we want)
    c2w_files = list(s4_dir.glob("image_*_cam2world.npy"))
    if len(c2w_files) != 1:
        sys.exit(f"Expected exactly one image_*_cam2world.npy in {s4_dir}, found {len(c2w_files)}")
    target_idx = int(c2w_files[0].stem.split("_")[1])
    image_c2w = np.load(c2w_files[0]).astype(np.float64)
    logger.info("Target step4 anchor: %s (image_%d_cam2world)", c2w_files[0].name, target_idx)

    step4_to_og_path = s4_dir / "step4_to_og_tf.npy"
    if step4_to_og_path.exists():
        step4_to_og = np.load(step4_to_og_path).astype(np.float64)
    else:
        logger.info("step4_to_og_tf.npy missing — treating step4 as OG world (identity).")
        step4_to_og = np.eye(4, dtype=np.float64)

    ext_src = np.load(s2_npz)["extrinsics"].astype(np.float64)   # (N, 3, 4) cam2world

    # Decide bridge mode
    mode = sec.bridge_mode
    if mode == "auto":
        mode = "single" if scene == target_scene else "umeyama"
    logger.info("Bridge mode: %s", mode)

    scale = 1.0   # Sim(3) scale written to the pose sidecar for OG to apply (1.0 unless cross-scene umeyama)

    if mode == "single":
        if sec.anchor_row is not None:
            anchor_row = sec.anchor_row
        elif scene == target_scene:
            anchor_row = target_idx
        else:
            sys.exit("Cross-scene + bridge-mode=single: --anchor-row is required.")
        if anchor_row >= ext_src.shape[0]:
            sys.exit(f"anchor_row={anchor_row} out of range for {ext_src.shape[0]} extrinsics")
        logger.info("Splat anchor row in %s/s2_da extrinsics: %d", scene, anchor_row)

        # ext is w2c: maps DA3 world -> cam-N frame. NO inverse.
        P_anchor_w2c = np.eye(4, dtype=np.float64)
        P_anchor_w2c[:3, :4] = ext_src[anchor_row]
        M_src_to_step4 = image_c2w @ P_anchor_w2c
        M_src_to_og = step4_to_og @ M_src_to_step4

    elif mode == "umeyama":
        # Fit Sim(3) from source DA3 -> target DA3 over the common src frames.
        src_frames = _src_frames_for_scene(scene)
        tgt_frames = _src_frames_for_scene(target_scene)
        if len(src_frames) != ext_src.shape[0]:
            sys.exit(f"src frame count {len(src_frames)} != extrinsics rows {ext_src.shape[0]}")
        s2_npz_tgt = REPO_ROOT / "Data" / target_scene / "s2_da" / "da" / "exports" / "npz" / "results.npz"
        ext_tgt = np.load(s2_npz_tgt)["extrinsics"].astype(np.float64)
        if len(tgt_frames) != ext_tgt.shape[0]:
            sys.exit(f"tgt frame count {len(tgt_frames)} != extrinsics rows {ext_tgt.shape[0]}")

        src_lookup = {f: i for i, f in enumerate(src_frames)}
        tgt_lookup = {f: i for i, f in enumerate(tgt_frames)}
        common = sorted(set(src_lookup) & set(tgt_lookup))
        logger.info("Common src frames between %s and %s: %s", scene, target_scene, common)
        if len(common) < 3:
            sys.exit(f"Need >=3 common src frames for Sim(3) fit, got {len(common)}: {common}")

        # Camera POSITIONS in each DA3 world (under w2c convention: pos = -R.T @ t).
        P_src = camera_centers_from_world2cam(np.stack([ext_src[src_lookup[f]] for f in common]))
        P_tgt = camera_centers_from_world2cam(np.stack([ext_tgt[tgt_lookup[f]] for f in common]))

        T_src_to_tgt_DA3, scale = umeyama_alignment(P_src, P_tgt)
        pred = (T_src_to_tgt_DA3[:3, :3] @ P_src.T + T_src_to_tgt_DA3[:3, 3:4]).T
        res = np.linalg.norm(pred - P_tgt, axis=1)
        logger.info("Umeyama Sim(3) %s_DA3 -> %s_DA3: scale=%.4f, mean res=%.4f m, max res=%.4f m",
                    scene, target_scene, scale, res.mean(), res.max())

        # Compose: source_DA3 -> target_DA3 -> target_cam_anchor -> step4 -> OG.
        # target's cam_anchor reached by w2c: ext_tgt[target_idx] @ p_target_DA3.
        P_anchor_tgt_w2c = np.eye(4, dtype=np.float64)
        P_anchor_tgt_w2c[:3, :4] = ext_tgt[target_idx]
        M_tgtDA3_to_step4 = image_c2w @ P_anchor_tgt_w2c
        M_src_to_og = step4_to_og @ M_tgtDA3_to_step4 @ T_src_to_tgt_DA3

    else:
        sys.exit(f"Unknown bridge mode: {mode}")

    logger.info("M_src_to_og (%s -> %s OG):\n%s", scene, target_scene, M_src_to_og)
    logger.info("Rotation block det = %.4f (should be ~%.4f^3 = %.4f)",
                np.linalg.det(M_src_to_og[:3, :3]), scale, scale ** 3)

    in_ply = Path(sec.in_ply)
    out_ply = Path(sec.out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)

    # Don't bake M into the PLY. Copy the trained splat verbatim and emit a
    # sidecar pose JSON so the loader can apply M as a single rigid prim transform.
    if out_ply.resolve() != in_ply.resolve():
        if out_ply.exists() or out_ply.is_symlink():
            out_ply.unlink()
        shutil.copy2(in_ply, out_ply)
        logger.info("Copied unrotated PLY %s -> %s", in_ply, out_ply)
    else:
        logger.info("in_ply and out_ply are the same path; leaving PLY untouched.")

    # Decompose M = T * R (scale folded into R for Sim(3); pure rotation for SE(3)).
    R_with_scale = M_src_to_og[:3, :3].astype(np.float64)
    t = M_src_to_og[:3, 3].astype(np.float64)
    det_R = float(np.linalg.det(R_with_scale))
    if det_R <= 0:
        sys.exit(f"M_src_to_og has non-positive determinant ({det_R}); cannot decompose to pos/ori/scale.")
    # Uniform scale assumption (true for single-anchor SE(3) where scale=1,
    # and for Umeyama which returns a Sim(3) with one isotropic scale).
    s = det_R ** (1.0 / 3.0)
    R_pure = R_with_scale / s
    # Re-orthogonalize to suppress accumulated float64 error.
    U, _, Vt = np.linalg.svd(R_pure)
    R_pure = U @ Vt
    if np.linalg.det(R_pure) < 0:
        # Flip the smallest singular vector to preserve right-handedness.
        U[:, -1] *= -1
        R_pure = U @ Vt
    q_xyzw = Rotation.from_matrix(R_pure).as_quat().tolist()
    pos = t.tolist()

    sidecar = out_ply.with_name(out_ply.name + ".pose.json")
    sidecar_payload = {
        "pos": pos,
        "ori_xyzw": q_xyzw,
        "scale": float(s),
        "frame": {
            "source": f"{scene} DA3 (s2_da extrinsics)",
            "target": f"{target_scene} OG world",
        },
        "bridge_mode": mode,
        "M_src_to_og": M_src_to_og.tolist(),
    }
    sidecar.write_text(json.dumps(sidecar_payload, indent=2))
    logger.info(
        "Wrote pose sidecar -> %s  pos=%s  ori_xyzw=%s  scale=%.6f",
        sidecar, pos, q_xyzw, s,
    )


if __name__ == "__main__":
    main()
