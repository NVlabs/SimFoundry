"""User-requested SAM 2. This is not the paper's SAM vit_h and not section 3.3.

Section 3.3 names SAM, every tenth view, and the vit_h checkpoint this tree
already ran. The user asked for SAM 2 as its own path. A run here does not
mark the SAM-mask row or section 3.3 done.
"""

from __future__ import annotations

import os
from pathlib import Path

LABEL = "user-requested SAM 2, not the paper's SAM vit_h"
PAPER_SAM = "vit_h"


def _base(status, reason, weights_loaded=False):
    return {
        "label": LABEL,
        "tool": "SAM 2",
        "paper_sam": PAPER_SAM,
        "paper_section": None,
        "not_section_3_3": True,
        "closes_section_3_3": False,
        "not_paper_sam_vit_h": True,
        "status": status,
        "weights_loaded": weights_loaded,
        "reason": reason,
    }


def _checkpoint(explicit):
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    env = os.environ.get("SAM2_CHECKPOINT")
    if env and Path(env).is_file():
        return Path(env)
    for name in ("sam2_hiera_large.pt", "checkpoints/sam2_hiera_large.pt"):
        candidate = Path(name)
        if candidate.is_file():
            return candidate
    return None


def sam2_status(weights_path=None, gpu_busy=False):
    if gpu_busy:
        return _base(
            "not_run_gpu_busy",
            "Quebec GPU is busy. SAM 2 was not installed and was not run. "
            "This is not the paper's SAM vit_h.",
        )
    weights = _checkpoint(weights_path)
    try:
        import sam2  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    if not installed or weights is None:
        return _base(
            "not_run",
            "SAM 2 weights are not loaded. This path is not section 3.3 and not SAM vit_h.",
        )
    return _base("weights_present", "SAM 2 checkpoint is on disk. Inference was not started.", True)


def run_sam2(image_path, box=None, config_name=None, weights_path=None, gpu_busy=False):
    """Run SAM 2 image prediction when the package and a checkpoint exist.

    ``box`` is xyxy in pixels. Without a box the call does not invent a mask.
    """
    if gpu_busy:
        return sam2_status(gpu_busy=True)
    image = Path(image_path) if image_path else None
    if image is None or not image.is_file():
        body = sam2_status(weights_path)
        body["status"] = "not_run"
        body["reason"] = "No image. " + LABEL
        return body
    try:
        import numpy as np
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        return sam2_status(weights_path)
    weights = _checkpoint(weights_path)
    if weights is None:
        body = sam2_status(weights_path)
        body["reason"] = "SAM 2 imported but no checkpoint file was found. No mask was invented."
        return body
    if box is None:
        body = sam2_status(weights_path)
        body["status"] = "not_run"
        body["weights_loaded"] = True
        body["reason"] = "A checkpoint is present and no box was supplied, so SAM 2 did not emit a mask."
        return body
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = config_name or os.environ.get("SAM2_CONFIG") or "configs/sam2/sam2_hiera_l.yaml"
    model = build_sam2(config, str(weights), device=device)
    predictor = SAM2ImagePredictor(model)
    from PIL import Image

    frame = np.array(Image.open(image).convert("RGB"))
    predictor.set_image(frame)
    masks, scores, _logits = predictor.predict(
        box=np.asarray(box, dtype=np.float32),
        multimask_output=False,
    )
    return {
        "label": LABEL,
        "tool": "SAM 2",
        "paper_sam": PAPER_SAM,
        "paper_section": None,
        "not_section_3_3": True,
        "closes_section_3_3": False,
        "not_paper_sam_vit_h": True,
        "status": "ran",
        "weights_loaded": True,
        "device": device,
        "mask_count": int(len(masks)),
        "scores": [float(score) for score in scores],
        "reason": "SAM 2 ran. This is not the paper's SAM vit_h and does not close section 3.3.",
    }
