"""Website step 02. This is not paper section 3.3.

The project site says GroundingDINO + SAM assign semantic features to each
Gaussian. Section 3.3 uses SAM masks, OpenCLIP, DINOv2, and a LangSplat-style
association, and it says the two embeddings are not used to decide which
Gaussians belong to an object. This hook does not mark section 3.3 done.

SAM 2 lives in ``sceneagent.sam2_user``. It is not this website SAM and it is
not the paper's SAM vit_h.
"""

from __future__ import annotations

import os
from pathlib import Path

LABEL = "website step 02"
TOOL = "GroundingDINO"


def _base(status, reason, weights_loaded=False):
    return {
        "label": LABEL,
        "paper_section": None,
        "not_section_3_3": True,
        "closes_section_3_3": False,
        "tool": TOOL,
        "status": status,
        "weights_loaded": weights_loaded,
        "reason": reason,
    }


def _weights_file(explicit):
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    env = os.environ.get("GROUNDINGDINO_WEIGHTS")
    if env and Path(env).is_file():
        return Path(env)
    for name in ("groundingdino_swint_ogc.pth", "weights/groundingdino_swint_ogc.pth"):
        candidate = Path(name)
        if candidate.is_file():
            return candidate
    return None


def _config_file(explicit):
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    env = os.environ.get("GROUNDINGDINO_CONFIG")
    if env and Path(env).is_file():
        return Path(env)
    return None


def groundingdino_status(weights_path=None, gpu_busy=False):
    """Report the website-step-02 hook. Default is not_run without weights.

    ``gpu_busy=True`` records that the Quebec GPU was left alone. That status
    is not a section 3.3 result.
    """
    if gpu_busy:
        return _base(
            "not_run_gpu_busy",
            "Quebec GPU is busy. GroundingDINO was not installed and was not run. "
            "Website step 02 is not section 3.3.",
        )
    weights = _weights_file(weights_path)
    try:
        import groundingdino.util.inference  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    if not installed or weights is None:
        return _base(
            "not_run",
            "GroundingDINO weights are not loaded. Website step 02 is not "
            "section 3.3, and this hook does not complete that section.",
        )
    return _base("weights_present", "Weights are on disk. Inference was not started.", True)


def run_groundingdino(
    image_path,
    caption,
    config_path=None,
    weights_path=None,
    box_threshold=0.35,
    text_threshold=0.25,
    gpu_busy=False,
):
    """Run GroundingDINO when weights are installed. Otherwise do not invent boxes.

    The predict call is the IDEA-Research inference API. A busy GPU returns
    ``not_run_gpu_busy`` and does not load the checkpoint.
    """
    if gpu_busy:
        return groundingdino_status(gpu_busy=True)
    image = Path(image_path) if image_path else None
    if image is None or not image.is_file():
        body = groundingdino_status(weights_path)
        body["status"] = "not_run"
        body["reason"] = "No image. Website step 02 was not run."
        return body
    try:
        from groundingdino.util.inference import load_image, load_model, predict
    except ImportError:
        return groundingdino_status(weights_path)
    weights = _weights_file(weights_path)
    config = _config_file(config_path)
    if weights is None or config is None:
        body = groundingdino_status(weights_path)
        body["reason"] = (
            "GroundingDINO is importable but the config or checkpoint file is missing. "
            "No boxes were invented. Website step 02 is not section 3.3."
        )
        return body
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(config), str(weights), device=device)
    _image_source, tensor = load_image(str(image))
    boxes, logits, phrases = predict(
        model=model,
        image=tensor,
        caption=caption,
        box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        device=device,
    )
    return {
        "label": LABEL,
        "paper_section": None,
        "not_section_3_3": True,
        "closes_section_3_3": False,
        "tool": TOOL,
        "status": "ran",
        "weights_loaded": True,
        "device": device,
        "phrases": [str(phrase) for phrase in phrases],
        "boxes": boxes.detach().cpu().tolist(),
        "logits": logits.detach().cpu().tolist(),
        "reason": "GroundingDINO ran as website step 02. This does not close section 3.3.",
    }
