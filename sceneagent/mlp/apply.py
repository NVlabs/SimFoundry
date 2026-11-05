"""Load our physics prior and cast a section 3.4 vote.

The npz is trained in this tree. It is not a released Harvard checkpoint.
Inference runs the width-32 MLP on each in-mask Gaussian and averages the
33 outputs. ``log_mass`` and ``log_density`` are those averaged slots.
They are written with ``store_log_mass_density`` and do not pass through
``_positive_log``.
"""

from __future__ import annotations

import math

import numpy as np

from sceneagent.mlp.prior import (
    N_BODY,
    N_COLLISION,
    N_MATERIALS,
    OUTPUT_DIM,
    SLOT_BODY,
    SLOT_COLLISION,
    SLOT_DEFORMABILITY,
    SLOT_LOG_DENSITY,
    SLOT_LOG_MASS,
    SLOT_MATERIAL,
    SLOT_RIGIDITY,
    PhysicsPrior,
    mean_in_mask,
)
from sceneagent.physics import build_physics_record, proposals_for_prior, store_log_mass_density, vote_material
from sceneagent.taxonomy import BODY_TYPES, COLLISION_PRIMITIVES, MATERIAL_CLASSES


def decode_mean(mean):
    """Read class names and numeric heads from one in-mask mean.

    Rigidity and deformability are sigmoid values in (0, 1). Mass and density
    slots are already natural logs. The proposal weight is the softmax
    probability of the material class. Slots 28–32 are ignored.
    """

    mean = np.asarray(mean, dtype=np.float64)
    if mean.shape != (OUTPUT_DIM,):
        raise ValueError(f"in-mask mean must have shape ({OUTPUT_DIM},)")
    if not np.isfinite(mean[:28]).all():
        raise ValueError("named prior outputs are not finite")
    material_index = int(np.argmax(mean[SLOT_MATERIAL]))
    body_index = int(np.argmax(mean[SLOT_BODY]))
    collision_index = int(np.argmax(mean[SLOT_COLLISION]))
    probabilities = _softmax(mean[SLOT_MATERIAL])
    return {
        "material": MATERIAL_CLASSES[material_index],
        "body_type": BODY_TYPES[body_index],
        "collision": COLLISION_PRIMITIVES[collision_index],
        "log_mass": float(mean[SLOT_LOG_MASS]),
        "log_density": float(mean[SLOT_LOG_DENSITY]),
        "rigidity": _sigmoid(mean[SLOT_RIGIDITY]),
        "deformability": _sigmoid(mean[SLOT_DEFORMABILITY]),
        "weight": float(probabilities[material_index]),
        "material_probability": probabilities,
    }


def proposal_from_mean(mean):
    """``{source: physics_prior, class, weight}`` from an in-mask mean."""

    decoded = decode_mean(mean)
    return {
        "source": "physics_prior",
        "class": decoded["material"],
        "weight": decoded["weight"],
    }, decoded


def vote_loaded_prior(features, mask, *, vlm, keyword, checkpoint, name, prior=None):
    """Average the in-mask outputs and vote beside the VLM and keyword proposals.

    ``keyword`` is whatever the caller passes. This function does not look up
    ``_KEYWORD_RULES``. ``vlm`` is a proposal mapping with ``class`` and
    ``weight``. A tie in ``vote_material`` leaves the material class empty.
    """

    model = prior if prior is not None else PhysicsPrior.load(checkpoint)
    if model.w_out.shape != (32, 33):
        raise ValueError(f"w_out shape is {model.w_out.shape}, expected (32, 33)")
    if len(MATERIAL_CLASSES) != N_MATERIALS or len(BODY_TYPES) != N_BODY or len(COLLISION_PRIMITIVES) != N_COLLISION:
        raise ValueError("taxonomy counts do not match the prior heads")
    averaged = mean_in_mask(model.forward(features), mask)
    proposal, decoded = proposal_from_mean(averaged)
    proposals = proposals_for_prior(proposal, vlm=vlm, keyword=keyword)
    winner, scores = vote_material(proposals)
    record = build_physics_record(
        name,
        proposals,
        numerical={
            "rigidity": decoded["rigidity"],
            "deformability": decoded["deformability"],
        },
        body_type=decoded["body_type"],
        collision=decoded["collision"],
    )
    record = store_log_mass_density(record, decoded["log_mass"], decoded["log_density"])
    if record.get("mass_kg") is not None or record.get("density_kg_m3") is not None:
        raise ValueError("log slots were also stored as kilograms")
    return {
        "proposal": proposal,
        "decoded": {key: value for key, value in decoded.items() if key != "material_probability"},
        "winner": winner,
        "scores": scores,
        "record": record,
        "mean": averaged,
    }


def _softmax(logits):
    shifted = np.asarray(logits, dtype=np.float64)
    shifted = shifted - np.max(shifted)
    exponential = np.exp(shifted)
    total = float(exponential.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("material softmax is not finite")
    return exponential / total


def _sigmoid(value):
    value = float(value)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
