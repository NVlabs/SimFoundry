"""Provisional physics record (SceneAgent section 3.4).

The paper's per-Gaussian MLP is trained on VLM labels from GSO, YCB, and ABO.
Those weights are not released. Phase 1 fuses the sources the paper names
(an optional physics-prior proposal, a VLM proposal, and a keyword prior)
with the published weighted vote:

    y* = argmax_y  sum_v  w_v * 1[y_v = y]

A proposal whose class is ``unknown`` has its weight multiplied by 0.4.
A tie returns no class. The keyword proposal is passed in by the caller.
Mass and density, when a caller supplies them in kilograms, are stored as
natural logs. A value that is already a logarithm is written with
``store_log_mass_density`` and does not pass through ``_positive_log``.

Young's modulus, Poisson's ratio, and the rendering material are looked up
from the material class in the paper. That table is not published, so phase 1
leaves the fields unresolved instead of inventing coefficients.

An object becomes a soft body when deformability is at least 0.65. The paper
does not define deformability as ``1 - rigidity``, so rigidity alone does not
set the body type.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from sceneagent.taxonomy import require_body, require_collision, require_material

UNKNOWN_WEIGHT_SCALE = 0.4
SOFT_BODY_DEFORMABILITY = 0.65
UNPUBLISHED_LOOKUPS = ("youngs_modulus", "poisson_ratio", "rendering_material")


def scaled_weight(material_class, weight):
    require_material(material_class)
    weight = float(weight)
    if weight < 0:
        raise ValueError("proposal weight must be non-negative")
    if material_class == "unknown":
        return weight * UNKNOWN_WEIGHT_SCALE
    return weight


def vote_material(proposals):
    """Weighted material vote.

    Each proposal is a mapping with ``class`` and ``weight``. An unknown
    class is scaled by 0.4. When two or more classes share the highest
    score, the vote returns no class (``None``).
    """
    if not proposals:
        raise ValueError("material vote needs at least one proposal")
    scores = {}
    order = []
    for proposal in proposals:
        material = require_material(proposal["class"])
        weight = scaled_weight(material, proposal["weight"])
        if material not in scores:
            order.append(material)
            scores[material] = 0.0
        scores[material] += weight
    top = max(scores[name] for name in order)
    winners = [
        name
        for name in order
        if math.isclose(scores[name], top, rel_tol=0.0, abs_tol=1e-9)
    ]
    if len(winners) != 1:
        return None, scores
    return winners[0], scores


def proposals_for_prior(prior, vlm=None, keyword=None):
    """Assemble voters for the physics prior.

    ``keyword`` is included only when the caller passes it. This function
    does not consult ``taxonomy._KEYWORD_RULES`` and does not call
    ``keyword_material``.
    """
    proposals = [_copy_proposal(prior, "physics prior")]
    if vlm is not None:
        proposals.append(_copy_proposal(vlm, "VLM"))
    if keyword is not None:
        proposals.append(_copy_proposal(keyword, "keyword"))
    return proposals


def _copy_proposal(proposal, role):
    if not isinstance(proposal, Mapping):
        raise ValueError(f"{role} proposal must be a mapping")
    if "class" not in proposal or "weight" not in proposal:
        raise ValueError(f"{role} proposal needs class and weight")
    require_material(proposal["class"])
    return dict(proposal)


def resolve_body_type(deformability, explicit=None):
    """Apply the published deformability rule, else an explicit proposal.

    ``explicit`` is a body-type string from a caller (the MLP head or a VLM).
    The deformability rule wins when it fires, because the paper says an object
    becomes a soft body at that threshold.
    """
    if deformability is not None:
        deformability = float(deformability)
        if not 0.0 <= deformability <= 1.0:
            raise ValueError("deformability must lie in [0, 1]")
        if deformability >= SOFT_BODY_DEFORMABILITY:
            return "soft deformable", "deformability_at_least_0.65"
    if explicit:
        return require_body(explicit), "explicit_proposal"
    return None, "unresolved"


def _positive_log(name, value):
    if value is None:
        return None
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive to take a logarithm")
    return math.log(value)


def store_log_mass_density(record, log_mass, log_density):
    """Write mass and density that are already natural logs.

    Does not call ``_positive_log``. A second call with the stored values
    leaves them unchanged.
    """
    log_mass = float(log_mass)
    log_density = float(log_density)
    if not math.isfinite(log_mass) or not math.isfinite(log_density):
        raise ValueError("log mass and log density must be finite")
    updated = dict(record)
    updated["log_mass"] = log_mass
    updated["log_density"] = log_density
    return updated


def build_physics_record(name, proposals, numerical=None, body_type=None, collision=None):
    """Fuse proposals into one object physics record.

    ``numerical`` may contain friction_static, friction_dynamic, restitution,
    mass_kg, density_kg_m3, rigidity, and deformability. Missing keys stay
    null. ``mass_kg`` and ``density_kg_m3`` are kilograms and are logged once
    here. An already-logged prior value is written with
    ``store_log_mass_density`` instead. Coefficients are not filled from a
    private table.
    """
    numerical = dict(numerical or {})
    winner, scores = vote_material(proposals)
    rigidity = numerical.get("rigidity")
    if rigidity is not None:
        rigidity = float(rigidity)
        if not 0.0 <= rigidity <= 1.0:
            raise ValueError("rigidity must lie in [0, 1]")
    deformability = numerical.get("deformability")
    resolved_body, body_source = resolve_body_type(deformability, body_type)
    resolved_collision = require_collision(collision) if collision else None
    mass = numerical.get("mass_kg")
    density = numerical.get("density_kg_m3")
    return {
        "object": name,
        "material_class": winner,
        "material_scores": scores,
        "material_sources": [
            {
                "source": proposal.get("source", "unspecified"),
                "class": proposal["class"],
                "weight": float(proposal["weight"]),
                "weight_after_unknown_scale": scaled_weight(proposal["class"], proposal["weight"]),
            }
            for proposal in proposals
        ],
        "body_type": resolved_body,
        "body_type_source": body_source,
        "collision_primitive": resolved_collision,
        "friction_static": numerical.get("friction_static"),
        "friction_dynamic": numerical.get("friction_dynamic"),
        "restitution": numerical.get("restitution"),
        "mass_kg": None if mass is None else float(mass),
        "log_mass": _positive_log("mass_kg", mass),
        "density_kg_m3": None if density is None else float(density),
        "log_density": _positive_log("density_kg_m3", density),
        "rigidity": rigidity,
        "deformability": None if deformability is None else float(deformability),
        "soft_body_threshold": SOFT_BODY_DEFORMABILITY,
        "unpublished_class_lookups": list(UNPUBLISHED_LOOKUPS),
        "note": (
            "Provisional prior for domain randomization. "
            "SceneAgent section 3.4 treats these values as approximate, not as measured parameters."
        ),
    }
