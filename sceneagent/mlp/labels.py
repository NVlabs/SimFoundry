"""Parser for one VLM physics label.

The schema is material, body_type, collision, mass_kg, density_kg_m3,
rigidity, and deformability. Mass and density are stored as natural logs
once, from the kilogram values. Friction and restitution, when the label
includes them, stay beside the record. They are not written into slots
28–32, and those slots receive no invented targets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from sceneagent.mlp.prior import OUTPUT_DIM, SLOT_DEFORMABILITY, SLOT_LOG_DENSITY, SLOT_LOG_MASS, SLOT_RIGIDITY, SLOT_UNPUBLISHED
from sceneagent.taxonomy import require_body, require_collision, require_material

REQUIRED_LABEL_KEYS = (
    "material",
    "body_type",
    "collision",
    "mass_kg",
    "density_kg_m3",
    "rigidity",
    "deformability",
)

_BESIDE_KEYS = ("friction_static", "friction_dynamic", "restitution")


@dataclass(frozen=True)
class ParsedLabel:
    """One label after a single logarithm on mass and density."""

    material: str
    body_type: str
    collision: str
    mass_kg: float
    density_kg_m3: float
    log_mass: float
    log_density: float
    rigidity: float
    deformability: float
    friction_static: float | None = None
    friction_dynamic: float | None = None
    restitution: float | None = None

    def numeric_target(self):
        """A length-33 target. Slots 28–32 stay NaN. Class slots stay NaN.

        The classification heads are trained from ``material``, ``body_type``,
        and ``collision``, not from invented numeric targets.
        """

        target = np.full(OUTPUT_DIM, np.nan, dtype=np.float64)
        target[SLOT_LOG_MASS] = self.log_mass
        target[SLOT_LOG_DENSITY] = self.log_density
        target[SLOT_RIGIDITY] = self.rigidity
        target[SLOT_DEFORMABILITY] = self.deformability
        return target


def parse_label(payload):
    """Parse one label mapping. Mass and density are logged once."""

    if not isinstance(payload, Mapping):
        raise ValueError("label must be a mapping")
    missing = [key for key in REQUIRED_LABEL_KEYS if payload.get(key) is None]
    if missing:
        raise ValueError("label is missing " + ", ".join(missing))
    material = require_material(str(payload["material"]))
    body_type = require_body(str(payload["body_type"]))
    collision = require_collision(str(payload["collision"]))
    mass_kg = float(payload["mass_kg"])
    density_kg_m3 = float(payload["density_kg_m3"])
    if mass_kg <= 0.0 or density_kg_m3 <= 0.0:
        raise ValueError("mass_kg and density_kg_m3 must be positive")
    rigidity = float(payload["rigidity"])
    deformability = float(payload["deformability"])
    if not 0.0 <= rigidity <= 1.0:
        raise ValueError("rigidity must lie in [0, 1]")
    if not 0.0 <= deformability <= 1.0:
        raise ValueError("deformability must lie in [0, 1]")
    beside = {key: _optional_float(payload, key) for key in _BESIDE_KEYS}
    parsed = ParsedLabel(
        material=material,
        body_type=body_type,
        collision=collision,
        mass_kg=mass_kg,
        density_kg_m3=density_kg_m3,
        log_mass=math.log(mass_kg),
        log_density=math.log(density_kg_m3),
        rigidity=rigidity,
        deformability=deformability,
        friction_static=beside["friction_static"],
        friction_dynamic=beside["friction_dynamic"],
        restitution=beside["restitution"],
    )
    target = parsed.numeric_target()
    if not np.isnan(target[SLOT_UNPUBLISHED]).all():
        raise ValueError("slots 28-32 must stay empty")
    return parsed


def _optional_float(payload, key):
    if key not in payload or payload[key] is None:
        return None
    value = float(payload[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value
