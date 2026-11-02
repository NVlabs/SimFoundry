"""Width-32 physics prior, checkpoint loader, and masked loss.

The network is two tanh hidden layers of width 32 and a linear head.
``w_out`` is ``(32, 33)``. Inference runs the MLP on each in-mask Gaussian
and averages the 33 outputs, including the logits. Averaging the 912 inputs
and running the MLP once is a different function.

Slots 28–32 are unnamed. The loss does not read them. ``PhysicsPrior.load``
refuses an old 8-class checkpoint and a wrong width.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from sceneagent.mlp.contract import PRIOR_DIM, PriorContractError
from sceneagent.taxonomy import BODY_TYPES, COLLISION_PRIMITIVES, MATERIAL_CLASSES

HIDDEN = 32
OUTPUT_DIM = 33
N_MATERIALS = len(MATERIAL_CLASSES)
N_BODY = len(BODY_TYPES)
N_COLLISION = len(COLLISION_PRIMITIVES)
N_NUMERIC = 9
SLOT_MATERIAL = slice(0, N_MATERIALS)
SLOT_BODY = slice(N_MATERIALS, N_MATERIALS + N_BODY)
SLOT_COLLISION = slice(N_MATERIALS + N_BODY, N_MATERIALS + N_BODY + N_COLLISION)
SLOT_LOG_MASS = 24
SLOT_LOG_DENSITY = 25
SLOT_RIGIDITY = 26
SLOT_DEFORMABILITY = 27
SLOT_UNPUBLISHED = slice(28, 33)

_CHECKPOINT_WEIGHTS = ("w1", "b1", "w2", "b2", "w_out", "b_out")


def loss_slot_mask():
    """1 on the named heads, 0 on slots 28–32."""

    mask = np.ones(OUTPUT_DIM, dtype=np.float64)
    mask[SLOT_UNPUBLISHED] = 0.0
    return mask


def mean_in_mask(outputs, mask):
    """Arithmetic mean of the raw 33-d outputs on the in-mask rows.

    The mean is over logits as well as the numeric slots. It is not a mean
    of probabilities and not an average of the 912 inputs.
    """

    outputs = np.asarray(outputs, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if outputs.ndim != 2 or outputs.shape[1] != OUTPUT_DIM:
        raise PriorContractError(f"in-mask mean expects outputs of shape (N, {OUTPUT_DIM})")
    if mask.shape != (outputs.shape[0],):
        raise PriorContractError("mask length must match the Gaussian count")
    if not mask.any():
        raise PriorContractError("in-mask mean needs at least one Gaussian")
    selected = outputs[mask]
    if not np.isfinite(selected).all():
        raise PriorContractError("in-mask outputs are not finite; missing values are not filled")
    return selected.mean(axis=0)


def masked_prior_loss(
    pred,
    *,
    material,
    body,
    collision,
    log_mass,
    log_density,
    rigidity,
    deformability,
):
    """Loss on the named heads of one in-mask mean.

    Classification loss on the material, body, and collision logits.
    Squared error on ``log_mass`` and ``log_density`` as logarithms, and on
    the sigmoid of rigidity and deformability. Slots 28–32 are not read.
    """

    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != (OUTPUT_DIM,):
        raise PriorContractError(f"loss expects the in-mask mean of shape ({OUTPUT_DIM},)")
    named = pred[:SLOT_UNPUBLISHED.start]
    material_index = _class_index(material, MATERIAL_CLASSES, "material")
    body_index = _class_index(body, BODY_TYPES, "body")
    collision_index = _class_index(collision, COLLISION_PRIMITIVES, "collision")
    log_mass = float(log_mass)
    log_density = float(log_density)
    rigidity = float(rigidity)
    deformability = float(deformability)
    if not 0.0 <= rigidity <= 1.0 or not 0.0 <= deformability <= 1.0:
        raise PriorContractError("rigidity and deformability targets lie in [0, 1]")
    loss = (
        _cross_entropy(named[SLOT_MATERIAL], material_index)
        + _cross_entropy(named[SLOT_BODY], body_index)
        + _cross_entropy(named[SLOT_COLLISION], collision_index)
        + (named[SLOT_LOG_MASS] - log_mass) ** 2
        + (named[SLOT_LOG_DENSITY] - log_density) ** 2
        + (_sigmoid_scalar(named[SLOT_RIGIDITY]) - rigidity) ** 2
        + (_sigmoid_scalar(named[SLOT_DEFORMABILITY]) - deformability) ** 2
    )
    return float(loss)


class PhysicsPrior:
    """Two hidden layers of width 32, tanh, then a linear map to 33 outputs."""

    def __init__(self, w1, b1, w2, b2, w_out, b_out):
        self.w1 = np.asarray(w1, dtype=np.float64)
        self.b1 = np.asarray(b1, dtype=np.float64)
        self.w2 = np.asarray(w2, dtype=np.float64)
        self.b2 = np.asarray(b2, dtype=np.float64)
        self.w_out = np.asarray(w_out, dtype=np.float64)
        self.b_out = np.asarray(b_out, dtype=np.float64)
        _check_weight_shapes(self.w1, self.b1, self.w2, self.b2, self.w_out, self.b_out)

    @classmethod
    def initialize(cls, seed=0):
        rng = np.random.default_rng(seed)
        return cls(
            _glorot(rng, PRIOR_DIM, HIDDEN),
            np.zeros(HIDDEN, dtype=np.float64),
            _glorot(rng, HIDDEN, HIDDEN),
            np.zeros(HIDDEN, dtype=np.float64),
            _glorot(rng, HIDDEN, OUTPUT_DIM),
            np.zeros(OUTPUT_DIM, dtype=np.float64),
        )

    def forward(self, features):
        """Map ``(N, 912)`` to ``(N, 33)`` raw outputs. No activation on the head."""

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != PRIOR_DIM:
            raise PriorContractError(
                f"forward expects (N, {PRIOR_DIM}), got {tuple(features.shape)}"
            )
        if not np.isfinite(features).all():
            raise PriorContractError("missing feature values are not filled")
        hidden = np.tanh(features @ self.w1 + self.b1)
        hidden = np.tanh(hidden @ self.w2 + self.b2)
        outputs = hidden @ self.w_out + self.b_out
        if outputs.shape != (features.shape[0], OUTPUT_DIM):
            raise PriorContractError("forward did not return (N, 33)")
        return outputs

    def in_mask_mean(self, features, mask):
        """Forward every row, then average the in-mask 33-d outputs."""

        return mean_in_mask(self.forward(features), mask)

    def save(self, path):
        save_checkpoint(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w_out=self.w_out,
            b_out=self.b_out,
        )
        return Path(path)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            _refuse_old_or_narrow(data)
            weights = {name: np.asarray(data[name], dtype=np.float64) for name in _CHECKPOINT_WEIGHTS}
        return cls(**weights)


def save_checkpoint(
    path,
    *,
    w1,
    b1,
    w2,
    b2,
    w_out,
    b_out,
    n_materials=N_MATERIALS,
    n_body=N_BODY,
    n_collision=N_COLLISION,
    n_numeric=N_NUMERIC,
    materials=MATERIAL_CLASSES,
    body_types=BODY_TYPES,
    collision_primitives=COLLISION_PRIMITIVES,
):
    """Write the npz fields ``PhysicsPrior.load`` reads.

    Tests use this to build an 8-class file and a wrong-width file. A caller
    who wants a loadable checkpoint should use ``PhysicsPrior.save``.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        n_materials=np.int64(n_materials),
        n_body=np.int64(n_body),
        n_collision=np.int64(n_collision),
        n_numeric=np.int64(n_numeric),
        w1=np.asarray(w1, dtype=np.float64),
        b1=np.asarray(b1, dtype=np.float64),
        w2=np.asarray(w2, dtype=np.float64),
        b2=np.asarray(b2, dtype=np.float64),
        w_out=np.asarray(w_out, dtype=np.float64),
        b_out=np.asarray(b_out, dtype=np.float64),
        materials=np.asarray(list(materials), dtype="U64"),
        body_types=np.asarray(list(body_types), dtype="U64"),
        collision_primitives=np.asarray(list(collision_primitives), dtype="U64"),
    )
    return destination


def _refuse_old_or_narrow(data):
    for name in ("n_materials", "n_body", "n_collision", "n_numeric", *_CHECKPOINT_WEIGHTS, "materials", "body_types", "collision_primitives"):
        if name not in data.files:
            raise PriorContractError(f"checkpoint is missing {name}")
    n_materials = _scalar_int(data["n_materials"], "n_materials")
    materials = _string_tuple(data["materials"])
    if n_materials == 8 or len(materials) == 8:
        raise PriorContractError("refuse an old 8-class npz")
    if n_materials != N_MATERIALS or materials != MATERIAL_CLASSES:
        raise PriorContractError(
            f"n_materials must be {N_MATERIALS} in taxonomy order, got {n_materials}"
        )
    if _scalar_int(data["n_body"], "n_body") != N_BODY or _string_tuple(data["body_types"]) != BODY_TYPES:
        raise PriorContractError(f"n_body must be {N_BODY} in taxonomy order")
    if (
        _scalar_int(data["n_collision"], "n_collision") != N_COLLISION
        or _string_tuple(data["collision_primitives"]) != COLLISION_PRIMITIVES
    ):
        raise PriorContractError(f"n_collision must be {N_COLLISION} in taxonomy order")
    if _scalar_int(data["n_numeric"], "n_numeric") != N_NUMERIC:
        raise PriorContractError(f"n_numeric must be {N_NUMERIC}")
    _check_weight_shapes(
        data["w1"],
        data["b1"],
        data["w2"],
        data["b2"],
        data["w_out"],
        data["b_out"],
    )


def _check_weight_shapes(w1, b1, w2, b2, w_out, b_out):
    shapes = {
        "w1": (tuple(np.shape(w1)), (PRIOR_DIM, HIDDEN)),
        "b1": (tuple(np.shape(b1)), (HIDDEN,)),
        "w2": (tuple(np.shape(w2)), (HIDDEN, HIDDEN)),
        "b2": (tuple(np.shape(b2)), (HIDDEN,)),
        "w_out": (tuple(np.shape(w_out)), (HIDDEN, OUTPUT_DIM)),
        "b_out": (tuple(np.shape(b_out)), (OUTPUT_DIM,)),
    }
    for name, (actual, expected) in shapes.items():
        if actual != expected:
            raise PriorContractError(f"wrong width: {name} is {actual}, expected {expected}")


def _scalar_int(value, name):
    array = np.asarray(value)
    if array.size != 1:
        raise PriorContractError(f"{name} must be a scalar")
    return int(array.reshape(()))


def _string_tuple(value):
    return tuple(str(item) for item in np.asarray(value).tolist())


def _class_index(value, classes, role):
    if isinstance(value, str):
        if value not in classes:
            raise PriorContractError(f"unknown {role} class {value!r}")
        return classes.index(value)
    index = int(value)
    if index < 0 or index >= len(classes):
        raise PriorContractError(f"{role} index {index} is outside 0..{len(classes) - 1}")
    return index


def _cross_entropy(logits, index):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    return float(np.log(np.exp(shifted).sum()) - shifted[index])


def _sigmoid_scalar(value):
    value = float(value)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _glorot(rng, fan_in, fan_out):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))
