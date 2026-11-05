"""Stratified sampling, masked training step, and the section 3.4 vote.

No Gemini calls and no downloaded meshes. Tensors in the training test are fake.
"""

import numpy as np
import pytest

from sceneagent.mlp.apply import vote_loaded_prior
from sceneagent.mlp.prior import PhysicsPrior
from sceneagent.mlp.stratum import (
    guess_material,
    is_near_duplicate,
    labels_should_stop,
    order_bucket,
)
from sceneagent.mlp.train_prior import _Adam, holdout_ids, loss_and_gradients
from sceneagent.taxonomy import MATERIAL_CLASSES


def test_guess_and_near_duplicate():
    assert guess_material("tomato soup can") == "metal"
    assert guess_material("cracker box") == "paper and cardboard"
    assert guess_material("throw pillow") == "fabric"
    assert guess_material("candle") == "unknown"
    assert is_near_duplicate("Blue Mug", ["Red Mug"])
    assert not is_near_duplicate("tomato soup can", ["master chef can"])


def test_order_bucket_keeps_a_small_and_a_large_object():
    names = {
        "plastic": ("bottle", "helmet", "crate", "bin"),
        "wood": ("table", "desk", "shelf", "cabinet"),
    }
    items = []
    for material in ("plastic", "wood"):
        for index, extent in enumerate((0.05, 0.12, 0.4, 1.5)):
            items.append({
                "id": f"{material}-{index}",
                "name": f"{material} {names[material][index]}",
                "material": material,
                "extent": extent,
            })
    picked = []
    for material in ("plastic", "wood"):
        group = [row for row in items if row["material"] == material]
        picked.extend(order_bucket(group, limit=4))
    for material in ("plastic", "wood"):
        extents = [row["extent"] for row in picked if row["material"] == material]
        assert min(extents) == pytest.approx(0.05)
        assert max(extents) == pytest.approx(1.5)
        assert len(extents) == 4


def test_labels_stop_when_unknown_is_common_or_a_source_collapses():
    mixed = [
        {"status": "ok", "source": "ycb", "name": f"object {index}", "material": MATERIAL_CLASSES[index % 5]}
        for index in range(20)
    ]
    assert labels_should_stop(mixed) is None
    unknown = [
        {"status": "ok", "source": "gso", "name": f"object {index}", "material": "unknown" if index < 8 else "plastic"}
        for index in range(20)
    ]
    assert "unknown" in labels_should_stop(unknown)
    collapsed = [
        {"status": "ok", "source": "abo", "name": f"chair {index}", "material": "plastic"}
        for index in range(12)
    ]
    assert "abo" in labels_should_stop(collapsed)


def test_identity_holdout_includes_the_three_known_objects():
    ids = [
        "002_master_chef_can",
        "Weisshai_Great_White_Shark",
        "B079TXJCR5",
        "003_cracker_box",
        "other_object",
    ]
    held = holdout_ids(ids, rate=0.0)
    assert {"002_master_chef_can", "Weisshai_Great_White_Shark", "B079TXJCR5"} <= held
    train = [obj_id for obj_id in ids if obj_id not in held]
    assert set(train).isdisjoint(held)
    assert "003_cracker_box" in train


def test_training_step_ignores_slots_28_through_32_and_reduces_loss():
    prior = PhysicsPrior.initialize(0)
    features = np.random.default_rng(0).normal(scale=0.02, size=(6, 912))
    label = {
        "material": "plastic",
        "body_type": "dynamic rigid",
        "collision": "box",
        "log_mass": -0.4,
        "log_density": 6.1,
        "rigidity": 0.8,
        "deformability": 0.2,
    }
    before, grads = loss_and_gradients(prior, features, label)
    assert np.all(grads["b_out"][28:] == 0.0)
    assert np.all(grads["w_out"][:, 28:] == 0.0)
    slot = 24
    epsilon = 1e-5
    prior.b_out[slot] += epsilon
    plus, _plus_grads = loss_and_gradients(prior, features, label)
    prior.b_out[slot] -= 2 * epsilon
    minus, _minus_grads = loss_and_gradients(prior, features, label)
    prior.b_out[slot] += epsilon
    numerical = (plus - minus) / (2 * epsilon)
    assert grads["b_out"][slot] == pytest.approx(numerical, rel=1e-4, abs=1e-5)
    optimizer = _Adam(
        [prior.w1, prior.b1, prior.w2, prior.b2, prior.w_out, prior.b_out],
        lr=0.05,
        decay=0.0,
    )
    optimizer.step([grads[name] for name in ("w1", "b1", "w2", "b2", "w_out", "b_out")])
    after, _after_grads = loss_and_gradients(prior, features, label)
    assert after < before


def test_vote_writes_log_mass_once_and_scales_unknown(tmp_path):
    prior = PhysicsPrior.initialize(1)
    prior.b1[:] = 0.0
    prior.b2[:] = 0.0
    prior.b_out[:] = 0.0
    prior.b_out[0] = 6.0
    prior.b_out[24] = -0.7
    prior.b_out[25] = 6.2
    prior.b_out[26] = 2.0
    prior.b_out[27] = -2.0
    path = tmp_path / "physics_prior.npz"
    prior.save(path)
    features = np.zeros((3, 912))
    mask = np.array([True, False, True])
    voted = vote_loaded_prior(
        features,
        mask,
        vlm={"source": "vlm", "class": "plastic", "weight": 1.0},
        keyword={"source": "keyword", "class": "plastic", "weight": 1.0},
        checkpoint=path,
        name="can",
        prior=PhysicsPrior.load(path),
    )
    assert voted["proposal"]["source"] == "physics_prior"
    assert voted["proposal"]["class"] == "plastic"
    assert voted["record"]["mass_kg"] is None
    assert voted["record"]["density_kg_m3"] is None
    assert voted["record"]["log_mass"] == pytest.approx(-0.7)
    assert voted["record"]["log_density"] == pytest.approx(6.2)
    assert voted["winner"] == "plastic"

    prior.b_out[:] = 0.0
    prior.b_out[9] = 8.0
    prior.b_out[24] = 1.5
    prior.b_out[25] = 2.5
    unknown = vote_loaded_prior(
        features,
        mask,
        vlm={"source": "vlm", "class": "plastic", "weight": 0.3},
        keyword={"source": "keyword", "class": "unknown", "weight": 1.0},
        checkpoint=path,
        name="mystery",
        prior=prior,
    )
    assert unknown["proposal"]["source"] == "physics_prior"
    assert unknown["proposal"]["class"] == "unknown"
    assert unknown["winner"] == "unknown"
    assert unknown["scores"]["unknown"] == pytest.approx(0.4 * unknown["proposal"]["weight"] + 0.4)
    assert unknown["record"]["log_mass"] == pytest.approx(1.5)
    assert unknown["record"]["mass_kg"] is None
