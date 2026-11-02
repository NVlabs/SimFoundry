"""Frozen MLP contract. Fake tensors only. No Gemini and no downloaded meshes."""

import math

import numpy as np
import pytest

from sceneagent.mlp.contract import (
    PriorContractError,
    geometry_from_ply,
    pack_prior_input,
    save_prior_input,
    write_gaussian_ply,
)
from sceneagent.mlp.labels import parse_label
from sceneagent.mlp.prior import (
    PhysicsPrior,
    loss_slot_mask,
    masked_prior_loss,
    mean_in_mask,
    save_checkpoint,
)
from sceneagent.physics import proposals_for_prior, store_log_mass_density, vote_material
from sceneagent.taxonomy import keyword_material


def _label(**overrides):
    payload = {
        "material": "metal",
        "body_type": "dynamic rigid",
        "collision": "cylinder",
        "mass_kg": 0.12,
        "density_kg_m3": 700.0,
        "rigidity": 0.4,
        "deformability": 0.2,
    }
    payload.update(overrides)
    return payload


def test_schema_parser_reads_the_named_fields():
    parsed = parse_label(
        _label(
            friction_static=0.35,
            friction_dynamic=0.25,
            restitution=0.1,
            unpublished_slot=99,
        )
    )
    assert parsed.material == "metal"
    assert parsed.body_type == "dynamic rigid"
    assert parsed.collision == "cylinder"
    assert parsed.mass_kg == pytest.approx(0.12)
    assert parsed.density_kg_m3 == pytest.approx(700.0)
    assert parsed.rigidity == pytest.approx(0.4)
    assert parsed.deformability == pytest.approx(0.2)
    assert parsed.friction_static == pytest.approx(0.35)
    assert parsed.friction_dynamic == pytest.approx(0.25)
    assert parsed.restitution == pytest.approx(0.1)
    target = parsed.numeric_target()
    assert target.shape == (33,)
    assert target[24] == pytest.approx(math.log(0.12))
    assert target[25] == pytest.approx(math.log(700.0))
    assert target[26] == pytest.approx(0.4)
    assert target[27] == pytest.approx(0.2)
    assert np.isnan(target[28:]).all()
    assert np.isnan(target[:24]).all()
    with pytest.raises(ValueError, match="unknown material"):
        parse_label(_label(material="granite"))
    with pytest.raises(ValueError, match="missing"):
        parse_label({"material": "metal"})


def test_log_once_for_mass_and_density():
    parsed = parse_label(_label())
    assert parsed.log_mass == pytest.approx(math.log(parsed.mass_kg))
    assert parsed.log_density == pytest.approx(math.log(parsed.density_kg_m3))
    assert parsed.log_mass < 0.0
    stored = store_log_mass_density({}, parsed.log_mass, parsed.log_density)
    assert stored["log_mass"] == pytest.approx(math.log(0.12))
    assert stored["log_density"] == pytest.approx(math.log(700.0))
    again = store_log_mass_density(stored, stored["log_mass"], stored["log_density"])
    assert again["log_mass"] == pytest.approx(stored["log_mass"])
    assert again["log_density"] == pytest.approx(stored["log_density"])


def test_refuse_a_short_ply(tmp_path):
    path = tmp_path / "short.ply"
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float opacity\n"
        "end_header\n"
        "0 0 0 0\n",
        encoding="ascii",
    )
    with pytest.raises(PriorContractError, match="short ply"):
        geometry_from_ply(path)

    full = tmp_path / "one_nan.ply"
    write_gaussian_ply(
        full,
        xyz=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        log_scales=np.array([[0.0, 0.0, math.log(2.0)], [0.0, 0.0, 0.0]]),
        opacity=np.array([0.0, 0.0]),
        rotation=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        f_dc=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    )
    text = full.read_bytes()
    # Flip one stored opacity to a NaN so that Gaussian cannot fill its row.
    header_end = text.find(b"end_header\n") + len(b"end_header\n")
    values = np.frombuffer(text[header_end:], dtype="<f4").reshape(2, 14).copy()
    values[1, 6] = np.nan
    full.write_bytes(text[:header_end] + values.astype("<f4").tobytes())
    geometry = geometry_from_ply(full)
    assert geometry.kept == 1
    assert geometry.dropped == 1
    assert geometry.rows.shape == (1, 16)
    assert np.isfinite(geometry.rows).all()
    expected = np.array([1, 2, 3, 1, 1, 2, 0.5, 1, 0, 0, 0, 0.1, 0.2, 0.3, math.log(2.0), 2.0])
    assert geometry.rows[0] == pytest.approx(expected)


def test_refuse_a_vit_b32_16_plus_16_vector(tmp_path):
    geometry = np.zeros((2, 16))
    with pytest.raises(PriorContractError, match="16\\+16 ViT-B-32"):
        pack_prior_input(np.zeros((2, 16)), np.zeros((2, 16)), geometry)
    with pytest.raises(PriorContractError, match="16\\+16 ViT-B-32"):
        pack_prior_input(np.zeros((2, 32)), np.zeros((2, 512)), geometry)
    destination = tmp_path / "prior_input.npy"
    with pytest.raises(PriorContractError, match="only at"):
        save_prior_input(destination, np.zeros((2, 32)))
    assert not destination.exists()


def test_refuse_an_8_class_npz_and_a_wrong_width(tmp_path):
    prior = PhysicsPrior.initialize(0)
    good = tmp_path / "physics_prior.npz"
    prior.save(good)
    loaded = PhysicsPrior.load(good)
    features = np.linspace(-0.2, 0.2, 2 * 912).reshape(2, 912)
    assert loaded.forward(features) == pytest.approx(prior.forward(features))

    old = tmp_path / "old8.npz"
    save_checkpoint(
        old,
        w1=prior.w1,
        b1=prior.b1,
        w2=prior.w2,
        b2=prior.b2,
        w_out=prior.w_out,
        b_out=prior.b_out,
        n_materials=8,
        materials=tuple(f"class{i}" for i in range(8)),
    )
    with pytest.raises(PriorContractError, match="8-class"):
        PhysicsPrior.load(old)

    narrow = tmp_path / "narrow.npz"
    save_checkpoint(
        narrow,
        w1=prior.w1,
        b1=prior.b1,
        w2=prior.w2,
        b2=prior.b2,
        w_out=np.zeros((32, 31)),
        b_out=np.zeros(31),
    )
    with pytest.raises(PriorContractError, match="wrong width"):
        PhysicsPrior.load(narrow)

    hidden = tmp_path / "hidden.npz"
    save_checkpoint(
        hidden,
        w1=np.zeros((912, 16)),
        b1=np.zeros(16),
        w2=prior.w2,
        b2=prior.b2,
        w_out=prior.w_out,
        b_out=prior.b_out,
    )
    with pytest.raises(PriorContractError, match="wrong width"):
        PhysicsPrior.load(hidden)


def test_forward_n912_to_n33_and_packs_only_that_width(tmp_path):
    prior = PhysicsPrior.initialize(1)
    features = np.linspace(-1.0, 1.0, 4 * 912, dtype=np.float64).reshape(4, 912)
    outputs = prior.forward(features)
    assert outputs.shape == (4, 33)
    hidden = np.tanh(features @ prior.w1 + prior.b1)
    hidden = np.tanh(hidden @ prior.w2 + prior.b2)
    assert outputs == pytest.approx(hidden @ prior.w_out + prior.b_out)
    with pytest.raises(PriorContractError):
        prior.forward(np.zeros((4, 32)))

    dino = features[:, :384]
    clip = np.linspace(0.1, 0.4, 4 * 512).reshape(4, 512)
    geometry = np.linspace(-0.5, 0.5, 4 * 16).reshape(4, 16)
    packed = pack_prior_input(dino, clip, geometry)
    assert packed.shape == (4, 912)
    assert packed[:, :384] == pytest.approx(dino)
    assert packed[:, 384:896] == pytest.approx(clip)
    assert packed[:, 896:] == pytest.approx(geometry)
    path = save_prior_input(tmp_path / "prior_input.npy", packed)
    assert np.load(path).shape == (4, 912)
    sparse = dino.copy()
    sparse[0, 0] = np.nan
    with pytest.raises(PriorContractError, match="not filled"):
        pack_prior_input(sparse, clip, geometry)


def test_in_mask_mean_is_the_mean_of_the_logits():
    prior = PhysicsPrior.initialize(2)
    features = np.linspace(-0.5, 0.5, 6 * 912).reshape(6, 912)
    mask = np.array([True, False, True, False, True, False])
    outputs = prior.forward(features)
    averaged = prior.in_mask_mean(features, mask)
    assert averaged.shape == (33,)
    assert averaged == pytest.approx(outputs[mask].mean(axis=0))
    assert averaged[:10] == pytest.approx(outputs[mask, :10].mean(axis=0))
    once = prior.forward(features[mask].mean(axis=0, keepdims=True))[0]
    assert not np.allclose(averaged, once)

    logits = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    block = np.zeros((2, 33))
    block[:, :10] = logits
    mean_logits = mean_in_mask(block, np.array([True, True]))
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    assert mean_logits[:10] == pytest.approx(logits.mean(axis=0))
    assert not np.allclose(mean_logits[:10], np.log(probabilities.mean(axis=0)))


def test_a_tie_returns_no_class():
    winner, scores = vote_material(
        [
            {"class": "metal", "weight": 1.0},
            {"class": "plastic", "weight": 1.0},
        ]
    )
    assert winner is None
    assert scores["metal"] == pytest.approx(scores["plastic"])
    winner, scores = vote_material(
        [
            {"class": "unknown", "weight": 1.0},
            {"class": "wood", "weight": 0.4},
        ]
    )
    assert winner is None
    assert scores["unknown"] == pytest.approx(0.4)
    assert scores["wood"] == pytest.approx(0.4)


def test_an_unknown_vote_is_multiplied_by_0_4():
    winner, scores = vote_material(
        [
            {"source": "vlm", "class": "unknown", "weight": 1.0},
            {"source": "physics_prior", "class": "plastic", "weight": 0.3},
        ]
    )
    assert scores["unknown"] == pytest.approx(0.4)
    assert scores["plastic"] == pytest.approx(0.3)
    assert winner == "unknown"
    winner, scores = vote_material(
        [
            {"class": "unknown", "weight": 1.0},
            {"class": "plastic", "weight": 0.5},
        ]
    )
    assert scores["unknown"] == pytest.approx(0.4)
    assert winner == "plastic"


def test_slots_28_through_32_are_excluded_from_the_loss():
    mask = loss_slot_mask()
    assert mask.shape == (33,)
    assert mask[:28].tolist() == [1.0] * 28
    assert mask[28:].tolist() == [0.0] * 5
    kwargs = {
        "material": "plastic",
        "body": "dynamic rigid",
        "collision": "box",
        "log_mass": 0.0,
        "log_density": 1.0,
        "rigidity": 0.5,
        "deformability": 0.25,
    }
    base = np.zeros(33)
    loss = masked_prior_loss(base, **kwargs)
    moved_tail = base.copy()
    moved_tail[28:] = 1.0e6
    assert masked_prior_loss(moved_tail, **kwargs) == pytest.approx(loss)
    moved_mass = base.copy()
    moved_mass[24] = 4.0
    assert masked_prior_loss(moved_mass, **kwargs) != pytest.approx(loss)


def test_keyword_proposal_is_passed_by_the_caller(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("keyword rules were consulted on the prior path")

    monkeypatch.setattr("sceneagent.taxonomy.keyword_material", boom)
    alone = proposals_for_prior({"source": "physics_prior", "class": "wood", "weight": 1.0})
    assert [item["source"] for item in alone] == ["physics_prior"]
    winner, _scores = vote_material(alone)
    assert winner == "wood"
    monkeypatch.undo()

    keyword = keyword_material("cardboard box")
    proposals = proposals_for_prior(
        {"source": "physics_prior", "class": "metal", "weight": 0.1},
        vlm={"source": "vlm", "class": "metal", "weight": 0.1},
        keyword=keyword,
    )
    assert [item["source"] for item in proposals] == ["physics_prior", "vlm", "keyword"]
    assert proposals[-1]["class"] == "paper and cardboard"
    winner, scores = vote_material(proposals)
    assert winner == "paper and cardboard"
    assert scores["paper and cardboard"] == pytest.approx(1.0)
    assert scores["metal"] == pytest.approx(0.2)
