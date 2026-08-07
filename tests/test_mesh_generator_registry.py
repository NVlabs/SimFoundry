# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared mesh-generator registry and backend construction contracts.

These cover the parts of `simfoundry.models.mesh_generator` that need no GPU, no model
weights, and no `deps/` checkout — i.e. registry lookup, the low_vram signature filtering,
and the constructor guards. Actual generation is not exercised here.
"""

import inspect

import pytest

from simfoundry.models.mesh_generator import (
    MESH_GENERATORS,
    Direct3D,
    Hunyuan,
    MeshGenerator,
    Pixal3D,
    ShapeGenerator,
    TextureGenerator,
    Trellis,
    Trellis2,
    _RejectingBackgroundRemover,
    filter_generation_kwargs,
    get_mesh_generator_cls,
    make_generator,
    publish_mesh_atomically,
)


def test_registry_contains_every_backend():
    # Both stage scripts resolve backend names through this one dict; a backend missing here
    # is silently unselectable (which is how `trellis2` went missing from the cousin stage).
    assert sorted(MESH_GENERATORS) == ["direct3d", "hunyuan", "pixal3d", "trellis", "trellis2"]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("direct3d", Direct3D),
        ("hunyuan", Hunyuan),
        ("pixal3d", Pixal3D),
        ("trellis", Trellis),
        ("trellis2", Trellis2),
    ],
)
def test_get_mesh_generator_cls_resolves(name, expected):
    assert get_mesh_generator_cls(name) is expected


def test_get_mesh_generator_cls_rejects_unknown_name():
    # ValueError rather than assert, so the check survives `python -O`.
    with pytest.raises(ValueError) as excinfo:
        get_mesh_generator_cls("not-a-backend")
    message = str(excinfo.value)
    assert "not-a-backend" in message
    # The error doubles as the list of valid options, so it must enumerate them.
    assert "pixal3d" in message


def test_every_registered_backend_generates_shape_or_texture():
    for name, generator_cls in MESH_GENERATORS.items():
        assert issubclass(generator_cls, (ShapeGenerator, TextureGenerator)), name


def test_pixal3d_is_a_single_pipeline_mesh_generator():
    assert issubclass(Pixal3D, MeshGenerator)
    assert issubclass(Pixal3D, ShapeGenerator)
    assert issubclass(Pixal3D, TextureGenerator)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"create_shape_pipeline": True, "create_texture_pipeline": False},
        {"create_shape_pipeline": False, "create_texture_pipeline": True},
    ],
)
def test_pixal3d_rejects_split_shape_texture_use(kwargs):
    # Pixal3D produces geometry and texture from one cascade. Selecting it as only one half
    # must fail immediately, not after loading ~18 GB of weights and then raising
    # NotImplementedError from the un-overridden half way through the stage loop.
    with pytest.raises(ValueError, match="single pipeline"):
        Pixal3D(**kwargs)


def test_pixal3d_set_repo_path_ignores_missing_checkout(tmp_path):
    # Stage modules call set_repo_path at import time in every mamba env, including ones that
    # cannot run Pixal3D. A missing checkout must not land on sys.path.
    original = Pixal3D.REPO_PATH
    try:
        Pixal3D.REPO_PATH = None
        Pixal3D.set_repo_path(str(tmp_path / "does-not-exist"))
        assert Pixal3D.REPO_PATH is None

        real_repo = tmp_path / "Pixal3D"
        real_repo.mkdir()
        Pixal3D.set_repo_path(str(real_repo))
        assert Pixal3D.REPO_PATH == str(real_repo)
    finally:
        Pixal3D.REPO_PATH = original


def test_make_generator_omits_low_vram_for_backends_that_lack_it():
    # Direct3D's constructor takes no arguments at all, so passing low_vram would TypeError.
    assert "low_vram" not in inspect.signature(Direct3D).parameters
    for generator_cls in (Hunyuan, Trellis, Trellis2, Pixal3D):
        assert "low_vram" in inspect.signature(generator_cls).parameters

    recorded = {}

    class FakeNoLowVram:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    make_generator(FakeNoLowVram, low_vram=True, create_shape_pipeline=True)
    assert recorded == {"create_shape_pipeline": True}

    recorded.clear()

    class FakeWithLowVram:
        def __init__(self, low_vram=False, **kwargs):
            recorded.update(kwargs)
            recorded["low_vram"] = low_vram

    make_generator(FakeWithLowVram, low_vram=True, create_shape_pipeline=True)
    assert recorded == {"create_shape_pipeline": True, "low_vram": True}


# ---------------------------------------------------------------------------
# Background-remover substitution (avoids gated, non-commercial BRIA weights)
# ---------------------------------------------------------------------------

def test_rejecting_background_remover_never_runs_but_survives_device_moves():
    # The pipeline calls .to()/.cpu() on this during device placement, so it must tolerate them.
    stub = _RejectingBackgroundRemover(model_name="briaai/RMBG-2.0")
    assert stub.to("cuda") is stub
    assert stub.cpu() is stub
    assert stub.eval() is stub

    # Actually invoking it means an input slipped through without usable alpha. That must be a
    # loud, explanatory failure rather than a silent gated download.
    with pytest.raises(RuntimeError) as excinfo:
        stub(object())
    message = str(excinfo.value)
    assert "briaai/RMBG-2.0" in message
    assert "non-commercial" in message
    assert "_transparent.png" in message


# ---------------------------------------------------------------------------
# Atomic publication
# ---------------------------------------------------------------------------

def _unit_box():
    import trimesh
    return trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def test_publish_mesh_atomically_publishes_and_leaves_no_temp_files(tmp_path):
    out_fpath = tmp_path / "iter_0_mesh.glb"
    publish_mesh_atomically(str(out_fpath), _unit_box().export)

    assert out_fpath.exists()
    # No leftovers, and nothing that a `*_mesh.glb` consumer could mistake for a result.
    assert [p.name for p in tmp_path.iterdir()] == ["iter_0_mesh.glb"]


def test_publish_mesh_atomically_leaves_destination_untouched_on_failure(tmp_path):
    out_fpath = tmp_path / "iter_0_mesh.glb"
    out_fpath.write_bytes(b"previous-good-mesh")

    def failing_write(_tmp_path):
        raise RuntimeError("generation blew up mid-export")

    with pytest.raises(RuntimeError, match="blew up"):
        publish_mesh_atomically(str(out_fpath), failing_write)

    # The half-written state must never become visible at the watched path.
    assert out_fpath.read_bytes() == b"previous-good-mesh"
    assert [p.name for p in tmp_path.iterdir()] == ["iter_0_mesh.glb"]


def test_publish_mesh_atomically_rejects_a_corrupt_export(tmp_path):
    out_fpath = tmp_path / "iter_0_mesh.glb"

    def truncated_write(tmp_fpath):
        with open(tmp_fpath, "wb") as f:
            f.write(b"not a glb")

    # Validation must catch this before publishing, so consumers never see the bad file.
    with pytest.raises(Exception):
        publish_mesh_atomically(str(out_fpath), truncated_write)
    assert not out_fpath.exists()
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# generation_kwargs forwarding
# ---------------------------------------------------------------------------

def test_filter_generation_kwargs_passes_everything_to_var_keyword_backends():
    kwargs = {"seed": 0, "resolution": 1536, "fov": 0.2}
    for generate_fn in (Pixal3D.generate_mesh, Trellis2.generate_mesh, Trellis.generate_mesh):
        assert filter_generation_kwargs(generate_fn, kwargs) == kwargs


def test_filter_generation_kwargs_drops_names_fixed_signature_backends_reject():
    # Hunyuan.generate_shape enumerates its parameters and has no **kwargs, so forwarding a
    # config dict unfiltered raises TypeError -- which would break the *default* backend the
    # moment any generation option (or the inherited global seed) is set.
    kwargs = {"seed": 0, "resolution": 1536, "octree_resolution": 512}
    filtered = filter_generation_kwargs(Hunyuan.generate_shape, kwargs)
    assert filtered == {"octree_resolution": 512}

    with pytest.raises(TypeError):
        inspect.signature(Hunyuan.generate_shape).bind(None, out_fpath="x.glb", **kwargs)
    # The filtered set binds cleanly.
    inspect.signature(Hunyuan.generate_shape).bind(None, out_fpath="x.glb", **filtered)


def test_filter_generation_kwargs_drops_everything_for_direct3d():
    assert filter_generation_kwargs(Direct3D.generate_shape, {"seed": 0}) == {}


def test_resolve_generation_kwargs_defaults_seed_to_global_seed():
    from omegaconf import OmegaConf

    from simfoundry.pipeline.stage_utils import resolve_generation_kwargs

    cfg = OmegaConf.create({"seed": 7, "s7_mesh": {"generation_kwargs": {"resolution": 1024}}})
    resolved = resolve_generation_kwargs(cfg, cfg.s7_mesh)
    assert resolved == {"resolution": 1024, "seed": 7}
    assert isinstance(resolved, dict)  # plain dict, splattable into generator calls

    # An explicit seed wins over the global one.
    cfg2 = OmegaConf.create({"seed": 7, "s7_mesh": {"generation_kwargs": {"seed": 99}}})
    assert resolve_generation_kwargs(cfg2, cfg2.s7_mesh) == {"seed": 99}

    # Absent generation_kwargs still yields the global seed rather than the backend default.
    cfg3 = OmegaConf.create({"seed": 3, "s7_mesh": {}})
    assert resolve_generation_kwargs(cfg3, cfg3.s7_mesh) == {"seed": 3}


# ---------------------------------------------------------------------------
# Reserved-name collisions (would otherwise be "multiple values for keyword argument")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "generate_fn,key",
    [
        (Pixal3D.generate_mesh, "generate_mesh"),
        (Trellis2.generate_mesh, "generate_mesh"),
        (Hunyuan.generate_shape, "generate_shape"),
        (Direct3D.generate_shape, "generate_shape"),
        (Hunyuan.generate_texture, "generate_texture"),
    ],
)
def test_reserved_names_are_never_forwarded(generate_fn, key):
    # The stage scripts bind these themselves. `visualize` is in every backend signature and
    # var-keyword backends accept anything, so signature filtering alone does not remove them.
    from simfoundry.models.mesh_generator import RESERVED_GENERATION_KWARGS

    candidate = {name: "x" for name in RESERVED_GENERATION_KWARGS[key]}
    candidate["seed"] = 0
    filtered = filter_generation_kwargs(
        generate_fn, candidate, reserved=RESERVED_GENERATION_KWARGS[key]
    )
    assert not (set(filtered) & set(RESERVED_GENERATION_KWARGS[key]))


def test_base_generate_mesh_routes_loose_kwargs_to_both_halves():
    # MeshGenerator.generate_mesh ends in **kwargs, so filter_generation_kwargs reports every
    # name as accepted. Without explicit routing the values are silently discarded -- and this is
    # the default path, since Hunyuan does not override generate_mesh.
    assert Hunyuan.generate_mesh is MeshGenerator.generate_mesh

    seen = {}

    class Recording(MeshGenerator):
        def __init__(self):
            pass  # skip create_pipelines

        def generate_shape(self, out_fpath, image_path=None, prompt=None, visualize=False,
                           octree_resolution=384):
            seen["shape"] = {"octree_resolution": octree_resolution}

        def generate_texture(self, shape_fpath, out_fpath, image_path=None, prompt=None,
                             visualize=False, use_remesh=True):
            seen["texture"] = {"use_remesh": use_remesh}

    Recording().generate_mesh(
        out_fpath="/tmp/does-not-matter.glb",
        octree_resolution=512,
        use_remesh=False,
        seed=7,  # accepted by neither half -> must be reported, not silently dropped
    )
    assert seen["shape"] == {"octree_resolution": 512}
    assert seen["texture"] == {"use_remesh": False}


# ---------------------------------------------------------------------------
# Pinned local snapshots
# ---------------------------------------------------------------------------

def _install_snapshot(tmp_path, key, revision=None):
    """Create a snapshot dir the way install_pixal3d.sh leaves it (marker written last)."""
    snapshot = tmp_path / Pixal3D.SNAPSHOT_SUBDIRS[key]
    snapshot.mkdir(exist_ok=True)
    inner = Pixal3D.SNAPSHOT_FILES.get(key)
    if inner:
        (snapshot / inner).write_bytes(b"x")
    rev = revision if revision is not None else Pixal3D.SNAPSHOT_REVISIONS[key]
    (snapshot / Pixal3D.REVISION_MARKER).write_text(rev, encoding="utf-8")
    return snapshot


def test_resolve_model_source_prefers_local_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv(Pixal3D.WEIGHTS_DIR_ENV_VAR, str(tmp_path))

    # No snapshot installed -> fall back to the (mutable) repo id.
    assert Pixal3D.resolve_model_source("pixal3d", "TencentARC/Pixal3D") == "TencentARC/Pixal3D"

    snapshot = _install_snapshot(tmp_path, "pixal3d")
    assert Pixal3D.resolve_model_source("pixal3d", "TencentARC/Pixal3D") == str(snapshot)


def test_resolve_model_source_rejects_snapshot_without_revision_marker(tmp_path, monkeypatch):
    # An interrupted `hf download` leaves a plausible-looking directory with no marker: the
    # marker is written last. Existence alone must not count as installed, or a half-downloaded
    # snapshot loads silently.
    monkeypatch.setenv(Pixal3D.WEIGHTS_DIR_ENV_VAR, str(tmp_path))
    (tmp_path / Pixal3D.SNAPSHOT_SUBDIRS["pixal3d"]).mkdir()
    assert Pixal3D.resolve_model_source("pixal3d", "TencentARC/Pixal3D") == "TencentARC/Pixal3D"


def test_resolve_model_source_rejects_snapshot_at_unexpected_revision(tmp_path, monkeypatch):
    # A bumped pin downloaded over an existing tree, or a hand-edited snapshot, must not be
    # accepted as the pinned revision this build expects.
    monkeypatch.setenv(Pixal3D.WEIGHTS_DIR_ENV_VAR, str(tmp_path))
    _install_snapshot(tmp_path, "pixal3d", revision="deadbeef" * 5)
    assert Pixal3D.resolve_model_source("pixal3d", "TencentARC/Pixal3D") == "TencentARC/Pixal3D"


def test_snapshot_revisions_cover_every_snapshot():
    # A key present in SNAPSHOT_SUBDIRS but missing from SNAPSHOT_REVISIONS would skip
    # verification entirely and silently accept any tree on disk.
    assert set(Pixal3D.SNAPSHOT_REVISIONS) == set(Pixal3D.SNAPSHOT_SUBDIRS)
    for key, rev in Pixal3D.SNAPSHOT_REVISIONS.items():
        assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev), key


def test_resolve_model_source_covers_every_runtime_model(tmp_path, monkeypatch):
    # A model missing from SNAPSHOT_SUBDIRS silently stays unpinned, so assert the full set.
    assert set(Pixal3D.SNAPSHOT_SUBDIRS) == {"pixal3d", "dinov3", "moge"}
    monkeypatch.setenv(Pixal3D.WEIGHTS_DIR_ENV_VAR, str(tmp_path))
    for key in Pixal3D.SNAPSHOT_SUBDIRS:
        snapshot = _install_snapshot(tmp_path, key)
        inner = Pixal3D.SNAPSHOT_FILES.get(key)
        expected = str(snapshot / inner) if inner else str(snapshot)
        assert Pixal3D.resolve_model_source(key, "some/repo") == expected


def test_moge_snapshot_resolves_to_a_checkpoint_file_not_a_directory(tmp_path, monkeypatch):
    # MoGeModel.from_pretrained torch.load()s the path it is given, so a directory raises
    # IsADirectoryError at pipeline construction. It must receive model.pt.
    monkeypatch.setenv(Pixal3D.WEIGHTS_DIR_ENV_VAR, str(tmp_path))
    _install_snapshot(tmp_path, "moge")
    resolved = Pixal3D.resolve_model_source("moge", "Ruicheng/moge-2-vitl")
    assert resolved.endswith("model.pt")
    import os
    assert os.path.isfile(resolved)


# ---------------------------------------------------------------------------
# set_repo_path must not mutate sys.path (cross-backend interference)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "generator_cls,subdirs",
    [
        (Hunyuan, ("hy3dshape", "hy3dpaint")),
        (Trellis, ()),
        (Trellis2, ()),
        (Pixal3D, ()),
    ],
)
def test_set_repo_path_does_not_touch_sys_path(generator_cls, subdirs, tmp_path):
    """Stage scripts call set_repo_path for EVERY backend at import time, including ones that
    are not selected. Eagerly inserting those repos leaks their packages into the interpreter.

    This is not hypothetical: Hunyuan3D-2.1/hy3dpaint contains a top-level regular package named
    `src`, and a regular package shadows a namespace package anywhere on sys.path. That broke
    Pixal3D's NAF loader (`from src.model.naf import NAF` -> "No module named 'src.model'")
    whenever hunyuan's paths had been registered, even with pixal3d selected.
    """
    import sys

    repo = tmp_path / "repo"
    repo.mkdir()
    for sub in subdirs:
        (repo / sub).mkdir()

    original_repo_path = generator_cls.REPO_PATH
    before = list(sys.path)
    try:
        generator_cls.set_repo_path(str(repo))
        assert sys.path == before, (
            f"{generator_cls.__name__}.set_repo_path mutated sys.path; defer it to create_pipelines"
        )
        # It must still record the path, otherwise create_pipelines cannot find the checkout.
        assert generator_cls.REPO_PATH == str(repo)
    finally:
        generator_cls.REPO_PATH = original_repo_path
        sys.path[:] = before
