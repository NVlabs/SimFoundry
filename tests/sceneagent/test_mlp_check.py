"""Suitability checks for MLP downloads. Files are local and tiny; no network."""

from pathlib import Path

import pytest
import trimesh
from PIL import Image

from sceneagent.mlp.check import check_sample


def _png(path: Path, mode="RGB"):
    if mode == "RGB":
        color = (20, 40, 60)
    elif mode == "RGBA":
        color = (20, 40, 60, 255)
    else:
        color = 20
    Image.new(mode, (4, 4), color).save(path)


def _triangle(path: Path):
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )


def _row(mesh: Path, image: Path, **overrides):
    sample = {
        "id": "obj1",
        "source": "ycb",
        "name": "triangle",
        "source_url": "http://example.invalid/obj",
        "mesh_path": str(mesh),
        "image_path": str(image),
    }
    sample.update(overrides)
    return sample


def _manifest(sample):
    return {"id": sample["id"], "source": sample["source"]}


def test_obj_with_rgb_image_passes(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.jpg"
    _triangle(mesh)
    Image.new("RGB", (4, 4), (8, 16, 32)).save(image, format="JPEG")
    sample = _row(mesh, image)
    result = check_sample(sample, _manifest(sample))
    assert result.ok
    assert result.status == "pass"
    assert result.reasons == ()
    assert result.extent_m[0] == pytest.approx(1.0)
    assert result.extent_m[1] == pytest.approx(1.0)
    assert result.extent_m[2] == pytest.approx(0.0)


def test_rgba_image_passes(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.png"
    _triangle(mesh)
    _png(image, "RGBA")
    sample = _row(mesh, image)
    result = check_sample(sample, _manifest(sample))
    assert result.status == "pass"


def test_glb_box_passes(tmp_path):
    mesh = tmp_path / "model.glb"
    image = tmp_path / "texture.png"
    trimesh.creation.box(extents=(0.2, 0.1, 0.04)).export(mesh)
    _png(image)
    sample = _row(mesh, image, source="abo", id="B000TEST")
    result = check_sample(sample, _manifest(sample))
    assert result.status == "pass"
    assert result.extent_m == pytest.approx((0.2, 0.1, 0.04))


def test_sdf_with_referenced_obj_passes(tmp_path):
    mesh = tmp_path / "meshes"
    mesh.mkdir()
    obj = mesh / "model.obj"
    _triangle(obj)
    sdf = tmp_path / "model.sdf"
    sdf.write_text(
        """<?xml version="1.0"?>
        <sdf version="1.6"><model name="m"><link name="l">
          <visual><geometry><mesh><uri>meshes/model.obj</uri></mesh></geometry></visual>
          <collision><geometry><mesh><uri>meshes/model.obj</uri></mesh></geometry></collision>
        </link></model></sdf>
        """,
        encoding="utf-8",
    )
    image = tmp_path / "texture.png"
    _png(image)
    sample = _row(sdf, image, source="gso", id="scan")
    result = check_sample(sample, _manifest(sample))
    assert result.status == "pass"
    assert result.extent_m[0] == pytest.approx(1.0)


def test_html_disguised_as_obj_fails(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.png"
    mesh.write_text("<!DOCTYPE html><html><body>nope</body></html>", encoding="utf-8")
    _png(image)
    result = check_sample(_row(mesh, image), _manifest(_row(mesh, image)))
    assert result.status == "fail"
    assert any("HTML" in reason for reason in result.reasons)


def test_empty_mesh_fails(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.png"
    mesh.write_bytes(b"")
    _png(image)
    result = check_sample(_row(mesh, image), {"id": "obj1", "source": "ycb"})
    assert result.status == "fail"
    assert any("empty" in reason for reason in result.reasons)


def test_zip_archive_fails(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.png"
    mesh.write_bytes(b"PK\x03\x04" + b"\x00" * 32)
    _png(image)
    result = check_sample(_row(mesh, image), {"id": "obj1", "source": "ycb"})
    assert result.status == "fail"
    assert any("zip" in reason for reason in result.reasons)


def test_zero_extent_fails(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.png"
    mesh.write_text("v 0 0 0\nv 0 0 0\nv 0 0 0\nf 1 2 3\n", encoding="utf-8")
    _png(image)
    result = check_sample(_row(mesh, image), {"id": "obj1", "source": "ycb"})
    assert result.status == "fail"


def test_zero_area_fails(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.png"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 2 0 0\nf 1 2 3\n", encoding="utf-8")
    _png(image)
    result = check_sample(_row(mesh, image), {"id": "obj1", "source": "ycb"})
    assert result.status == "fail"
    assert any("zero" in reason for reason in result.reasons)


def test_nonfinite_extent_fails(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.png"
    mesh.write_text("v 0 0 0\nv 1e309 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    _png(image)
    result = check_sample(_row(mesh, image), {"id": "obj1", "source": "ycb"})
    assert result.status == "fail"
    assert result.ok is False


def test_missing_image_fails(tmp_path):
    mesh = tmp_path / "model.obj"
    _triangle(mesh)
    sample = _row(mesh, tmp_path / "missing.png")
    result = check_sample(sample, _manifest(sample))
    assert result.status == "fail"
    assert any("image" in reason for reason in result.reasons)


def test_grayscale_image_fails(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.png"
    _triangle(mesh)
    _png(image, "L")
    result = check_sample(_row(mesh, image), {"id": "obj1", "source": "ycb"})
    assert result.status == "fail"
    assert any("not RGB" in reason for reason in result.reasons)


def test_bad_source_tag_fails(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.png"
    _triangle(mesh)
    _png(image)
    sample = _row(mesh, image, source="graspclutter")
    result = check_sample(sample, {"id": "obj1", "source": "graspclutter"})
    assert result.status == "fail"
    assert any("source tag" in reason for reason in result.reasons)


def test_id_mismatch_fails(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.png"
    _triangle(mesh)
    _png(image)
    sample = _row(mesh, image)
    result = check_sample(sample, {"id": "other", "source": "ycb"})
    assert result.status == "fail"
    assert any("id does not match" in reason for reason in result.reasons)


def test_source_mismatch_fails(tmp_path):
    mesh = tmp_path / "model.obj"
    image = tmp_path / "texture.png"
    _triangle(mesh)
    _png(image)
    sample = _row(mesh, image, source="ycb")
    result = check_sample(sample, {"id": "obj1", "source": "abo"})
    assert result.status == "fail"
    assert any("source does not match" in reason for reason in result.reasons)


def test_sdf_missing_mesh_fails(tmp_path):
    sdf = tmp_path / "model.sdf"
    sdf.write_text(
        '<sdf version="1.6"><model name="m"><link name="l">'
        "<visual><geometry><mesh><uri>meshes/missing.obj</uri></mesh></geometry></visual>"
        "</link></model></sdf>",
        encoding="utf-8",
    )
    image = tmp_path / "texture.png"
    _png(image)
    sample = _row(sdf, image, source="gso", id="scan")
    result = check_sample(sample, _manifest(sample))
    assert result.status == "fail"
    assert any("missing" in reason for reason in result.reasons)


def test_sdf_remote_uri_fails(tmp_path):
    sdf = tmp_path / "model.sdf"
    sdf.write_text(
        '<sdf version="1.6"><model name="m"><link name="l">'
        "<visual><geometry><mesh><uri>https://example.invalid/mesh.obj</uri></mesh></geometry></visual>"
        "</link></model></sdf>",
        encoding="utf-8",
    )
    image = tmp_path / "texture.png"
    _png(image)
    sample = _row(sdf, image, source="gso", id="scan")
    result = check_sample(sample, _manifest(sample))
    assert result.status == "fail"
    assert any("remote" in reason for reason in result.reasons)


def test_sdf_path_escape_fails(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    sdf = model_dir / "model.sdf"
    sdf.write_text(
        '<sdf version="1.6"><model name="m"><link name="l">'
        "<visual><geometry><mesh><uri>../outside.obj</uri></mesh></geometry></visual>"
        "</link></model></sdf>",
        encoding="utf-8",
    )
    image = tmp_path / "texture.png"
    _png(image)
    sample = _row(sdf, image, source="gso", id="scan")
    result = check_sample(sample, _manifest(sample))
    assert result.status == "fail"
    assert any("escapes" in reason for reason in result.reasons)
