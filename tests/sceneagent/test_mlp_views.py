"""Front-view rule for labeling renders. No network and no Gemini."""

import numpy as np
import trimesh

from sceneagent.mlp.splat_from_mesh import (
    VIEW_SIZE,
    choose_front_view,
    front_view_unusable,
    render_views,
)


def _box(extents):
    mesh = trimesh.creation.box(extents=extents)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def test_flat_box_front_is_the_broad_face():
    vertices, faces = _box((1.0, 0.9, 0.08))
    choice = choose_front_view(vertices, faces)
    front, side, top = (choice[name][0] for name in ("front", "side", "top"))
    assert abs(float(front[2])) > 0.9
    assert abs(float(np.dot(front, side))) < 1e-6
    assert abs(float(np.dot(front, top))) < 1e-6
    assert abs(float(np.dot(side, top))) < 1e-6
    assert choice["front_aspect"] > 0.7
    assert choice["alternative_aspect"] < 0.4


def test_cylinder_front_is_not_the_cap():
    mesh = trimesh.creation.cylinder(radius=0.05, height=0.16, sections=24)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    choice = choose_front_view(vertices, faces)
    front = choice["front"][0]
    assert abs(float(front[2])) < 0.45


def test_edge_on_sliver_is_rejected_when_a_broad_face_exists():
    rgb = np.full((64, 64, 3), 232, dtype=np.uint8)
    depth = np.full((64, 64), np.inf)
    rgb[30:34, 4:60] = (180, 40, 40)
    depth[30:34, 4:60] = 1.0
    reason = front_view_unusable(
        rgb,
        depth,
        choice={"alternative_aspect": 0.9, "front_aspect": 0.1},
    )
    assert reason == "edge-on"


def test_narrow_largest_face_is_kept():
    vertices, faces = _box((0.18, 0.012, 0.012))
    choice = choose_front_view(vertices, faces)
    rgb = np.full((518, 518, 3), 232, dtype=np.uint8)
    depth = np.full((518, 518), np.inf)
    rgb[250:268, 40:478] = (30, 40, 200)
    depth[250:268, 40:478] = 1.0
    assert choice["front_projected"] + 1e-9 >= choice["alternative_projected"]
    assert front_view_unusable(rgb, depth, choice=choice) is None


def test_thin_object_without_a_broad_face_is_kept():
    rgb = np.full((64, 64, 3), 232, dtype=np.uint8)
    depth = np.full((64, 64), np.inf)
    rgb[24:40, 4:60] = (20, 90, 200)
    depth[24:40, 4:60] = 1.0
    reason = front_view_unusable(
        rgb,
        depth,
        choice={"alternative_aspect": 0.2, "front_aspect": 0.25},
    )
    assert reason is None


def test_white_porcelain_texture_is_kept():
    rgb = np.full((32, 32, 3), (212, 215, 220), dtype=np.uint8)
    rgb[:, :16] = (200, 203, 208)
    depth = np.ones((32, 32), dtype=np.float64)
    image = np.full((32, 32, 3), (230, 233, 239), dtype=np.uint8)
    assert front_view_unusable(rgb, depth, image=image) is None


def test_colorful_texture_rendered_gray_is_rejected():
    rgb = np.full((32, 32, 3), (160, 160, 160), dtype=np.uint8)
    depth = np.ones((32, 32), dtype=np.float64)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 1::2] = (20, 30, 210)
    assert front_view_unusable(rgb, depth, image=image) == "gray"


def test_flat_gray_is_rejected():
    rgb = np.full((32, 32, 3), (160, 160, 160), dtype=np.uint8)
    depth = np.ones((32, 32), dtype=np.float64)
    assert front_view_unusable(rgb, depth) == "gray"


def test_textured_render_binds_the_texture_on_the_broad_face():
    mesh = trimesh.creation.box(extents=(1.0, 0.9, 0.08))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    span = np.ptp(vertices, axis=0)
    span[span < 1e-8] = 1.0
    uv = (vertices - vertices.min(axis=0)) / span
    uv = uv[:, :2]
    image = np.full((64, 64, 3), 240, dtype=np.uint8)
    image[:, 1::4] = (20, 30, 210)
    views = render_views(mesh, image, uv)
    forward = views["front"]["forward"]
    assert abs(float(forward[2])) > 0.9
    assert views["front"]["textured"] is True
    rgb = views["front"]["rgb"]
    assert rgb.shape == (VIEW_SIZE, VIEW_SIZE, 3)
    row = rgb[VIEW_SIZE // 2].astype(np.int16)
    changes = int(np.sum(np.max(np.abs(np.diff(row, axis=0)), axis=1) > 15))
    assert changes >= 8
    assert front_view_unusable(
        rgb,
        views["front"]["depth"],
        image=image,
        choice={"alternative_aspect": 0.2, "front_aspect": 0.9},
        vertices=vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        uv=uv,
        camera=views["front"],
    ) is None
