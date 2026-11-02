"""Frozen physics-prior input contract.

One row is one Gaussian. ``prior_input`` is written only at shape ``(N, 912)``:

- 0–383: DINOv2 ViT-S/14
- 384–895: OpenCLIP ViT-B-16, OpenAI weights
- 896–911: the 16 geometry quantities, in ``GEOMETRY_NAMES`` order

A ply that lacks a source column is refused. A Gaussian with a non-finite
source value is dropped. Missing numbers are not filled with zeros. An old
16+16 ViT-B-32 vector is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DINO_DIM = 384
CLIP_DIM = 512
GEOM_DIM = 16
PRIOR_DIM = DINO_DIM + CLIP_DIM + GEOM_DIM
assert PRIOR_DIM == 912

# Source columns a Gaussian ply must already contain. log_volume and
# anisotropy are derived from the three log-scales; they are not ply fields.
REQUIRED_PLY_PROPERTIES = (
    "x",
    "y",
    "z",
    "scale_0",
    "scale_1",
    "scale_2",
    "opacity",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
)

GEOMETRY_NAMES = (
    "x",
    "y",
    "z",
    "exp_scale_0",
    "exp_scale_1",
    "exp_scale_2",
    "sigmoid_opacity",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "log_volume",
    "anisotropy",
)

_PLY_TYPE_SIZES = {
    "char": ("<i1", 1),
    "int8": ("<i1", 1),
    "uchar": ("<u1", 1),
    "uint8": ("<u1", 1),
    "short": ("<i2", 2),
    "int16": ("<i2", 2),
    "ushort": ("<u2", 2),
    "uint16": ("<u2", 2),
    "int": ("<i4", 4),
    "int32": ("<i4", 4),
    "uint": ("<u4", 4),
    "uint32": ("<u4", 4),
    "float": ("<f4", 4),
    "float32": ("<f4", 4),
    "double": ("<f8", 8),
    "float64": ("<f8", 8),
}


class PriorContractError(ValueError):
    """The row, ply, or checkpoint does not match the frozen contract."""

    def __init__(self, message, *, kept=None, dropped=None):
        super().__init__(message)
        self.kept = kept
        self.dropped = dropped


@dataclass(frozen=True)
class PlyGeometry:
    """Geometry rows that survived the column check.

    ``rows`` has shape ``(kept, 16)`` and is finite. ``dropped`` counts
    Gaussians whose source values were not finite. Those rows are absent,
    not replaced.
    """

    rows: np.ndarray
    kept: int
    dropped: int


def pack_prior_input(dino, clip, geometry):
    """Pack aligned Gaussian rows into ``(N, 912)``.

    ``dino`` is ``(N, 384)``, ``clip`` is ``(N, 512)``, and ``geometry`` is
    ``(N, 16)``. A 16+16 ViT-B-32 pair, a 32-wide feature, or a non-finite
    value raises ``PriorContractError`` and writes nothing.
    """

    dino = np.asarray(dino, dtype=np.float64)
    clip = np.asarray(clip, dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.float64)
    if dino.ndim != 2 or clip.ndim != 2 or geometry.ndim != 2:
        raise PriorContractError("prior rows must be rank-2 arrays")
    if dino.shape[0] != clip.shape[0] or clip.shape[0] != geometry.shape[0]:
        raise PriorContractError("DINOv2, OpenCLIP, and geometry row counts differ")
    dino_dim = int(dino.shape[1])
    clip_dim = int(clip.shape[1])
    geom_dim = int(geometry.shape[1])
    if (dino_dim, clip_dim) == (16, 16) or dino_dim == 32 or clip_dim == 32:
        raise PriorContractError(
            "refuse an old 16+16 ViT-B-32 vector; "
            "expected DINOv2 ViT-S/14 (384) and OpenCLIP ViT-B-16 (512)"
        )
    if dino_dim != DINO_DIM or clip_dim != CLIP_DIM or geom_dim != GEOM_DIM:
        raise PriorContractError(
            f"prior input width must be {PRIOR_DIM}; "
            f"got dino {dino_dim}, clip {clip_dim}, geometry {geom_dim}"
        )
    if (
        not np.isfinite(dino).all()
        or not np.isfinite(clip).all()
        or not np.isfinite(geometry).all()
    ):
        raise PriorContractError("missing feature values are not filled")
    packed = np.concatenate([dino, clip, geometry], axis=1)
    if packed.shape != (dino.shape[0], PRIOR_DIM):
        raise PriorContractError("packed width is not 912")
    return packed


def save_prior_input(path, packed):
    """Write ``prior_input.npy`` only when ``packed`` is finite and ``(N, 912)``."""

    array = np.asarray(packed)
    if array.ndim != 2 or array.shape[1] != PRIOR_DIM or array.shape[0] < 1:
        raise PriorContractError(
            f"prior_input.npy is written only at (N, {PRIOR_DIM}) with N >= 1"
        )
    if not np.isfinite(array).all():
        raise PriorContractError("missing feature values are not filled")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, np.asarray(array, dtype=np.float32))
    return destination


def geometry_from_ply(path):
    """Read Gaussian geometry, dropping rows that cannot fill the 16 columns."""

    table, total = read_ply_vertex_table(Path(path))
    missing = [name for name in REQUIRED_PLY_PROPERTIES if name not in table]
    if missing:
        raise PriorContractError("short ply: lacks columns " + ", ".join(missing))
    if total < 1:
        raise PriorContractError("short ply: vertex count is 0", kept=0, dropped=0)

    raw = np.column_stack([np.asarray(table[name], dtype=np.float64) for name in REQUIRED_PLY_PROPERTIES])
    xyz = raw[:, 0:3]
    log_scales = raw[:, 3:6]
    opacity = raw[:, 6]
    rotation = raw[:, 7:11]
    f_dc = raw[:, 11:14]

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        exp_scales = np.exp(log_scales)
        log_volume = np.sum(log_scales, axis=1)
        min_scale = np.min(exp_scales, axis=1)
        max_scale = np.max(exp_scales, axis=1)
        anisotropy = max_scale / min_scale
        sigmoid_opacity = _sigmoid(opacity)

    finite = (
        np.isfinite(raw).all(axis=1)
        & np.isfinite(exp_scales).all(axis=1)
        & np.isfinite(log_volume)
        & np.isfinite(anisotropy)
        & np.isfinite(sigmoid_opacity)
        & (min_scale > 0.0)
    )
    kept_count = int(finite.sum())
    dropped = int(total - kept_count)
    if kept_count == 0:
        raise PriorContractError(
            f"every Gaussian was dropped ({dropped} of {total}); missing values are not filled",
            kept=0,
            dropped=dropped,
        )
    rows = np.concatenate(
        [
            xyz[finite],
            exp_scales[finite],
            sigmoid_opacity[finite, None],
            rotation[finite],
            f_dc[finite],
            log_volume[finite, None],
            anisotropy[finite, None],
        ],
        axis=1,
    )
    if rows.shape != (kept_count, GEOM_DIM) or not np.isfinite(rows).all():
        raise PriorContractError("geometry rows are not a finite (K, 16) array")
    return PlyGeometry(rows=rows, kept=kept_count, dropped=dropped)


def write_gaussian_ply(path, *, xyz, log_scales, opacity, rotation, f_dc):
    """Write a binary little-endian ply with the source columns this contract reads.

    Every array must be finite. This does not invent a column the caller did
    not pass.
    """

    xyz = np.asarray(xyz, dtype=np.float64)
    log_scales = np.asarray(log_scales, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64).reshape(-1)
    rotation = np.asarray(rotation, dtype=np.float64)
    f_dc = np.asarray(f_dc, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise PriorContractError("xyz must have shape (N, 3)")
    count = int(xyz.shape[0])
    if count < 1:
        raise PriorContractError("refusing to write a ply with no Gaussians")
    if log_scales.shape != (count, 3):
        raise PriorContractError("log scales must have shape (N, 3)")
    if opacity.shape != (count,):
        raise PriorContractError("opacity must have shape (N,)")
    if rotation.shape != (count, 4):
        raise PriorContractError("rotation must have shape (N, 4)")
    if f_dc.shape != (count, 3):
        raise PriorContractError("f_dc must have shape (N, 3)")
    columns = {
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "scale_0": log_scales[:, 0],
        "scale_1": log_scales[:, 1],
        "scale_2": log_scales[:, 2],
        "opacity": opacity,
        "rot_0": rotation[:, 0],
        "rot_1": rotation[:, 1],
        "rot_2": rotation[:, 2],
        "rot_3": rotation[:, 3],
        "f_dc_0": f_dc[:, 0],
        "f_dc_1": f_dc[:, 1],
        "f_dc_2": f_dc[:, 2],
    }
    for name, values in columns.items():
        if not np.isfinite(values).all():
            raise PriorContractError(
                f"refusing to write a non-finite {name}; missing values are not filled"
            )
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header.extend(f"property float {name}" for name in REQUIRED_PLY_PROPERTIES)
    header.append("end_header")
    header_bytes = ("\n".join(header) + "\n").encode("ascii")
    packed = np.column_stack([columns[name] for name in REQUIRED_PLY_PROPERTIES]).astype("<f4")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header_bytes + packed.tobytes())
    return destination


def read_ply_vertex_table(path):
    """Return ``(columns, vertex_count)`` for the vertex element.

    ``columns`` maps a property name to a float64 array of length
    ``vertex_count``. A truncated body raises ``PriorContractError``.
    """

    path = Path(path)
    raw = path.read_bytes()
    if not raw.startswith(b"ply"):
        raise PriorContractError("short ply: file is not a ply")
    marker = b"end_header"
    header_at = raw.find(marker)
    if header_at < 0:
        raise PriorContractError("short ply: header does not end")
    header = raw[:header_at].decode("ascii", errors="replace")
    body = raw[_body_offset(raw, header_at + len(marker)) :]
    fmt, elements = _parse_ply_header(header)
    vertex = next((item for item in elements if item["name"] == "vertex"), None)
    if vertex is None:
        raise PriorContractError("short ply: vertex element is missing")
    if fmt == "ascii":
        return _read_ascii_vertices(body, elements, vertex)
    if fmt == "binary_little_endian":
        return _read_binary_vertices(body, elements, vertex)
    raise PriorContractError(f"short ply: unsupported format {fmt}")


def _body_offset(raw, offset):
    if offset < len(raw) and raw[offset : offset + 1] == b"\r":
        offset += 1
    if offset < len(raw) and raw[offset : offset + 1] == b"\n":
        offset += 1
    return offset


def _parse_ply_header(header):
    fmt = None
    elements = []
    current = None
    for line in header.splitlines():
        parts = line.strip().split()
        if not parts or parts[0] in {"ply", "comment", "obj_info"}:
            continue
        if parts[0] == "format":
            if len(parts) < 2:
                raise PriorContractError("short ply: format line is incomplete")
            fmt = parts[1]
        elif parts[0] == "element":
            if len(parts) < 3:
                raise PriorContractError("short ply: element line is incomplete")
            current = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(current)
        elif parts[0] == "property":
            if current is None:
                raise PriorContractError("short ply: property appears before an element")
            if len(parts) >= 2 and parts[1] == "list":
                raise PriorContractError("short ply: list properties are not gaussian columns")
            if len(parts) < 3:
                raise PriorContractError("short ply: property line is incomplete")
            current["props"].append((parts[1], parts[2]))
    if fmt is None:
        raise PriorContractError("short ply: format is missing")
    return fmt, elements


def _read_ascii_vertices(body, elements, vertex):
    lines = [line for line in body.decode("ascii", errors="replace").splitlines() if line.strip()]
    cursor = 0
    for element in elements:
        count = element["count"]
        width = len(element["props"])
        block = lines[cursor : cursor + count]
        cursor += count
        if element is not vertex:
            if len(block) != count:
                raise PriorContractError("short ply: body ended before the vertex element")
            continue
        if len(block) != count:
            raise PriorContractError("short ply: vertex rows are truncated")
        columns = {name: np.empty(count, dtype=np.float64) for _typ, name in element["props"]}
        for row_index, line in enumerate(block):
            values = line.split()
            if len(values) < width:
                raise PriorContractError("short ply: a vertex row is truncated")
            for prop_index, (_typ, name) in enumerate(element["props"]):
                columns[name][row_index] = float(values[prop_index])
        return columns, count
    raise PriorContractError("short ply: vertex element is missing")


def _read_binary_vertices(body, elements, vertex):
    offset = 0
    for element in elements:
        names = []
        formats = []
        offsets = []
        cursor = 0
        for typ, name in element["props"]:
            if typ not in _PLY_TYPE_SIZES:
                raise PriorContractError(f"short ply: unsupported property type {typ}")
            np_type, size = _PLY_TYPE_SIZES[typ]
            names.append(name)
            formats.append(np_type)
            offsets.append(cursor)
            cursor += size
        stride = cursor
        count = element["count"]
        nbytes = stride * count
        block = body[offset : offset + nbytes]
        offset += nbytes
        if len(block) != nbytes:
            raise PriorContractError("short ply: binary vertex data is truncated")
        if element is not vertex:
            continue
        if not names:
            raise PriorContractError("short ply: vertex element has no properties")
        dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": stride})
        table = np.frombuffer(block, dtype=dtype, count=count)
        columns = {name: np.asarray(table[name], dtype=np.float64).copy() for name in names}
        return columns, count
    raise PriorContractError("short ply: vertex element is missing")


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    out = np.empty(values.shape, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative = ~positive
    exponential = np.exp(values[negative])
    out[negative] = exponential / (1.0 + exponential)
    return out
