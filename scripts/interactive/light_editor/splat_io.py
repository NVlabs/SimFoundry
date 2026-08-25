# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Read a NuRec Gaussian-splat room out of a USDZ and write a browser ``.splat`` proxy.

A NuRec room is a ``Volume`` prim whose ``OmniNuRecFieldAsset`` children point
at a ``gs.nurec`` payload: a gzip stream wrapping a MessagePack document,
readable without NuRec, Isaac Sim or a GPU. The gaussian tensors inside it::

    positions        (N, 3)   metres, in the volume prim's frame
    rotations        (N, 4)   quaternion, **wxyz** (INRIA 3DGS order), unnormalised
    scales           (N, 3)   log metres; the config's scale_activation is exp
    densities        (N, 1)   logits; the config's density_activation is sigmoid
    features_albedo  (N, 3)   spherical-harmonic DC term
    features_specular(N, 45)  SH degrees 1-3, view-dependent colour

``features_specular`` is never read: it is by far the largest tensor and a
placement editor does not need view-dependent colour, so specular highlights
are baked to their DC average.

The output is a ``.splat`` file: a small JSON header and then planar arrays the
browser can view as typed arrays. See :func:`write_splat_file` for the layout.

Usage:
    python splat_io.py <room.usdz> [--out room.splat] [--budget 1000000]
"""

import argparse
import gzip
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

try:
    from pxr import Usd, UsdGeom
except ImportError:  # pragma: no cover - environment guard
    sys.exit(
        "pxr not found. This tool needs standalone OpenUSD, not Isaac Sim's copy:\n"
        "    pip install -r requirements.txt"
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))
import usd_cache  # noqa: E402

#: Attribute Omniverse stamps on a NuRec ``Volume`` prim; its presence is what
#: identifies a gaussian room, not the file extension.
NUREC_VOLUME_FLAG = "omni:nurec:isNuRecVolume"

#: Prim type of the field assets under the volume. Standalone OpenUSD has no
#: schema for it, so the prim loads untyped but reports this token.
NUREC_FIELD_TYPE = "OmniNuRecFieldAsset"

#: Zeroth-order spherical-harmonic basis: ``rgb = 0.5 + C0 * dc`` (3DGS convention).
SH_C0 = 0.28209479177387814

#: Prefix every gaussian tensor shares in the serialized state dict.
_TENSOR_PREFIX = ".gaussians_nodes.gaussians."

#: Activations this reader knows how to undo. The file names them, so an
#: exporter that changes one fails loudly instead of rendering wrong.
_EXPECTED_ACTIVATIONS = {
    "density_activation": "sigmoid",
    "scale_activation": "exp",
    "rotation_activation": "normalize",
}

#: Most gaussians a browser proxy carries before thinning. At 40 bytes a
#: gaussian this is a 40 MB download.
DEFAULT_SPLAT_BUDGET = 1_000_000

#: Below this an alpha rounds to zero in the stored 8-bit colour, so the
#: gaussian would be uploaded to draw nothing.
_MIN_OPACITY = 1.0 / 255.0

#: Read size when stepping over an unwanted tensor in the stream.
_SKIP_CHUNK = 1 << 20

#: Gaussians per pass when deriving covariances; bounds peak memory.
_CHUNK = 1 << 21

SPLAT_MAGIC = b"SFSPLAT\x01"

#: Bytes per gaussian in a ``.splat``: 3 float32 centre, 6 float32 covariance,
#: 4 uint8 colour.
SPLAT_STRIDE = 12 + 24 + 4


class SplatReadError(RuntimeError):
    """Raised when a NuRec file cannot be read as gaussians."""


# --------------------------------------------------------------------------
# MessagePack, streamed
#
# Hand-rolled because the parse must stream: `msgpack.unpackb` would
# materialise the gigabyte-scale specular blob this module skips.
# --------------------------------------------------------------------------


class _Stream:
    """A forward-only reader over a decompressed NuRec file."""

    def __init__(self, handle):
        self._handle = handle
        self.offset = 0

    def exact(self, count):
        """Read exactly *count* bytes, or raise."""
        chunks, remaining = [], count
        while remaining > 0:
            chunk = self._handle.read(remaining)
            if not chunk:
                raise SplatReadError(
                    f"truncated at byte {self.offset}: wanted {count}, short by {remaining}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        self.offset += count
        return b"".join(chunks) if len(chunks) > 1 else chunks[0]

    def skip(self, count):
        """Step over *count* bytes without keeping them."""
        remaining = count
        while remaining > 0:
            chunk = self._handle.read(min(remaining, _SKIP_CHUNK))
            if not chunk:
                raise SplatReadError(f"truncated while skipping {count} bytes")
            remaining -= len(chunk)
        self.offset += count

    def byte(self):
        return self.exact(1)[0]

    def unpack(self, fmt, size):
        return struct.unpack(fmt, self.exact(size))[0]


def _decode(stream, keep, path=""):
    """Decode one MessagePack value, skipping binary blobs *keep* rejects.

    Args:
        stream (_Stream): Reader positioned at a value.
        keep (callable): Given a slash-joined key path, returns whether that
            binary blob's bytes are wanted. A rejected blob decodes to None.
        path (str): Key path of the value being decoded.

    Returns:
        The decoded value: dict, list, str, bytes, int, float, bool or None.
    """
    code = stream.byte()

    if code <= 0x7F:                                    # positive fixint
        return code
    if code >= 0xE0:                                    # negative fixint
        return code - 0x100
    if 0x80 <= code <= 0x8F:
        return _decode_map(stream, code & 0x0F, keep, path)
    if 0x90 <= code <= 0x9F:
        return _decode_array(stream, code & 0x0F, keep, path)
    if 0xA0 <= code <= 0xBF:
        return stream.exact(code & 0x1F).decode("utf-8")

    if code == 0xC0:
        return None
    if code == 0xC2:
        return False
    if code == 0xC3:
        return True
    if code == 0xC4:
        return _decode_bin(stream, stream.byte(), keep, path)
    if code == 0xC5:
        return _decode_bin(stream, stream.unpack(">H", 2), keep, path)
    if code == 0xC6:
        return _decode_bin(stream, stream.unpack(">I", 4), keep, path)
    if code == 0xCA:
        return stream.unpack(">f", 4)
    if code == 0xCB:
        return stream.unpack(">d", 8)
    if code == 0xCC:
        return stream.byte()
    if code == 0xCD:
        return stream.unpack(">H", 2)
    if code == 0xCE:
        return stream.unpack(">I", 4)
    if code == 0xCF:
        return stream.unpack(">Q", 8)
    if code == 0xD0:
        return stream.unpack(">b", 1)
    if code == 0xD1:
        return stream.unpack(">h", 2)
    if code == 0xD2:
        return stream.unpack(">i", 4)
    if code == 0xD3:
        return stream.unpack(">q", 8)
    if code == 0xD9:
        return stream.exact(stream.byte()).decode("utf-8")
    if code == 0xDA:
        return stream.exact(stream.unpack(">H", 2)).decode("utf-8")
    if code == 0xDB:
        return stream.exact(stream.unpack(">I", 4)).decode("utf-8")
    if code == 0xDC:
        return _decode_array(stream, stream.unpack(">H", 2), keep, path)
    if code == 0xDD:
        return _decode_array(stream, stream.unpack(">I", 4), keep, path)
    if code == 0xDE:
        return _decode_map(stream, stream.unpack(">H", 2), keep, path)
    if code == 0xDF:
        return _decode_map(stream, stream.unpack(">I", 4), keep, path)

    raise SplatReadError(
        f"unsupported MessagePack type 0x{code:02x} at byte {stream.offset - 1}"
    )


def _decode_map(stream, count, keep, path):
    result = {}
    for _ in range(count):
        key = _decode(stream, keep, path)
        result[key] = _decode(stream, keep, f"{path}/{key}" if path else str(key))
    return result


def _decode_array(stream, count, keep, path):
    return [_decode(stream, keep, path) for _ in range(count)]


def _decode_bin(stream, length, keep, path):
    if keep(path):
        return stream.exact(length)
    stream.skip(length)
    return None


# --------------------------------------------------------------------------
# Locating the volume
# --------------------------------------------------------------------------


def _archive_member(reference, usd_dir):
    """Split ``archive.usdz[inner/path]`` into an archive path and a member.

    Duplicated from ``extract`` because ``extract`` imports this module.
    """
    if not reference or "[" not in reference or not reference.endswith("]"):
        return None
    archive, member = reference[:-1].split("[", 1)
    archive_path = Path(archive)
    if not archive_path.is_absolute():
        archive_path = (usd_dir / archive_path).resolve()
    return archive_path, member


def nurec_volume(usd_path):
    """Describe the NuRec gaussian volume in a USD, or return None.

    Args:
        usd_path (str or Path): USD, ``.usdz`` or otherwise.

    Returns:
        dict or None: ``prim`` (path string), ``archive``/``member`` or ``file``
        for the ``.nurec`` payload, ``transform`` (4x4 row-vector, the volume's
        local-to-world inside its own stage), ``extent`` and ``upAxis``.
        None when this USD carries no gaussian volume, which is the common case
        and not an error.
    """
    usd_path = Path(usd_path)
    try:
        stage = usd_cache.open_stage(usd_path)
    except Exception:  # noqa: BLE001 - an unreadable stage is simply not NuRec
        return None
    if stage is None:
        return None

    for prim in stage.Traverse():
        flag = prim.GetAttribute(NUREC_VOLUME_FLAG)
        if not (flag and flag.Get()):
            continue
        field = next(
            (child for child in prim.GetChildren()
             if child.GetTypeName() == NUREC_FIELD_TYPE
             and child.GetAttribute("filePath")),
            None,
        )
        if field is None:
            continue

        asset = field.GetAttribute("filePath").Get()
        raw = str(getattr(asset, "path", "") or "")
        resolved = str(getattr(asset, "resolvedPath", "") or "")
        # A payload packaged inside a USDZ is spelled relatively ("./gs.nurec");
        # only the resolved path names the archive it came from.
        archive = _archive_member(resolved, usd_path.parent) or _archive_member(
            raw, usd_path.parent)

        info = {
            "prim": str(prim.GetPath()),
            "transform": np.asarray(
                UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim),
                dtype=np.float64,
            ),
            "extent": None,
            "upAxis": str(UsdGeom.GetStageUpAxis(stage)),
            "metersPerUnit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        }
        extent = prim.GetAttribute("extent").Get()
        if extent is not None and len(extent) == 2:
            info["extent"] = [[float(v) for v in point] for point in extent]

        if archive:
            info["archive"], info["member"] = str(archive[0]), archive[1]
        else:
            candidate = Path(resolved or raw)
            if not candidate.is_absolute():
                candidate = (usd_path.parent / candidate).resolve()
            info["file"] = str(candidate)
        return info
    return None


def is_nurec_usd(usd_path):
    """Whether a USD is a NuRec Gaussian-splat volume."""
    return nurec_volume(usd_path) is not None


def _open_payload(volume):
    """Open the ``.nurec`` byte stream a :func:`nurec_volume` record names."""
    if "archive" in volume:
        archive = zipfile.ZipFile(volume["archive"])
        try:
            return gzip.GzipFile(fileobj=archive.open(volume["member"])), archive
        except Exception:
            archive.close()
            raise
    return gzip.GzipFile(filename=volume["file"]), None


# --------------------------------------------------------------------------
# Reading the gaussians
# --------------------------------------------------------------------------


_WANTED_TENSORS = ("positions", "rotations", "scales", "densities", "features_albedo")


def _tensor(state, name, precision):
    """View one tensor out of the decoded state dict, without copying."""
    blob = state.get(_TENSOR_PREFIX + name)
    shape = state.get(_TENSOR_PREFIX + name + ".shape")
    if blob is None or not shape:
        raise SplatReadError(f"{name} is missing from this NuRec file")
    dtype = np.float16 if precision == 16 else np.float32
    array = np.frombuffer(blob, dtype=dtype)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise SplatReadError(
            f"{name} holds {array.size} values but its shape says {expected}"
        )
    return array.reshape([int(v) for v in shape])


def read_nurec(usd_path, *, want=_WANTED_TENSORS, on_progress=None):
    """Decode the raw gaussian tensors of a NuRec USD.

    Args:
        usd_path (str or Path): A USD holding a NuRec volume.
        want (tuple[str]): Tensor names to keep. Everything else is stepped
            over in the stream rather than decompressed and dropped.
        on_progress (callable or None): Receives one status string.

    Returns:
        dict: ``volume`` (the :func:`nurec_volume` record), ``count``, and one
        numpy view per requested tensor, still in the file's own units.

    Raises:
        SplatReadError: If the USD is not NuRec, the payload is unreadable, or
            it was written with activations this reader does not undo.
    """
    volume = nurec_volume(usd_path)
    if volume is None:
        raise SplatReadError(f"{usd_path} carries no NuRec gaussian volume")

    keys = {_TENSOR_PREFIX + name for name in want}

    def keep(path):
        return path.rsplit("/", 1)[-1] in keys

    if on_progress:
        on_progress(f"reading gaussians from {Path(usd_path).name}")

    handle, archive = _open_payload(volume)
    try:
        document = _decode(_Stream(handle), keep)
    finally:
        handle.close()
        if archive is not None:
            archive.close()

    data = document.get("nre_data") if isinstance(document, dict) else None
    if not isinstance(data, dict) or "state_dict" not in data:
        raise SplatReadError("this .nurec has no nre_data/state_dict block")

    layer = ((data.get("config") or {}).get("layers") or {}).get("gaussians") or {}
    for field, expected in _EXPECTED_ACTIVATIONS.items():
        actual = layer.get(field)
        if actual is not None and actual != expected:
            raise SplatReadError(
                f"this NuRec file was written with {field}={actual!r}, and this "
                f"reader only knows how to undo {expected!r}. Reading it anyway "
                "would place the room at the wrong size or opacity, silently."
            )
    precision = int(layer.get("precision", 16))
    if precision not in (16, 32):
        raise SplatReadError(f"unsupported gaussian precision {precision}")

    # NuRec's own near clip, carried through so the browser culls the large
    # faint gaussians near the scan camera the way Isaac Sim does.
    culling = (data.get("config") or {}).get("renderer", {}).get("culling") or {}
    near_clip = culling.get("near_clip_distance")

    state = data["state_dict"]
    tensors = {name: _tensor(state, name, precision) for name in want}
    count = int(len(next(iter(tensors.values()))))
    for name, array in tensors.items():
        if len(array) != count:
            raise SplatReadError(
                f"{name} holds {len(array)} gaussians but positions hold {count}"
            )

    return {
        "volume": volume,
        "count": count,
        "version": data.get("version"),
        "near_clip": None if near_clip is None else float(near_clip),
        **tensors,
    }


#: Edge of the cell thinning shares its budget across, in metres.
_THIN_CELL = 0.1


def _importance(scales, densities, chunk=_CHUNK):
    """Per-gaussian importance: opacity times the area of the disc it draws.

    The area is the product of the two largest axes; a scanned gaussian is a
    flat ellipse on a surface. :func:`_thin` uses this within cells only, never
    as a global ranking.
    """
    total = len(densities)
    out = np.empty(total, dtype=np.float32)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        axes = np.exp(scales[start:stop].astype(np.float32))
        axes.sort(axis=1)
        opacity = 1.0 / (1.0 + np.exp(-densities[start:stop, 0].astype(np.float32)))
        out[start:stop] = opacity * axes[:, 1] * axes[:, 2]
    return out


def _thin(centres, score, budget, cell=_THIN_CELL):
    """Choose *budget* gaussians, keeping the room's own density distribution.

    The budget is shared out spatially: space is cut into cells, each cell
    keeps the same fraction of what it holds, and within a cell the largest and
    most opaque win. A global top-N by importance would instead spend much of
    the budget on the huge faint far-field blobs a trained splat carries.

    Args:
        centres (np.ndarray): (N, 3) gaussian centres.
        score (np.ndarray): (N,) importance, from :func:`_importance`.
        budget (int): How many to keep.
        cell (float): Cell edge in metres.

    Returns:
        np.ndarray: Indices into *centres*, ascending, of length *budget*.
    """
    total = len(centres)
    if total <= budget:
        return np.arange(total)

    grid = np.floor(centres / cell).astype(np.int64)
    grid -= grid.min(axis=0)
    # One integer key per cell; 21 bits an axis covers a 200 km span at 10 cm.
    if grid.max() >= (1 << 21):
        raise SplatReadError("gaussians span too much space to thin on a 0.1 m grid")
    key = (grid[:, 0] << 42) | (grid[:, 1] << 21) | grid[:, 2]

    # Sorted by cell, and by descending importance inside each cell, so a
    # gaussian's rank within its cell is its offset from the cell's first row.
    order = np.lexsort((-score, key))
    ranked_key = key[order]
    cells, start, counts = np.unique(ranked_key, return_index=True, return_counts=True)
    cell_of = np.repeat(np.arange(len(cells), dtype=np.int64), counts)
    rank = np.arange(total, dtype=np.int64) - start[cell_of]

    # Each cell keeps `count * rate` gaussians, almost never a whole number.
    # The fraction is resolved by rounding up with probability equal to it,
    # drawn from a hash of the cell's own key rather than an RNG, so two runs
    # on the same room keep the same gaussians.
    rate = budget / total
    quota = np.floor(counts * rate + _cell_dither(cells)).astype(np.int64)
    # floor(c*rate + u) with rate < 1 and u < 1 cannot exceed c, so a cell is
    # never asked for more than it has.
    keep = rank < quota[cell_of]
    kept = order[keep]

    # Quotas sum to the budget only in expectation; close the small residual by
    # score.
    if len(kept) > budget:
        kept = kept[np.argpartition(score[kept], len(kept) - budget)[-budget:]]
    elif len(kept) < budget:
        rest = order[~keep]
        short = budget - len(kept)
        kept = np.concatenate(
            [kept, rest[np.argpartition(score[rest], len(rest) - short)[-short:]]])
    return np.sort(kept)


def _cell_dither(keys):
    """A fixed number in [0, 1) per cell, from the cell's own key.

    Uses the murmur3 finalizer so thinning is deterministic across runs and
    machines.
    """
    with np.errstate(over="ignore"):
        h = keys.astype(np.uint64)
        h = h ^ (h >> np.uint64(33))
        h = h * np.uint64(0xFF51AFD7ED558CCD)
        h = h ^ (h >> np.uint64(33))
        h = h * np.uint64(0xC4CEB9FE1A85EC53)
        h = h ^ (h >> np.uint64(33))
    # 53 bits is what a float64 holds exactly, so this is uniform on [0, 1).
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))


def _covariances(scales, rotations, linear, chunk=_CHUNK):
    """World-space 3x3 covariances, packed as the six distinct terms.

    ``Sigma = M Sigma_local M^T`` with ``M = linear @ R @ diag(exp(scale))``.
    The volume prim's transform is folded in here so the browser shader never
    sees the USD frame.
    """
    total = len(scales)
    out = np.empty((total, 6), dtype=np.float32)
    linear = np.asarray(linear, dtype=np.float32)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        quat = rotations[start:stop].astype(np.float32)
        # The file stores quaternions unnormalised; normalising undoes the
        # declared "normalize" rotation_activation.
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        np.divide(quat, np.maximum(norm, 1e-20), out=quat)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        rot = np.empty((stop - start, 3, 3), dtype=np.float32)
        rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
        rot[:, 0, 1] = 2 * (x * y - z * w)
        rot[:, 0, 2] = 2 * (x * z + y * w)
        rot[:, 1, 0] = 2 * (x * y + z * w)
        rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
        rot[:, 1, 2] = 2 * (y * z - x * w)
        rot[:, 2, 0] = 2 * (x * z - y * w)
        rot[:, 2, 1] = 2 * (y * z + x * w)
        rot[:, 2, 2] = 1 - 2 * (x * x + y * y)

        axes = np.exp(scales[start:stop].astype(np.float32))
        # S is diagonal, so scaling R's columns is the same as R @ S.
        matrix = linear @ (rot * axes[:, None, :])
        sigma = matrix @ np.transpose(matrix, (0, 2, 1))
        out[start:stop, 0] = sigma[:, 0, 0]
        out[start:stop, 1] = sigma[:, 0, 1]
        out[start:stop, 2] = sigma[:, 0, 2]
        out[start:stop, 3] = sigma[:, 1, 1]
        out[start:stop, 4] = sigma[:, 1, 2]
        out[start:stop, 5] = sigma[:, 2, 2]
    return out


def load_gaussians(usd_path, *, budget=DEFAULT_SPLAT_BUDGET, on_progress=None):
    """Read a NuRec room as browser-ready gaussians.

    Everything is expressed in the volume prim's stage-root frame, matching what
    :func:`extract.load_visual_scene` does for a mesh: the scene JSON's pose and
    scale then position it exactly as OmniGibson does.

    Args:
        usd_path (str or Path): A USD holding a NuRec volume.
        budget (int or None): Most gaussians to keep. None or 0 keeps all.
        on_progress (callable or None): Receives status strings.

    Returns:
        dict: ``centres`` (N, 3) float32, ``cov`` (N, 6) float32, ``colour``
        (N, 4) uint8 (rgb plus opacity), ``bounds``, ``total``, ``kept``,
        ``dropped_faint``, ``budget`` and ``volume``.

    Raises:
        SplatReadError: As :func:`read_nurec`.
    """
    raw = read_nurec(usd_path, on_progress=on_progress)
    total = raw["count"]
    if on_progress:
        on_progress(f"{total:,} gaussians; deriving colour and covariance")

    densities, scales = raw["densities"], raw["scales"]
    opacity = 1.0 / (1.0 + np.exp(-densities[:, 0].astype(np.float32)))

    # Gaussians whose alpha rounds to zero are dropped before the budget is
    # spent rather than counted against it.
    visible = opacity >= _MIN_OPACITY
    dropped_faint = int(total - int(visible.sum()))
    index = np.flatnonzero(visible)

    limit = int(budget or 0)
    if limit and len(index) > limit:
        if on_progress:
            on_progress(f"thinning {len(index):,} gaussians to the {limit:,} budget")
        # Thinned on the gaussians' own coordinates: the volume prim's rigid
        # transform only changes which cell a gaussian lands in.
        index = index[_thin(
            raw["positions"][index].astype(np.float32),
            _importance(scales[index], densities[index]),
            limit,
        )]

    transform = raw["volume"]["transform"]
    # USD matrices are row-vector (world = local * M); transposing the linear
    # block gives the column-vector map used here and in the browser.
    linear = np.asarray(transform[:3, :3], dtype=np.float64).T
    offset = np.asarray(transform[3, :3], dtype=np.float64)

    positions = raw["positions"][index].astype(np.float32)
    centres = (positions @ linear.T.astype(np.float32)
               + offset.astype(np.float32)).astype(np.float32)
    cov = _covariances(scales[index], raw["rotations"][index], linear)

    colour = np.empty((len(index), 4), dtype=np.uint8)
    rgb = 0.5 + SH_C0 * raw["features_albedo"][index].astype(np.float32)
    np.clip(rgb, 0.0, 1.0, out=rgb)
    colour[:, :3] = np.round(rgb * 255.0).astype(np.uint8)
    colour[:, 3] = np.round(np.clip(opacity[index], 0.0, 1.0) * 255.0).astype(np.uint8)

    low = centres.min(axis=0) if len(centres) else np.zeros(3, dtype=np.float32)
    high = centres.max(axis=0) if len(centres) else np.zeros(3, dtype=np.float32)
    return {
        "centres": centres,
        "cov": cov,
        "colour": colour,
        "bounds": {"min": [float(v) for v in low], "max": [float(v) for v in high]},
        "total": total,
        "kept": int(len(index)),
        "dropped_faint": dropped_faint,
        "budget": limit or None,
        "near_clip": raw["near_clip"],
        "volume": raw["volume"],
        "version": raw["version"],
    }


def load_centres(usd_path, *, budget=DEFAULT_SPLAT_BUDGET):
    """Just the gaussian centres, as a point cloud in the room's own frame.

    Serves surface queries the way a mesh room's vertices do: enough to answer
    "is there room under this prop, and how far below".
    """
    raw = read_nurec(usd_path, want=("positions", "densities"))
    opacity = 1.0 / (1.0 + np.exp(-raw["densities"][:, 0].astype(np.float32)))
    index = np.flatnonzero(opacity >= _MIN_OPACITY)
    limit = int(budget or 0)
    if limit and len(index) > limit:
        # A stride samples the whole room evenly; taking the first N would not.
        index = index[:: max(1, len(index) // limit)][:limit]
    transform = raw["volume"]["transform"]
    linear = np.asarray(transform[:3, :3], dtype=np.float32).T
    offset = np.asarray(transform[3, :3], dtype=np.float32)
    return raw["positions"][index].astype(np.float32) @ linear.T + offset


# --------------------------------------------------------------------------
# The browser proxy
# --------------------------------------------------------------------------


def write_splat_file(path, gaussians):
    """Write gaussians as a ``.splat`` the browser can map onto typed arrays.

    Layout, little-endian throughout::

        0   8   magic "SFSPLAT\\x01"
        8   4   uint32 header length
        12  n   header JSON (utf-8), zero-padded to a 16-byte boundary
        ..      float32 centres[count][3]
        ..      float32 covariance[count][6]   xx xy xz yy yz zz
        ..      uint8   colour[count][4]       r g b opacity

    Blocks are planar so each maps to one typed-array view; every block starts
    on a 16-byte boundary (a ``Float32Array`` needs a 4-byte-aligned offset).

    Args:
        path (str or Path): Destination.
        gaussians (dict): From :func:`load_gaussians`.

    Returns:
        int: Bytes written.
    """
    centres = np.ascontiguousarray(gaussians["centres"], dtype=np.float32)
    cov = np.ascontiguousarray(gaussians["cov"], dtype=np.float32)
    colour = np.ascontiguousarray(gaussians["colour"], dtype=np.uint8)
    count = len(centres)

    header = json.dumps({
        "count": count,
        "bounds": gaussians["bounds"],
        "total": gaussians["total"],
        "kept": gaussians["kept"],
        "droppedFaint": gaussians["dropped_faint"],
        "budget": gaussians["budget"],
        "nearClip": gaussians.get("near_clip"),
        "nurecVersion": gaussians.get("version"),
        # Named so a reader can refuse a layout it does not know.
        "layout": ["centres:f32x3", "cov:f32x6", "colour:u8x4"],
    }, sort_keys=True).encode("utf-8")
    padding = (-(len(SPLAT_MAGIC) + 4 + len(header))) % 16

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "wb") as handle:
        for block in (SPLAT_MAGIC, struct.pack("<I", len(header) + padding),
                      header, b"\0" * padding,
                      centres.tobytes(), cov.tobytes(), colour.tobytes()):
            handle.write(block)
            written += len(block)
    return written


def read_splat_file(path):
    """Read back a ``.splat``; the inverse of :func:`write_splat_file`."""
    blob = Path(path).read_bytes()
    if blob[:8] != SPLAT_MAGIC:
        raise SplatReadError(f"{path} is not a .splat file")
    header_len = struct.unpack("<I", blob[8:12])[0]
    header = json.loads(blob[12:12 + header_len].rstrip(b"\0").decode("utf-8"))
    count = int(header["count"])
    at = 12 + header_len
    centres = np.frombuffer(blob, dtype=np.float32, count=count * 3, offset=at)
    at += count * 12
    cov = np.frombuffer(blob, dtype=np.float32, count=count * 6, offset=at)
    at += count * 24
    colour = np.frombuffer(blob, dtype=np.uint8, count=count * 4, offset=at)
    return {
        "header": header,
        "centres": centres.reshape(count, 3),
        "cov": cov.reshape(count, 6),
        "colour": colour.reshape(count, 4),
    }


def build_splat_proxy(usd_path, out_dir, name, *, budget=DEFAULT_SPLAT_BUDGET,
                      on_progress=None):
    """Convert a NuRec room to a ``.splat`` and describe the result.

    The gaussian counterpart of :func:`extract.build_proxy`; it returns the
    same manifest fields so the two are interchangeable to the caller. ``glb``
    is always None: there is no mesh.

    Args:
        usd_path (Path): USD holding a NuRec volume.
        out_dir (Path): Directory the ``.splat`` is written to.
        name (str): Filename for the proxy, relative to *out_dir*.
        budget (int or None): Gaussian budget; see :data:`DEFAULT_SPLAT_BUDGET`.
        on_progress (callable or None): Receives status strings.

    Returns:
        dict: Manifest fields -- ``splat``, ``splatCount``, ``splatTotal``,
        ``splatBudget``, ``splatBytes``, ``glb`` (None), ``status``, ``error``,
        ``upAxis``, ``metersPerUnit``, ``nativeSize``, ``nativeCentre``,
        ``scaleFidelity``, ``verts``, ``faces``, ``textured``,
        ``prim_failures`` and ``sourceMtimeNs``.
    """
    usd_path, out_dir = Path(usd_path), Path(out_dir)
    gaussians = load_gaussians(usd_path, budget=budget, on_progress=on_progress)
    written = write_splat_file(out_dir / name, gaussians)

    low = np.asarray(gaussians["bounds"]["min"], dtype=float)
    high = np.asarray(gaussians["bounds"]["max"], dtype=float)
    size = high - low
    centre = (high + low) / 2.0

    try:
        mtime = usd_path.stat().st_mtime_ns
    except OSError:
        mtime = None

    return {
        "glb": None,
        "splat": name,
        "splatCount": gaussians["kept"],
        "splatTotal": gaussians["total"],
        "splatBudget": gaussians["budget"],
        "splatBytes": written,
        "splatBounds": gaussians["bounds"],
        "upAxis": gaussians["volume"]["upAxis"],
        "metersPerUnit": gaussians["volume"]["metersPerUnit"],
        "status": "ok",
        "error": None,
        "prim_failures": [],
        # A gaussian cloud has no vertices or faces; zero is the honest count.
        "textured": True,
        "verts": 0,
        "faces": 0,
        "nativeSize": [round(float(v), 6) for v in size],
        "nativeCentre": [round(float(v), 6) for v in centre],
        # Exactly linear: OmniGibson loads a NuRec room as one visual-only prim
        # with no links to scale separately.
        "scaleFidelity": {"kind": "linear", "rootScale": None, "offsetLinks": []},
        "sourceMtimeNs": mtime,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert a NuRec Gaussian-splat room to a browser .splat proxy",
    )
    parser.add_argument("usd", help="Path to a NuRec .usdz (or .usd)")
    parser.add_argument("--out", default=None, help="Output .splat (default: beside the USD)")
    parser.add_argument(
        "--budget", type=int, default=DEFAULT_SPLAT_BUDGET,
        help=f"Most gaussians to keep; 0 keeps all (default: {DEFAULT_SPLAT_BUDGET})",
    )
    parser.add_argument("--info", action="store_true", help="Report and write nothing")
    args = parser.parse_args()

    usd_path = Path(args.usd).expanduser().resolve()
    if not usd_path.is_file():
        sys.exit(f"ERROR: not found: {usd_path}")

    volume = nurec_volume(usd_path)
    if volume is None:
        sys.exit(f"ERROR: {usd_path.name} carries no NuRec gaussian volume")
    print(f"{usd_path.name}: NuRec volume at {volume['prim']}"
          f" (up={volume['upAxis']}, m/unit={volume['metersPerUnit']:g})")

    if args.info:
        raw = read_nurec(usd_path, want=("positions",))
        print(f"  {raw['count']:,} gaussians, nre {raw['version']}")
        return 0

    out = Path(args.out) if args.out else usd_path.with_suffix(".splat")
    try:
        proxy = build_splat_proxy(
            usd_path, out.parent, out.name, budget=args.budget,
            on_progress=lambda msg: print(f"  {msg}"),
        )
    except SplatReadError as e:
        sys.exit(f"ERROR: {e}")

    kept, total = proxy["splatCount"], proxy["splatTotal"]
    thinned = "" if kept == total else f" (thinned from {total:,})"
    size = " x ".join(f"{v:.2f}" for v in proxy["nativeSize"])
    print(f"\nWrote {out}  {proxy['splatBytes'] / 1e6:.1f} MB"
          f"  {kept:,} gaussians{thinned}  {size} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
