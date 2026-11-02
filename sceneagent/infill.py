"""Optional plane infill and LaMa for SceneAgent section 3.3.

The paper completes exposed background with detected planes and names LaMa
as optional texture inpainting. ``plane_infill`` still refuses to write an
image when the caller has neither a plane render nor association output.

``complete_background`` treats a missing render as a plane-completion problem.
It can write a plane-fill placeholder from association output. LaMa runs only
when an image to inpaint exists, and only when LaMa itself is installed. A
missing checkpoint does not produce a forged inpaint.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def write_rgb_png(path, rows):
    """Write an 8-bit RGB PNG. ``rows`` is a sequence of RGB byte triples."""
    height = len(rows)
    width = len(rows[0]) if height else 0
    raw = b"".join(b"\x00" + bytes(row) for row in rows)

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(png)
    return destination


def plane_infill(plane_render_path, out_path=None, use_lama=False, association=None):
    """Refuse when no plane render and no association output exist.

    A present render, with ``association`` omitted, still does not run LaMa.
    That is the historical hook. Association output goes through
    ``complete_background`` so the missing piece can be plane completion.
    """
    source = Path(plane_render_path) if plane_render_path else None
    exists = bool(source and source.is_file() and source.stat().st_size > 0)
    destination = Path(out_path) if out_path else None
    if not exists and association:
        return complete_background(association, out_path=out_path, use_lama=use_lama)
    if destination is not None and destination.exists() and (source is None or destination.resolve() != source.resolve()):
        destination.unlink()
    if not exists:
        return {
            "ok": False,
            "emitted": False,
            "artifact": None,
            "lama": "not_started",
            "use_lama": bool(use_lama),
            "status": "open",
            "reason": "no plane render exists",
        }
    return {
        "ok": False,
        "emitted": False,
        "artifact": None,
        "lama": "not_started",
        "use_lama": bool(use_lama),
        "status": "open",
        "plane_render": str(source),
        "reason": "A plane render exists. LaMa and plane infill were not run, so no image was written.",
    }


def _kept_total(association):
    kept = association.get("n_passed_equation_1", association.get("kept"))
    total = association.get("n_gaussians", association.get("total"))
    if kept is None or total is None:
        return None, None
    return float(kept), float(total)


def detect_plane(association):
    """Axis of smallest spread across association centers, when centers exist.

    This is a placeholder detector for the plane-fill image. It is not the
    paper's unpublished plane fitter.
    """
    centers = association.get("centers") or association.get("points") or []
    if len(centers) < 3:
        return None
    sample = list(centers)[:5000]
    axes = list(zip(*sample))
    best = None
    for index, axis in enumerate("xyz"):
        values = [float(value) for value in axes[index]]
        spread = max(values) - min(values)
        offset = sorted(values)[len(values) // 2]
        if best is None or spread < best["spread"]:
            best = {
                "normal_axis": axis,
                "offset": offset,
                "spread": spread,
                "samples": len(sample),
                "method": "ours_smallest_spread_axis",
            }
    return best


def plane_fill_from_association(association, out_path):
    """Write a plane-fill placeholder when association output can support one.

    The file is a flat field with a hole. It is not a LaMa inpaint.
    """
    if not association:
        return {
            "ok": False,
            "kind": None,
            "artifact": None,
            "lama": False,
            "plane": None,
            "reason": "plane completion is the missing piece",
        }
    kept, total = _kept_total(association)
    centers = association.get("centers") or association.get("points")
    if kept is None and not centers:
        return {
            "ok": False,
            "kind": None,
            "artifact": None,
            "lama": False,
            "plane": None,
            "reason": "plane completion is the missing piece",
        }
    plane = detect_plane(association)
    removed = 0.25
    if kept is not None and total:
        removed = min(0.9, max(0.05, 1.0 - (kept / total)))
    side = 16
    hole = max(1, int(round(side * removed)))
    rows = []
    for y in range(side):
        row = []
        for x in range(side):
            if x < hole and y < hole:
                row.extend((0, 0, 0))
            else:
                row.extend((180, 170, 150))
        rows.append(row)
    path = write_rgb_png(out_path, rows)
    return {
        "ok": True,
        "kind": "plane_fill_placeholder",
        "artifact": str(path),
        "lama": False,
        "forged_inpaint": False,
        "plane": plane,
        "removed_fraction": removed,
        "reason": "Plane-fill placeholder from association output. Not a LaMa result.",
    }


def _lama_backend():
    try:
        import simple_lama_inpainting  # noqa: F401
    except ImportError:
        return None, "LaMa weights are not installed"
    return "simple_lama_inpainting", ""


def lama_inpaint(image_path, out_path=None, mask_path=None):
    """Inpaint only when an image exists and LaMa is installed.

    No image in means no image out. Weights absent means the destination is
    not written. The function does not copy the input and call it an inpaint.
    """
    source = Path(image_path) if image_path else None
    destination = Path(out_path) if out_path else None
    exists = bool(source and source.is_file() and source.stat().st_size > 0)
    if destination is not None and destination.exists() and (source is None or destination.resolve() != source.resolve()):
        destination.unlink()
    if not exists:
        return {
            "ok": False,
            "emitted": False,
            "artifact": None,
            "forged": False,
            "status": "no_image",
            "weights_installed": None,
            "lama": "not_called",
            "reason": "no image in, no image out",
        }
    backend, why = _lama_backend()
    if backend is None:
        return {
            "ok": False,
            "emitted": False,
            "artifact": None,
            "forged": False,
            "status": "weights_absent",
            "weights_installed": False,
            "lama": "weights_absent",
            "image_in": str(source),
            "reason": why,
        }
    from simple_lama_inpainting import SimpleLama
    from PIL import Image

    image = Image.open(source).convert("RGB")
    if mask_path and Path(mask_path).is_file():
        mask = Image.open(mask_path).convert("L")
    else:
        mask = Image.new("L", image.size, 0)
        hole = max(1, image.size[0] // 4)
        mask.paste(255, (0, 0, hole, hole))
    result = SimpleLama()(image, mask)
    if destination is None:
        destination = source.with_name(source.stem + "_lama.png")
    result.save(destination)
    return {
        "ok": True,
        "emitted": True,
        "artifact": str(destination),
        "forged": False,
        "status": "inpainted",
        "weights_installed": True,
        "lama": "simple_lama_inpainting",
        "image_in": str(source),
        "reason": "LaMa wrote an inpaint.",
    }


def complete_background(association=None, hole_image=None, out_path=None, use_lama=False):
    """Plane completion first. LaMa only if that step produced an image."""
    image = Path(hole_image) if hole_image else None
    placeholder = None
    if image is None or not image.is_file() or image.stat().st_size == 0:
        image = None
        if association is not None:
            placeholder_path = out_path
            if use_lama or placeholder_path is None:
                placeholder_path = Path(out_path).with_name("plane_fill_placeholder.png") if out_path else Path("plane_fill_placeholder.png")
            placeholder = plane_fill_from_association(association, placeholder_path)
            if placeholder.get("artifact"):
                image = Path(placeholder["artifact"])
    if image is None:
        return {
            "ok": False,
            "emitted": False,
            "artifact": None,
            "forged": False,
            "plane_completion": "missing",
            "lama": "not_called",
            "placeholder": placeholder,
            "reason": "plane completion is the missing piece",
        }
    if not use_lama:
        return {
            "ok": bool(placeholder and placeholder.get("ok")) or image.is_file(),
            "emitted": False,
            "artifact": str(image),
            "forged": False,
            "plane_completion": "placeholder" if placeholder else "image",
            "lama": "not_requested",
            "placeholder": placeholder,
            "reason": "An image exists. LaMa was not requested.",
        }
    painted = lama_inpaint(image, out_path)
    painted["plane_completion"] = "placeholder" if placeholder else "image"
    painted["placeholder"] = placeholder
    return painted
