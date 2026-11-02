"""Section 3.2 readers that do not run COLMAP or download generators.

Still images are a sequence input. The project site says RGB sequences come
"from video or as images". The paper says "RGB image sequences" and figure 7
lists RGB images or video. Those sentences agree that an image sequence is
an input. This reader records the files and does not claim COLMAP ran.

Existing 3DGS, photogrammetry, LiDAR, and generated 3DGS files are accepted
only when the file is already on disk. World Labs Marble, NVIDIA Lyra, and
Tencent HY-World are not downloaded.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

STILL_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".ppm",
}

ASSET_SUFFIXES = {
    "existing_3dgs": {".ply", ".splat", ".spz", ".ksplat"},
    "photogrammetry": {".obj", ".ply", ".glb", ".gltf", ".fbx", ".usdz"},
    "lidar": {".las", ".laz", ".e57", ".ply", ".pcd"},
    "generated_3dgs": {".ply", ".splat", ".spz"},
}

# Ours. Section 3.2 names per-image color calibration and does not publish
# the correction. A frame is flagged when the L1 distance of its mean RGB
# (0-255) from the per-channel median of the set is greater than 30.
OURS_COLOR_HEURISTIC = "ours_mean_rgb_l1_above_30"
OURS_COLOR_L1_LIMIT = 30.0
INGEST_SKILL = "SCENEAGENT-INGEST-DATA"
_REMOTE_MARKERS = ("marble", "lyra", "hy-world", "hyworld")


def read_still_image_sequence(directory):
    """Record a directory of still images as a sequence.

    Does not sample frames and does not claim COLMAP ran.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    paths = sorted(
        item for item in root.iterdir()
        if item.is_file() and item.suffix.lower() in STILL_IMAGE_SUFFIXES
    )
    return {
        "kind": "still_image_sequence",
        "directory": str(root),
        "count": len(paths),
        "paths": [str(item) for item in paths],
        "colmap_ran": False,
        "reconstruction": "not_run",
        "skill": INGEST_SKILL,
        "decision": "accept" if paths else "reject",
        "paper": "Section 3.2 RGB image sequences",
        "site": "Figure 1: RGB sequences from video or as images",
        "paper_site_disagree": False,
        "note": "Still images recorded as a sequence. COLMAP did not run.",
    }


def _is_remote(path, source_name):
    text = f"{path} {source_name or ''}".lower()
    if "://" in str(path):
        return True
    return any(marker in text for marker in _REMOTE_MARKERS) and not Path(path).is_file()


def ingest_asset(path, kind, out_dir=None, source_name=None):
    """Ledger record for one existing asset. Accept only when the file exists."""
    if kind not in ASSET_SUFFIXES:
        raise ValueError(f"unknown asset kind {kind!r}")
    remote = _is_remote(path, source_name)
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    suffix_ok = suffix in ASSET_SUFFIXES[kind]
    exists = file_path.is_file() and file_path.stat().st_size > 0
    if remote or not exists or not suffix_ok:
        decision = "reject"
    else:
        decision = "accept"
    reason = ""
    if remote:
        reason = "Marble, Lyra, and HY-World are not downloaded."
    elif not exists:
        reason = "file does not exist"
    elif not suffix_ok:
        reason = f"suffix {suffix or '(none)'} is not a {kind} file"
    record = {
        "file_type": kind,
        "suffix": suffix,
        "path": str(file_path),
        "skill": INGEST_SKILL,
        "decision": decision,
        "downloaded": False,
        "generators_contacted": [],
        "reason": reason,
        "note": "Local file only. Marble, Lyra, and HY-World were not downloaded.",
    }
    if out_dir is not None:
        _append_ledger(out_dir, record)
    return record


def _append_ledger(out_dir, record):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ledger.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"section": "3.2", "entries": []}
    payload.setdefault("entries", []).append({
        "item": record["file_type"],
        "decision": record["decision"],
        "skill": record["skill"],
        "path": record["path"],
        "file_type": record["file_type"],
        "downloaded": False,
        "reason": record["reason"],
    })
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def mean_rgb_file(path):
    """Mean RGB in 0-255 for a binary PPM, or for an image Pillow can open."""
    file_path = Path(path)
    data = file_path.read_bytes()
    if data.startswith(b"P6"):
        return _mean_ppm(data)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError(f"{file_path.name} is not a binary PPM and Pillow is not installed") from exc
    image = Image.open(file_path).convert("RGB")
    pixels = list(image.getdata())
    if not pixels:
        raise ValueError(f"{file_path.name} has no pixels")
    count = len(pixels)
    totals = [0.0, 0.0, 0.0]
    for pixel in pixels:
        totals[0] += pixel[0]
        totals[1] += pixel[1]
        totals[2] += pixel[2]
    return tuple(channel / count for channel in totals)


def _mean_ppm(data):
    header, pixel_at = _ppm_header(data)
    width, height, _maxval = header
    count = width * height
    needed = count * 3
    pixels = data[pixel_at:pixel_at + needed]
    if len(pixels) < needed:
        raise ValueError("PPM pixel data is short")
    totals = [0, 0, 0]
    for index in range(0, needed, 3):
        totals[0] += pixels[index]
        totals[1] += pixels[index + 1]
        totals[2] += pixels[index + 2]
    return tuple(channel / count for channel in totals)


def _ppm_header(data):
    if not data.startswith(b"P6"):
        raise ValueError("not a binary PPM")
    tokens = []
    index = 2
    while len(tokens) < 3:
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index >= len(data):
            raise ValueError("PPM header is short")
        if data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < len(data) and data[index] not in b" \t\r\n":
            index += 1
        tokens.append(int(data[start:index]))
    # PPM P6 requires one whitespace byte after maxval. Later bytes are pixels,
    # including values 9, 10, 13, and 32, so they must not be skipped as header.
    if index >= len(data) or data[index] not in b" \t\r\n":
        raise ValueError("PPM maxval is not followed by one whitespace byte")
    return (tokens[0], tokens[1], tokens[2]), index + 1


def color_calibration_report(means):
    """Record per-image mean RGB and flag outliers with the labeled heuristic.

    ``means`` is a sequence of ``(path, (r, g, b))``. The threshold is ours.
    """
    rows = list(means)
    if not rows:
        raise ValueError("color calibration needs at least one image mean")
    channels = list(zip(*(rgb for _path, rgb in rows)))
    median = tuple(float(statistics.median(channel)) for channel in channels)
    frames = []
    for path, rgb in rows:
        l1 = sum(abs(float(channel) - center) for channel, center in zip(rgb, median))
        frames.append({
            "path": str(path),
            "mean_rgb": [float(channel) for channel in rgb],
            "l1_from_median": l1,
            "flagged": l1 > OURS_COLOR_L1_LIMIT,
        })
    return {
        "heuristic": OURS_COLOR_HEURISTIC,
        "limit_l1": OURS_COLOR_L1_LIMIT,
        "median_rgb": list(median),
        "paper_method": False,
        "label": "ours, not the paper's unpublished per-image color correction",
        "frames": frames,
        "flagged_paths": [frame["path"] for frame in frames if frame["flagged"]],
        "colmap_ran": False,
    }


def color_calibration_directory(directory):
    """Mean RGB for still images in a directory. Unreadable files are skipped."""
    sequence = read_still_image_sequence(directory)
    means = []
    skipped = []
    for path in sequence["paths"]:
        try:
            means.append((path, mean_rgb_file(path)))
        except (ValueError, OSError):
            skipped.append(path)
    if not means:
        return {
            "heuristic": OURS_COLOR_HEURISTIC,
            "limit_l1": OURS_COLOR_L1_LIMIT,
            "paper_method": False,
            "label": "ours, not the paper's unpublished per-image color correction",
            "frames": [],
            "flagged_paths": [],
            "skipped_unreadable": skipped,
            "colmap_ran": False,
            "reason": "no image mean could be read",
        }
    report = color_calibration_report(means)
    report["skipped_unreadable"] = skipped
    return report
