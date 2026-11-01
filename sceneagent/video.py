"""Read a SimFoundry example video far enough to record what phase 1 can know.

Section 3.2 starts from an RGB sequence and then runs COLMAP and splatfacto.
This module opens the mp4, checks the file type, and reads the movie header
when the box layout is the common version-0 `mvhd`. It does not estimate
cameras and it does not train a splat. Those stay on the COLMAP and splatfacto
rows of the gap matrix.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _boxes(data, start, end):
    cursor = start
    while cursor + 8 <= end:
        size = int.from_bytes(data[cursor:cursor + 4], "big")
        kind = data[cursor + 4:cursor + 8]
        header = 8
        if size == 1:
            if cursor + 16 > end:
                break
            size = int.from_bytes(data[cursor + 8:cursor + 16], "big")
            header = 16
        elif size == 0:
            size = end - cursor
        if size < header or cursor + size > end:
            break
        yield kind, cursor + header, cursor + size
        cursor += size


def _find(data, start, end, target):
    for kind, box_start, box_end in _boxes(data, start, end):
        if kind == target:
            return box_start, box_end
        if kind in {b"moov", b"trak", b"mdia", b"minf", b"stbl"}:
            found = _find(data, box_start, box_end, target)
            if found:
                return found
    return None


def _mvhd(payload):
    if len(payload) < 20:
        return None
    version = payload[0]
    if version == 0:
        timescale = int.from_bytes(payload[12:16], "big")
        duration = int.from_bytes(payload[16:20], "big")
    elif version == 1 and len(payload) >= 32:
        timescale = int.from_bytes(payload[20:24], "big")
        duration = int.from_bytes(payload[24:32], "big")
    else:
        return None
    if timescale <= 0:
        return None
    return {"timescale": timescale, "duration_ticks": duration, "duration_s": duration / timescale}


def inspect_video(path):
    """Return file facts and, when present, the movie-header duration."""
    file_path = Path(path)
    data = file_path.read_bytes()
    if len(data) < 12 or data[4:8] != b"ftyp":
        raise ValueError(f"{file_path.name} is not an mp4 file")
    major = data[8:12].decode("ascii", errors="replace")
    record = {
        "path": str(file_path),
        "name": file_path.name,
        "bytes": len(data),
        "container": "mp4",
        "major_brand": major,
        "duration_s": None,
        "timescale": None,
        "ffprobe": None,
        "reconstruction": "not_run",
        "note": "RGB file only. COLMAP poses and splatfacto were not run.",
    }
    found = _find(data, 0, len(data), b"mvhd")
    if found:
        start, end = found
        parsed = _mvhd(data[start:end])
        if parsed:
            record["duration_s"] = parsed["duration_s"]
            record["timescale"] = parsed["timescale"]
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        record["ffprobe"] = _ffprobe(ffprobe, file_path)
    return record


def _ffprobe(binary, file_path):
    completed = subprocess.run(
        [
            binary, "-v", "error", "-print_format", "json",
            "-show_entries", "format=duration:stream=width,height,codec_type,nb_frames",
            str(file_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"available": True, "ok": False, "stderr": completed.stderr[-400:]}
    payload = json.loads(completed.stdout or "{}")
    return {"available": True, "ok": True, "payload": payload}


def run_video(path, out_dir):
    """Write the video manifest and a ledger that stops before COLMAP."""
    from sceneagent.catalog import rows
    from sceneagent.external_tools import probe_environment

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = inspect_video(path)
    (out / "video.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    probe = probe_environment()
    (out / "tool_probe.json").write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
    colmap = next(item for item in probe if item["sub_block"] == "COLMAP camera poses")
    ledger = {
        "scene_name": Path(path).stem,
        "source": "simfoundry_example_video",
        "entries": [
            {
                "item": "video",
                "decision": "accept",
                "bytes": info["bytes"],
                "duration_s": info["duration_s"],
            },
            {
                "item": "COLMAP camera poses",
                "decision": "deferred",
                "available": colmap["available"],
                "reason": "Section 3.2 needs COLMAP. The probe records whether the binary is on PATH.",
            },
        ],
    }
    (out / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    deferred = [row["sub_block"] for row in rows() if row["status"] == "missing"]
    return {"video": info, "colmap_available": colmap["available"], "missing_rows": len(deferred), "out_dir": str(out)}
