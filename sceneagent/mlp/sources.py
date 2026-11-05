"""Download one checked object from YCB, one from GSO, and one from ABO.

This is the first gate. It stops after a single passing object from each set.
It does not download the rest of a pilot, call Gemini, or train.

YCB objects come from the object-model tarballs, not a scene dump.
GSO objects come from the Google Research Fuel collection.
ABO objects are glTF models from the public ABO index, not listing-only rows.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import shutil
import urllib.error
import urllib.request
import zipfile
import tarfile
from dataclasses import dataclass
from pathlib import Path

from sceneagent.mlp.check import CheckResult, check_sample, rgb_image_problem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "mlp"

USER_AGENT = "sceneagent-mlp-gate/0.1"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

YCB_DATA = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
YCB_SET_PAGE = "https://www.ycbbenchmarks.com/object-models/"
GSO_COLLECTION = (
    "https://fuel.gazebosim.org/1.0/GoogleResearch/collections/"
    "Scanned%20Objects%20by%20Google%20Research/models"
)
GSO_SET_PAGE = (
    "https://app.gazebosim.org/GoogleResearch/fuel/collections/"
    "Scanned%20Objects%20by%20Google%20Research"
)
ABO_MODELS_CSV = "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/metadata/3dmodels.csv.gz"
ABO_INDEX = "https://amazon-berkeley-objects.s3.amazonaws.com/index.html"
ABO_ASSET_ROOT = "https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/"
ABO_LISTING_SHARD = "https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_{shard}.json.gz"

MAX_ATTEMPTS = 3
MAX_ABO_BYTES = 40 * 1024 * 1024
MAX_EXTRACT_BYTES = 400 * 1024 * 1024
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
SOURCES = ("ycb", "gso", "abo")
PILOT_COUNTS = {"ycb": 8, "gso": 6, "abo": 6}
# The three checked meshes stay. 001_chips_can stays skipped.
PROTECTED_IDS = frozenset({
    "002_master_chef_can",
    "Weisshai_Great_White_Shark",
    "B079TXJCR5",
})
SKIP_IDS = frozenset({"001_chips_can"})
# One Gaussian per triangle. A million-face glTF is not a pilot row.
MAX_PILOT_FACES = 80000


@dataclass(frozen=True)
class Candidate:
    source: str
    id: str
    name: str
    source_url: str
    download_url: str
    kind: str  # "archive" or "mesh"


def download_one_from_each(root: Path | None = None, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Download and check one object from each source.

    A candidate that fails the check is deleted and replaced, up to
    ``max_attempts`` downloads per source. Objects that are absent from the
    publisher (HTTP 404, or an ABO glTF larger than the gate limit) are
    skipped and do not consume an attempt.
    """

    root = Path(root) if root is not None else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    samples = []
    rejected = []
    for source in SOURCES:
        sample, failures = _acquire_one(source, root, max_attempts)
        rejected.extend(failures)
        if sample is not None:
            samples.append(sample)
    manifest = {"samples": samples, "rejected": rejected}
    manifest_path = root / "samples_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _acquire_one(source: str, root: Path, max_attempts: int):
    failures = []
    attempts = 0
    for candidate in _candidates(source):
        if attempts >= max_attempts:
            break
        attempts += 1
        object_dir = root / source / _safe_id(candidate.id)
        if object_dir.exists():
            shutil.rmtree(object_dir)
        object_dir.mkdir(parents=True)
        print(f"[{source}] attempt {attempts}/{max_attempts}: {candidate.id} {candidate.download_url}")
        try:
            mesh_path, image_path, name = _install(candidate, object_dir)
        except Exception as exc:
            reason = f"download failed: {exc}"
            print(f"[{source}] {candidate.id} {reason}")
            failures.append(_rejection(candidate, reason))
            shutil.rmtree(object_dir, ignore_errors=True)
            continue
        sample = {
            "id": candidate.id,
            "source": candidate.source,
            "name": name or candidate.name,
            "source_url": candidate.source_url,
            "mesh_path": str(mesh_path.resolve()),
            "image_path": str(image_path.resolve()) if image_path else "",
        }
        manifest_row = {"id": candidate.id, "source": candidate.source}
        result = check_sample(sample, manifest_row)
        if not result.ok:
            reason = "; ".join(result.reasons)
            print(f"[{source}] {candidate.id} failed check: {reason}")
            failures.append(_rejection(candidate, reason))
            shutil.rmtree(object_dir, ignore_errors=True)
            continue
        sample["name"] = _display_name(candidate, sample["name"], mesh_path)
        sample["extent_m"] = _extent_meters(candidate.source, result)
        sample["check_status"] = result.status
        print(f"[{source}] {candidate.id} passed, extent_m={sample['extent_m']}")
        return sample, failures
    return None, failures


def download_pilot(root=None, counts=None, accept=None, protected_samples=None):
    """Finish the pilot counts. A failed download or mesh check is replaced.

    ``protected_samples`` are the objects already on disk. They are not
    downloaded again and they are not deleted. ``accept(sample)`` returns an
    error string when a new object must be replaced, or ``None`` to keep it.
    """

    root = Path(root) if root is not None else DEFAULT_ROOT
    counts = dict(PILOT_COUNTS if counts is None else counts)
    root.mkdir(parents=True, exist_ok=True)
    kept = {source: [] for source in SOURCES}
    rejected = []
    have = set()
    for sample in protected_samples or []:
        source = sample.get("source")
        if source not in kept or sample.get("id") in have:
            continue
        kept[source].append(sample)
        have.add(sample["id"])
    for source in SOURCES:
        if len(kept[source]) >= counts[source]:
            continue
        for candidate in _candidates(source):
            if len(kept[source]) >= counts[source]:
                break
            if candidate.id in have or candidate.id in SKIP_IDS:
                continue
            print(
                f"[{source}] {len(kept[source]) + 1}/{counts[source]} {candidate.id}",
                flush=True,
            )
            sample, rejection = attempt_download(candidate, root)
            if rejection is not None:
                print(f"[{source}] {candidate.id} rejected: {rejection['reason']}", flush=True)
                rejected.append(rejection)
                continue
            if accept is not None:
                error = accept(sample)
                if error:
                    print(f"[{source}] {candidate.id} replaced: {error}", flush=True)
                    rejected.append(_rejection(candidate, str(error)))
                    if candidate.id not in PROTECTED_IDS:
                        shutil.rmtree(root / source / _safe_id(candidate.id), ignore_errors=True)
                    continue
            kept[source].append(sample)
            have.add(sample["id"])
            print(f"[{source}] {candidate.id} kept", flush=True)
    samples = [row for source in SOURCES for row in kept[source]]
    return {"samples": samples, "rejected": rejected}


def attempt_download(candidate, root):
    """Install one candidate and run the mesh suitability check.

    Returns ``(sample, None)`` or ``(None, rejection)``. A protected id is
    never deleted.
    """

    if candidate.id in PROTECTED_IDS:
        raise RuntimeError(f"refusing to replace protected object {candidate.id}")
    if candidate.source == "abo" and candidate.kind == "mesh":
        length = _content_length(candidate.download_url)
        if length is not None and (length < 1 or length > MAX_ABO_BYTES):
            return None, _rejection(
                candidate, f"glTF byte size {length} is outside the mesh limit"
            )
    object_dir = Path(root) / candidate.source / _safe_id(candidate.id)
    if object_dir.exists():
        shutil.rmtree(object_dir)
    object_dir.mkdir(parents=True)
    try:
        mesh_path, image_path, name = _install(candidate, object_dir)
    except Exception as exc:
        shutil.rmtree(object_dir, ignore_errors=True)
        return None, _rejection(candidate, f"download failed: {exc}")
    sample = {
        "id": candidate.id,
        "source": candidate.source,
        "name": name or candidate.name,
        "source_url": candidate.source_url,
        "mesh_path": str(mesh_path.resolve()),
        "image_path": str(image_path.resolve()) if image_path else "",
    }
    result = check_sample(sample, {"id": candidate.id, "source": candidate.source})
    if not result.ok:
        reason = "; ".join(result.reasons)
        shutil.rmtree(object_dir, ignore_errors=True)
        return None, _rejection(candidate, reason)
    sample["name"] = _display_name(candidate, sample["name"], mesh_path)
    sample["extent_m"] = _extent_meters(candidate.source, result)
    sample["check_status"] = result.status
    return sample, None


def _candidates(source: str):
    if source == "ycb":
        yield from _ycb_candidates()
    elif source == "gso":
        yield from _gso_candidates()
    elif source == "abo":
        yield from _abo_candidates()
    else:
        raise ValueError(f"unknown source {source}")


def _ycb_candidates():
    payload = json.loads(_read_bytes(YCB_DATA + "objects.json"))
    for object_id in payload["objects"]:
        download_url = YCB_DATA + f"google/{object_id}_google_16k.tgz"
        if not _url_available(download_url):
            continue
        yield Candidate(
            source="ycb",
            id=object_id,
            name=_ycb_name(object_id),
            source_url=download_url,
            download_url=download_url,
            kind="archive",
        )


def _gso_candidates(per_page=20, max_pages=20):
    # Publisher order, restricted to Google Research scans. Pages continue
    # until the caller has enough passing objects.
    seen = set()
    for page in range(1, max_pages + 1):
        url = f"{GSO_COLLECTION}?page={page}&per_page={per_page}"
        try:
            models = json.loads(_read_bytes(url))
        except Exception as exc:
            print(f"[gso] page {page} failed: {exc}")
            break
        if not isinstance(models, list) or not models:
            break
        added = 0
        for model in models:
            if model.get("owner") != "GoogleResearch":
                continue
            name = model["name"]
            if name in seen:
                continue
            seen.add(name)
            added += 1
            description = (model.get("description") or "").strip().splitlines()
            label = description[0].strip() if description and description[0].strip() else name.replace("_", " ")
            yield Candidate(
                source="gso",
                id=name,
                name=label,
                source_url=f"https://app.gazebosim.org/GoogleResearch/fuel/models/{name}",
                download_url=f"https://fuel.gazebosim.org/1.0/GoogleResearch/models/{name}.zip",
                kind="archive",
            )
        if added == 0:
            break


def _abo_candidates():
    raw = _read_bytes(ABO_MODELS_CSV)
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode("utf-8"))))
    ranked = []
    for row in rows:
        rel = row.get("path") or ""
        if not rel.lower().endswith((".glb", ".gltf")):
            continue
        try:
            faces = int(row["faces"])
            images = int(row["images"])
            textures = int(row["textures"])
            side = max(int(row["image_width_max"]), int(row["image_height_max"]))
            extent = tuple(float(row[key]) for key in ("extent_x", "extent_y", "extent_z"))
        except (KeyError, TypeError, ValueError):
            continue
        if faces < 100 or faces > MAX_PILOT_FACES or images < 1 or textures < 1 or side > 4096:
            continue
        if not all(math_finite_positive(value) and value < 5 for value in extent):
            continue
        ranked.append((side, faces, row["3dmodel_id"], rel))
    ranked.sort()
    for _side, _faces, model_id, rel in ranked:
        download_url = ABO_ASSET_ROOT + rel
        length = _content_length(download_url)
        if length is None or length < 1 or length > MAX_ABO_BYTES:
            continue
        yield Candidate(
            source="abo",
            id=model_id,
            name=model_id,
            source_url=download_url,
            download_url=download_url,
            kind="mesh",
        )


def _install(candidate: Candidate, object_dir: Path):
    if candidate.kind == "mesh":
        mesh_path = object_dir / Path(candidate.download_url).name
        _download(candidate.download_url, mesh_path)
        image_path = _extract_glb_image(mesh_path, object_dir)
        return mesh_path, image_path, candidate.name
    archive = object_dir / ("download.zip" if candidate.download_url.endswith(".zip") else "download.tgz")
    _download(candidate.download_url, archive)
    _extract(archive, object_dir)
    archive.unlink(missing_ok=True)
    _flatten_single_root(object_dir)
    mesh_path = _select_mesh(object_dir)
    image_path = _select_image(object_dir)
    if mesh_path is None:
        raise ValueError("archive did not contain an OBJ, glTF, or SDF")
    return mesh_path, image_path, candidate.name


def _display_name(candidate: Candidate, current: str, mesh_path: Path) -> str:
    if candidate.source != "abo":
        return current
    label = _glb_label(mesh_path)
    if label and label != candidate.id:
        return label
    # The stratified queue already stores the listing name. Scanning every
    # ABO shard again spends tens of seconds and does not change that name.
    if current and current != candidate.id:
        return current
    listing = _lookup_abo_name(candidate.id)
    return listing or current


def _extent_meters(source: str, result: CheckResult):
    extent = result.extent_m
    if extent is None:
        return None
    scale = 1.0
    # The google_16k tarballs are published in meters. A bounding box of
    # several meters on a YCB household object means the file was stored in
    # millimeters. GSO (SDF) and ABO (glTF) are meters by their formats.
    if source == "ycb" and max(extent) > 5:
        scale = 0.001
    return [round(float(value) * scale, 6) for value in extent]


def _rejection(candidate: Candidate, reason: str) -> dict:
    return {
        "id": candidate.id,
        "source": candidate.source,
        "name": candidate.name,
        "source_url": candidate.source_url,
        "reason": reason,
    }


def _ycb_name(object_id: str) -> str:
    parts = object_id.split("_")
    if parts and re.fullmatch(r"[0-9]+(-[a-z])?", parts[0]):
        parts = parts[1:]
    elif parts:
        parts = [re.sub(r"^[0-9]+-", "", parts[0]), *parts[1:]]
    name = " ".join(part.replace("-", " ") for part in parts if part)
    return " ".join(name.split()) or object_id


def _select_mesh(root: Path) -> Path | None:
    sdfs = _files_with(root, ".sdf")
    if sdfs:
        return sdfs[0]
    objs = _files_with(root, ".obj")
    textured = [path for path in objs if path.name == "textured.obj"]
    if textured:
        return textured[0]
    if objs:
        return objs[0]
    glbs = _files_with(root, ".glb") + _files_with(root, ".gltf")
    return glbs[0] if glbs else None


def _select_image(root: Path) -> Path | None:
    images = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if _ignored(path):
            continue
        images.append(path)

    def rank(path: Path):
        text = path.as_posix().lower()
        if "texture" in text:
            priority = 0
        elif "thumb" in text:
            priority = 2
        else:
            priority = 1
        return (priority, -path.stat().st_size)

    for path in sorted(images, key=rank):
        if rgb_image_problem(path) is None:
            return path
    return None


def _files_with(root: Path, suffix: str) -> list[Path]:
    found = [
        path
        for path in root.rglob(f"*{suffix}")
        if path.is_file() and path.suffix.lower() == suffix and not _ignored(path)
    ]
    return sorted(found, key=lambda path: (len(path.parts), str(path)))


def _ignored(path: Path) -> bool:
    return any(part == "__MACOSX" or part.startswith(".") for part in path.parts)


def _extract(archive: Path, dest: Path) -> None:
    if zipfile.is_zipfile(archive):
        _extract_zip(archive, dest)
        return
    if tarfile.is_tarfile(archive):
        _extract_tar(archive, dest)
        return
    raise ValueError("download is not a zip or tar archive")


def _extract_zip(archive: Path, dest: Path) -> None:
    dest = dest.resolve()
    total = 0
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            target = _archive_target(dest, info.filename)
            if info.is_dir() or info.filename.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            total = _copy_member(handle.open(info), target, info.file_size, total)


def _extract_tar(archive: Path, dest: Path) -> None:
    dest = dest.resolve()
    total = 0
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"archive link is not allowed: {member.name}")
            target = _archive_target(dest, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            source = handle.extractfile(member)
            if source is None:
                continue
            total = _copy_member(source, target, member.size, total)


def _copy_member(source, target: Path, declared: int, total: int) -> int:
    if declared > MAX_EXTRACT_BYTES:
        raise ValueError("archive member is larger than the extract cap")
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source, target.open("wb") as output:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            written += len(block)
            total += len(block)
            if written > MAX_EXTRACT_BYTES or total > MAX_EXTRACT_BYTES:
                raise ValueError("archive expands past the extract cap")
            output.write(block)
    return total


def _archive_target(dest: Path, name: str) -> Path:
    if _unsafe_archive_name(name):
        raise ValueError(f"unsafe archive path: {name}")
    target = (dest / name).resolve()
    if not target.is_relative_to(dest):
        raise ValueError(f"unsafe archive path: {name}")
    return target


def _unsafe_archive_name(name: str) -> bool:
    if not name or name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
        return True
    return any(part == ".." for part in Path(name).parts)


def _flatten_single_root(dest: Path) -> None:
    entries = [path for path in dest.iterdir() if not path.name.endswith((".tgz", ".zip"))]
    directories = [path for path in entries if path.is_dir()]
    files = [path for path in entries if path.is_file()]
    if files or len(directories) != 1:
        return
    inner = directories[0]
    for child in inner.iterdir():
        shutil.move(str(child), str(dest / child.name))
    inner.rmdir()


def _extract_glb_image(mesh_path: Path, dest: Path) -> Path | None:
    if mesh_path.suffix.lower() != ".glb":
        return _select_image(dest)
    data = mesh_path.read_bytes()
    if data[:4] != b"glTF":
        return None
    document, binary = _glb_chunks(data)
    if not document:
        return None
    images = document.get("images") or []
    views = document.get("bufferViews") or []
    blobs = []
    for image in images:
        if "bufferView" in image:
            view = views[image["bufferView"]]
            start = int(view.get("byteOffset", 0))
            end = start + int(view["byteLength"])
            blobs.append(binary[start:end])
        elif isinstance(image.get("uri"), str) and image["uri"].startswith("data:"):
            blobs.append(_data_uri(image["uri"]))
    blobs.sort(key=len, reverse=True)
    for blob in blobs:
        suffix = _image_suffix(blob)
        if suffix is None:
            continue
        path = dest / f"texture{suffix}"
        path.write_bytes(blob)
        if rgb_image_problem(path) is None:
            return path
        path.unlink(missing_ok=True)
    return None


def _glb_chunks(data: bytes):
    import struct

    if len(data) < 12:
        return None, b""
    magic, _version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        return None, b""
    offset = 12
    document = None
    binary = b""
    while offset + 8 <= min(len(data), length):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.decode("utf-8").rstrip("\x00"))
        elif chunk_type == 0x004E4942:
            binary = chunk
    return document, binary


def _glb_label(mesh_path: Path) -> str | None:
    if mesh_path.suffix.lower() != ".glb":
        return None
    document, _binary = _glb_chunks(mesh_path.read_bytes())
    if not document:
        return None
    extras = (document.get("asset") or {}).get("extras") or {}
    for key in ("name", "title", "product_name", "item_name"):
        value = extras.get(key)
        if isinstance(value, str) and _useful_label(value):
            return value.strip()
    for scene in document.get("scenes") or []:
        name = scene.get("name")
        if isinstance(name, str) and _useful_label(name) and name.strip() != mesh_path.stem:
            return name.strip()
    return None


_GENERIC_GLTF_NAMES = {
    "scene",
    "default",
    "root",
    "root scene",
    "node",
    "mesh",
    "untitled",
    "scene root",
}


def _useful_label(name: str) -> bool:
    return name.strip().lower() not in _GENERIC_GLTF_NAMES


def _data_uri(uri: str) -> bytes:
    import base64

    _header, payload = uri.split(",", 1)
    return base64.b64decode(payload)


def _image_suffix(blob: bytes) -> str | None:
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if blob.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if len(blob) >= 12 and blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return ".webp"
    return None


def _lookup_abo_name(model_id: str) -> str | None:
    needle = model_id.encode("utf-8")
    first = model_id[-1].lower() if model_id else ""
    shards = [str(i) for i in range(10)] + list("abcdef")
    if first in shards:
        shards = [first, *[shard for shard in shards if shard != first]]
    for shard in shards:
        url = ABO_LISTING_SHARD.format(shard=shard)
        print(f"[abo] looking up product name in listings_{shard}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                with gzip.GzipFile(fileobj=response) as uncompressed:
                    for line in uncompressed:
                        if needle not in line:
                            continue
                        record = json.loads(line)
                        if record.get("item_id") != model_id and record.get("3dmodel_id") != model_id:
                            continue
                        name = _english_name(record.get("item_name"))
                        if name:
                            return name
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            print(f"[abo] name lookup skipped listings_{shard}: {exc}")
        except Exception as exc:
            print(f"[abo] name lookup skipped listings_{shard}: {exc}")
    return None


def _english_name(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, list):
        return None
    fallback = None
    for entry in value:
        if not isinstance(entry, dict):
            continue
        text = entry.get("value")
        if not isinstance(text, str) or not text.strip():
            continue
        tag = str(entry.get("language_tag") or "").lower()
        if tag.startswith("en"):
            return text.strip()
        if fallback is None:
            fallback = text.strip()
    return fallback


def _download(url: str, dest: Path) -> None:
    last_error = None
    for _try in range(2):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as output:
                shutil.copyfileobj(response, output)
            if dest.stat().st_size < 1:
                raise ValueError("empty download")
            return
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                raise
            last_error = exc
    raise last_error


def _url_available(url: str) -> bool:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 400:
                return False
            ctype = (response.headers.get("Content-Type") or "").lower()
            if "html" in ctype:
                return False
            length = response.headers.get("Content-Length")
            return length is None or int(length) > 0
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404, 405):
            return False
        raise


def _content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
    except urllib.error.HTTPError:
        return None


def _read_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"unsafe object id: {value}")
    return cleaned


def math_finite_positive(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf")) and value > 0


def main() -> None:
    download_one_from_each(DEFAULT_ROOT, MAX_ATTEMPTS)


if __name__ == "__main__":
    main()
