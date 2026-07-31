# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


CACHE_SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = ".cache/simfoundry/model_calls"


class RemoteCacheError(RuntimeError):
    """Base error for remote model cache failures."""


class RemoteCacheModeError(RemoteCacheError):
    """Raised when cache mode environment is invalid."""


class RemoteCacheMissError(RemoteCacheError):
    """Raised when TEST_MODE cannot find a requested cached response."""


class RemoteCacheConflictError(RemoteCacheError):
    """Raised when CACHE_MODE sees a conflicting response for an existing key."""


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def file_digest(path: os.PathLike | str) -> Dict[str, Any]:
    file_path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "suffix": file_path.suffix.lower(),
    }


def image_digests(paths: Optional[str | os.PathLike | Iterable[str | os.PathLike]]) -> list[Dict[str, Any]]:
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    return [file_digest(path) for path in paths]


class RemoteModelCache:
    def __init__(self, root_dir: os.PathLike | str, mode: str):
        if mode not in {"off", "cache", "test"}:
            raise ValueError(f"Invalid remote cache mode: {mode}")
        self.root_dir = Path(root_dir)
        self.mode = mode

    @classmethod
    def from_env(cls) -> "RemoteModelCache":
        cache_mode = _env_flag("CACHE_MODE")
        test_mode = _env_flag("TEST_MODE")
        if cache_mode and test_mode:
            raise RemoteCacheModeError("CACHE_MODE and TEST_MODE are mutually exclusive")
        mode = "test" if test_mode else "cache" if cache_mode else "off"
        root_dir = os.environ.get("SIMFOUNDRY_MODEL_CACHE_DIR", DEFAULT_CACHE_DIR)
        return cls(root_dir=root_dir, mode=mode)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def cache_enabled(self) -> bool:
        return self.mode == "cache"

    @property
    def test_enabled(self) -> bool:
        return self.mode == "test"

    def key_for(self, provider: str, model: str, request: Dict[str, Any]) -> str:
        return stable_hash({
            "schema_version": CACHE_SCHEMA_VERSION,
            "provider": provider,
            "model": model,
            "request": request,
        })

    def path_for(self, provider: str, key: str) -> Path:
        return self.root_dir / provider / f"{key}.json"

    def load_response(self, provider: str, key: str) -> Dict[str, Any]:
        path = self.path_for(provider=provider, key=key)
        if not path.exists():
            raise RemoteCacheMissError(
                f"Missing cached {provider} response for key {key}. "
                f"Expected cache entry at {path}."
            )
        with path.open("r", encoding="utf-8") as f:
            entry = json.load(f)
        return entry["response"]

    def load_response_if_exists(self, provider: str, key: str) -> Optional[Dict[str, Any]]:
        path = self.path_for(provider=provider, key=key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            entry = json.load(f)
        return entry["response"]

    def store_response(self, provider: str, model: str, key: str, request: Dict[str, Any], response: Dict[str, Any]) -> None:
        path = self.path_for(provider=provider, key=key)
        entry = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "provider": provider,
            "model": model,
            "key": key,
            "request": request,
            "response": response,
        }
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_core = {
                "schema_version": existing.get("schema_version"),
                "provider": existing.get("provider"),
                "model": existing.get("model"),
                "key": existing.get("key"),
                "request": existing.get("request"),
                "response": existing.get("response"),
            }
            if existing_core != entry:
                raise RemoteCacheConflictError(
                    f"Conflicting cached response for {provider} key {key} at {path}"
                )
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
