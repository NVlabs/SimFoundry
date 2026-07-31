# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for querying BEHAVIOR-1K (and other) asset datasets.

Provides filter parsing, model scanning, and category-spec loading used by
both the standalone ``query_assets.py`` CLI and the runtime distractor
object system in ``PickPlaceTask``.
"""

import json
import operator
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Filter parsing
# ---------------------------------------------------------------------------
_OPS: Dict[str, Callable] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

_FILTER_RE = re.compile(
    r"^\s*(\w+)\s*(<=|>=|!=|==|<|>)\s*([^\s]+)\s*$"
)

VALID_FIELDS = {"mass", "volume", "density", "bbox_volume", "bbox_x", "bbox_y", "bbox_z"}


def parse_filter(expr: str) -> Tuple[str, Callable, float]:
    """Parse ``"field op value"`` into ``(field, op_func, float_value)``."""
    m = _FILTER_RE.match(expr)
    if m is None:
        raise ValueError(
            f"Invalid filter expression: '{expr}'. "
            f"Expected format: 'field op value' (e.g. 'volume < 0.001')"
        )
    field, op_str, val_str = m.group(1), m.group(2), m.group(3)
    if field not in VALID_FIELDS:
        raise ValueError(
            f"Unknown filter field '{field}'. Must be one of: {sorted(VALID_FIELDS)}"
        )
    return field, _OPS[op_str], float(val_str)


def apply_filters(
    record: Dict[str, Any],
    parsed_filters: List[Tuple[str, Callable, float]],
) -> bool:
    """Return True if *record* satisfies **all** filters."""
    for field, op_func, threshold in parsed_filters:
        value = record.get(field)
        if value is None:
            return False
        if not op_func(value, threshold):
            return False
    return True


# ---------------------------------------------------------------------------
# Asset directory detection
# ---------------------------------------------------------------------------

def detect_assets_dir(dataset_name: str = "behavior-1k-assets") -> str:
    """
    Auto-detect the dataset root relative to the project root.

    Tries ``<project_root>/deps/BEHAVIOR-1K/datasets/<dataset_name>``.

    Args:
        dataset_name: Name of the dataset folder.

    Returns:
        Absolute path to the dataset root.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    # Walk up from this file: utils/ -> simfoundry/ -> project_root
    project_root = Path(__file__).resolve().parent.parent.parent
    candidate = project_root / "deps" / "BEHAVIOR-1K" / "datasets" / dataset_name
    if candidate.is_dir():
        return str(candidate)
    raise FileNotFoundError(
        f"Could not auto-detect assets dir for dataset '{dataset_name}'. "
        f"Tried: {candidate}"
    )


# ---------------------------------------------------------------------------
# Category specs
# ---------------------------------------------------------------------------

def load_category_specs(assets_dir: str) -> Dict[str, Dict[str, float]]:
    """Load ``avg_category_specs.json`` -> {category: {mass, volume, density}}."""
    path = os.path.join(assets_dir, "metadata", "avg_category_specs.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model scanning
# ---------------------------------------------------------------------------

def _extract_abilities(link_tags: Dict[str, List[str]]) -> set:
    """Collect the unique set of ability tags across all links."""
    abilities: set = set()
    for tags in link_tags.values():
        abilities.update(tags)
    return abilities


def scan_models(
    assets_dir: str,
    category_specs: Dict[str, Dict[str, float]],
    category_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Walk ``objects/<category>/<model>/`` and build a record per model.

    Each record contains:
        category, model, path,
        mass, volume, density        (from category specs),
        bbox_x, bbox_y, bbox_z, bbox_volume   (from model metadata),
        abilities                    (set of ability tags from link_tags).
    """
    objects_dir = os.path.join(assets_dir, "objects")
    if not os.path.isdir(objects_dir):
        return []
    records: List[Dict[str, Any]] = []

    for category in sorted(os.listdir(objects_dir)):
        cat_path = os.path.join(objects_dir, category)
        if not os.path.isdir(cat_path):
            continue

        # Optional category substring filter
        if category_filter and category_filter.lower() not in category.lower():
            continue

        # Category-level specs (may be absent for some categories)
        cat_spec = category_specs.get(category, {})

        for model in sorted(os.listdir(cat_path)):
            model_path = os.path.join(cat_path, model)
            if not os.path.isdir(model_path):
                continue

            metadata_path = os.path.join(model_path, "misc", "metadata.json")

            # Per-model bbox and abilities
            bbox_x = bbox_y = bbox_z = bbox_volume = None
            abilities: set = set()
            if os.path.isfile(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        meta = json.load(f)
                    bbox_size = meta.get("bbox_size")
                    if bbox_size and len(bbox_size) == 3:
                        bbox_x, bbox_y, bbox_z = bbox_size
                        bbox_volume = bbox_x * bbox_y * bbox_z
                    link_tags = meta.get("link_tags", {})
                    abilities = _extract_abilities(link_tags)
                except (json.JSONDecodeError, KeyError):
                    pass

            record = {
                "category": category,
                "model": model,
                "path": model_path,
                # Category-level
                "mass": cat_spec.get("mass"),
                "volume": cat_spec.get("volume"),
                "density": cat_spec.get("density"),
                # Per-model
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_z": bbox_z,
                "bbox_volume": bbox_volume,
                "abilities": abilities,
            }
            records.append(record)

    return records


def query_assets(
    assets_dir: str,
    filters: Optional[List[str]] = None,
    category: Optional[str] = None,
    abilities: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    High-level helper: scan *assets_dir* and return records matching all criteria.

    Args:
        assets_dir: Root of the dataset (e.g. ``behavior-1k-assets``).
        filters: List of filter expressions (e.g. ``["volume < 0.001"]``).
        category: Optional category substring filter.
        abilities: Optional list of required ability tags.

    Returns:
        List of matching model record dicts.
    """
    category_specs = load_category_specs(assets_dir)
    records = scan_models(assets_dir, category_specs, category_filter=category)

    if filters:
        parsed = [parse_filter(f) for f in filters]
        records = [r for r in records if apply_filters(r, parsed)]

    if abilities:
        required = set(abilities)
        records = [r for r in records if required.issubset(r["abilities"])]

    return records
