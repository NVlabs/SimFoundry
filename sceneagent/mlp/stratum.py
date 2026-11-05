"""Name buckets for the stratified GSO, YCB, and ABO sample.

Guesses are only a sampling aid. The stored material is the VLM label.
A near-duplicate is another object with the same product tokens, so a red
mug and a blue mug do not both enter one material bucket. Each bucket keeps
a small extent and a large extent when the catalog has both.
"""

from __future__ import annotations

import re
from collections import Counter

from sceneagent.taxonomy import MATERIAL_CLASSES

_STOP = frozenset({
    "a", "an", "the", "and", "or", "of", "for", "with", "to", "from", "in",
    "on", "by", "set", "pack", "pair", "piece", "pieces", "inch", "inches",
    "cm", "mm", "oz", "lb", "lbs", "count", "ct", "new", "size",
})

_COLOR_SIZE = frozenset({
    "red", "blue", "green", "black", "white", "yellow", "pink", "grey", "gray",
    "brown", "purple", "orange", "beige", "navy", "gold", "silver", "clear",
    "small", "medium", "large", "mini", "xl", "xs", "sm", "lg", "big", "little",
})

# Longer phrases are tried first. This list is a sampler, not a labeler.
_RULES = (
    ("tomato soup", "metal"),
    ("soup can", "metal"),
    ("tuna fish", "metal"),
    ("meat can", "metal"),
    ("tool box", "metal"),
    ("toolbox", "metal"),
    ("frying pan", "metal"),
    ("sauce pan", "metal"),
    ("saucepan", "metal"),
    ("can opener", "metal"),
    ("bottle cap", "plastic"),
    ("water bottle", "plastic"),
    ("juice box", "paper and cardboard"),
    ("cracker box", "paper and cardboard"),
    ("cereal box", "paper and cardboard"),
    ("pudding box", "paper and cardboard"),
    ("gelatin box", "paper and cardboard"),
    ("sugar box", "paper and cardboard"),
    ("milk carton", "paper and cardboard"),
    ("cutting board", "wood"),
    ("wine glass", "glass and ceramic"),
    ("action figure", "plastic"),
    ("stuffed animal", "fabric"),
    ("tennis ball", "rubber and foam"),
    ("golf ball", "rubber and foam"),
    ("baseball", "rubber and foam"),
    ("racquetball", "rubber and foam"),
    ("softball", "rubber and foam"),
    ("yoga mat", "rubber and foam"),
    ("floor mat", "rubber and foam"),
    ("throw pillow", "fabric"),
    ("whipstitch", "fabric"),
    ("cardboard", "paper and cardboard"),
    ("envelope", "paper and cardboard"),
    ("magazine", "paper and cardboard"),
    ("carton", "paper and cardboard"),
    ("paper", "paper and cardboard"),
    ("bookend", "wood"),
    ("bookshelf", "wood"),
    ("wooden", "wood"),
    ("bamboo", "wood"),
    ("walnut", "wood"),
    ("cabinet", "wood"),
    ("dresser", "wood"),
    ("bookshelf", "wood"),
    ("wood", "wood"),
    ("pillow", "fabric"),
    ("blanket", "fabric"),
    ("towel", "fabric"),
    ("hoodie", "fabric"),
    ("sweater", "fabric"),
    ("jacket", "fabric"),
    ("shirt", "fabric"),
    ("sock", "fabric"),
    ("glove", "fabric"),
    ("curtain", "fabric"),
    ("napkin", "fabric"),
    ("fabric", "fabric"),
    ("cloth", "fabric"),
    ("apparel", "fabric"),
    ("plush", "fabric"),
    ("rug", "fabric"),
    ("mattress", "fabric"),
    ("skillet", "metal"),
    ("hammer", "metal"),
    ("wrench", "metal"),
    ("screwdriver", "metal"),
    ("scissors", "metal"),
    ("padlock", "metal"),
    ("chain", "metal"),
    ("knife", "metal"),
    ("fork", "metal"),
    ("spoon", "metal"),
    ("clamp", "metal"),
    ("steel", "metal"),
    ("aluminum", "metal"),
    ("metal", "metal"),
    ("pan", "metal"),
    ("pot", "metal"),
    ("can", "metal"),
    ("nail", "metal"),
    ("screw", "metal"),
    ("key", "metal"),
    ("pitcher", "plastic"),
    ("cleanser", "plastic"),
    ("mustard", "plastic"),
    ("bleach", "plastic"),
    ("detergent", "plastic"),
    ("shampoo", "plastic"),
    ("lotion", "plastic"),
    ("marker", "plastic"),
    ("lego", "plastic"),
    ("duplo", "plastic"),
    ("controller", "plastic"),
    ("keyboard", "plastic"),
    ("headphone", "plastic"),
    ("remote", "plastic"),
    ("bucket", "plastic"),
    ("bottle", "plastic"),
    ("jug", "plastic"),
    ("helmet", "plastic"),
    ("plastic", "plastic"),
    ("toy", "plastic"),
    ("bin", "plastic"),
    ("basket", "plastic"),
    ("figurine", "plastic"),
    ("sponge", "rubber and foam"),
    ("eraser", "rubber and foam"),
    ("rubber", "rubber and foam"),
    ("foam", "rubber and foam"),
    ("silicone", "rubber and foam"),
    ("tire", "rubber and foam"),
    ("ceramic", "glass and ceramic"),
    ("porcelain", "glass and ceramic"),
    ("teacup", "glass and ceramic"),
    ("glass", "glass and ceramic"),
    ("vase", "glass and ceramic"),
    ("mirror", "glass and ceramic"),
    ("mug", "glass and ceramic"),
    ("plate", "glass and ceramic"),
    ("bowl", "glass and ceramic"),
    ("banana", "food and organic matter"),
    ("apple", "food and organic matter"),
    ("orange", "food and organic matter"),
    ("lemon", "food and organic matter"),
    ("peach", "food and organic matter"),
    ("pear", "food and organic matter"),
    ("strawberry", "food and organic matter"),
    ("fruit", "food and organic matter"),
    ("bread", "food and organic matter"),
    ("bagel", "food and organic matter"),
    ("cheese", "food and organic matter"),
    ("cereal", "food and organic matter"),
    ("food", "food and organic matter"),
    ("dish towel", "fabric"),
    ("dishtowel", "fabric"),
    ("bath sheet", "fabric"),
    ("washcloth", "fabric"),
    ("quilt", "fabric"),
    ("pressure cooker", "metal"),
    ("cooker", "metal"),
    ("kettle", "metal"),
    ("chocolate", "food and organic matter"),
    ("candy", "food and organic matter"),
    ("cookie", "food and organic matter"),
    ("snack", "food and organic matter"),
    ("cocoa", "food and organic matter"),
    ("rice", "food and organic matter"),
    ("olive oil", "liquid and granular matter"),
    ("vinegar", "liquid and granular matter"),
    ("syrup", "liquid and granular matter"),
    ("honey", "liquid and granular matter"),
    ("sauce", "liquid and granular matter"),
    ("liquid", "liquid and granular matter"),
    ("granular", "liquid and granular matter"),
    ("box", "paper and cardboard"),
    ("book", "paper and cardboard"),
    ("oak", "wood"),
)

_ORDERED_RULES = tuple(sorted(_RULES, key=lambda item: len(item[0]), reverse=True))


def guess_material(name):
    """Return a taxonomy material guessed from an object name.

    Unrecognized names return ``unknown``. The guess is not a training label.
    """

    text = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    padded = f" {text} "
    for phrase, material in _ORDERED_RULES:
        if material not in MATERIAL_CLASSES:
            raise ValueError(f"sampler rule {phrase!r} is not a material class")
        if f" {phrase} " in padded:
            return material
    return "unknown"


def name_tokens(name):
    raw = re.findall(r"[a-z0-9]+", (name or "").lower().replace("_", " "))
    return [token for token in raw if token not in _STOP and not token.isdigit()]


def core_tokens(name):
    return [token for token in name_tokens(name) if token not in _COLOR_SIZE]


def is_near_duplicate(name, others):
    """True when ``name`` is the same product as one of ``others``.

    Color and size words are ignored. A single shared token, such as ``can``,
    does not make two objects duplicates.
    """

    core = set(core_tokens(name))
    if not core:
        return False
    for other in others:
        other_core = set(core_tokens(other))
        if not other_core:
            continue
        if core == other_core:
            return True
        overlap = len(core & other_core)
        union = len(core | other_core)
        if overlap >= 2 and union and overlap / union >= 0.67:
            return True
    return False


def order_bucket(items, limit, occupied_names=()):
    """Order one material bucket, small and large first, without near-duplicates.

    ``items`` need ``name`` and may have ``extent`` in meters. Names in
    ``occupied_names`` count as already kept. The result length is at most
    ``limit``.
    """

    if limit < 1:
        return []
    known = [row for row in items if _extent_value(row) is not None]
    unknown = [row for row in items if _extent_value(row) is None]
    ordered = _alternate_extents(known) + list(unknown)
    picked = []
    taken = list(occupied_names)
    for row in ordered:
        if is_near_duplicate(row.get("name") or "", taken):
            continue
        picked.append(row)
        taken.append(row.get("name") or "")
        if len(picked) >= limit:
            break
    return picked


def labels_should_stop(records):
    """Return a reason to stop before training, or ``None`` to continue.

    Unknown above a quarter of the ok labels stops the run. A source whose
    labels collapse onto one class, or whose names and labels disagree as a
    group, stops the run as well.
    """

    ok = [row for row in records if row.get("status") == "ok" and row.get("material")]
    if len(ok) < 12:
        return None
    unknown = sum(row["material"] == "unknown" for row in ok) / len(ok)
    if unknown > 0.25:
        return f"unknown is {unknown:.0%} of labels"
    grouped = {}
    for row in ok:
        grouped.setdefault(row.get("source") or "", []).append(row)
    for source, rows in grouped.items():
        if len(rows) < 8:
            continue
        counts = Counter(row["material"] for row in rows)
        material, count = counts.most_common(1)[0]
        share = count / len(rows)
        if material == "unknown" and share > 0.5:
            return f"{source} labels are {share:.0%} unknown"
        if share >= 0.85:
            return f"{source} labels are {share:.0%} {material}"
        reason = _name_contradiction(source, rows)
        if reason:
            return reason
    return None


def _name_contradiction(source, rows):
    checks = (
        ("metal", {"fabric", "food and organic matter", "liquid and granular matter", "paper and cardboard"}),
        ("paper and cardboard", {"metal", "glass and ceramic", "liquid and granular matter"}),
        ("fabric", {"metal", "glass and ceramic", "food and organic matter", "liquid and granular matter"}),
    )
    for guessed, bad in checks:
        subset = [row for row in rows if guess_material(row.get("name") or "") == guessed]
        if len(subset) < 6:
            continue
        wrong = sum(row["material"] in bad for row in subset)
        if wrong / len(subset) >= 0.75:
            return (
                f"{source} {guessed} names are mostly labeled as other materials "
                f"({wrong} of {len(subset)})"
            )
    return None


def _extent_value(row):
    value = row.get("extent")
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(max(value))
        except (TypeError, ValueError):
            return None
    if number != number or number <= 0.0:
        return None
    return number


def _alternate_extents(items):
    ordered = sorted(items, key=lambda row: _extent_value(row))
    lo = 0
    hi = len(ordered) - 1
    take_small = True
    result = []
    while lo <= hi:
        result.append(ordered[lo] if take_small else ordered[hi])
        if take_small:
            lo += 1
        else:
            hi -= 1
        take_small = not take_small
    return result
