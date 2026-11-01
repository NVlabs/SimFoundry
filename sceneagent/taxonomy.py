"""Published SceneAgent material, body, and collision vocabularies.

Section 3.4 of arXiv:2610.34745 names ten material classes, six body types,
and eight collision primitives. The keyword map below is an implementation of
the paper's "keyword priors derived from the object name." The paper does not
publish the dictionary, so these pairs are ours and are labeled as such.
"""

MATERIAL_CLASSES = (
    "plastic",
    "metal",
    "wood",
    "fabric",
    "paper and cardboard",
    "rubber and foam",
    "glass and ceramic",
    "food and organic matter",
    "liquid and granular matter",
    "unknown",
)

BODY_TYPES = (
    "dynamic rigid",
    "static rigid",
    "kinematic",
    "soft deformable",
    "cloth shell",
    "fluid or granular",
)

COLLISION_PRIMITIVES = (
    "box",
    "sphere",
    "cylinder",
    "capsule",
    "convex hull",
    "triangle mesh",
    "tetrahedral mesh",
    "cloth mesh",
)

# Longer phrases first so "dry erase marker" hits plastic via "dry erase"
# before any shorter token. This table is not in the paper.
_KEYWORD_RULES = (
    ("dry erase", "plastic"),
    ("marker", "plastic"),
    ("cardboard", "paper and cardboard"),
    ("paper", "paper and cardboard"),
    ("teacup", "glass and ceramic"),
    ("ceramic", "glass and ceramic"),
    ("glass", "glass and ceramic"),
    ("bowl", "glass and ceramic"),
    ("cup", "glass and ceramic"),
    ("hoodie", "fabric"),
    ("towel", "fabric"),
    ("hat", "fabric"),
    ("fabric", "fabric"),
    ("cloth", "fabric"),
    ("rubber", "rubber and foam"),
    ("foam", "rubber and foam"),
    ("milk", "liquid and granular matter"),
    ("water", "liquid and granular matter"),
    ("juice", "liquid and granular matter"),
    ("apple", "food and organic matter"),
    ("flower", "food and organic matter"),
    ("wood", "wood"),
    ("metal", "metal"),
    ("screw", "metal"),
    ("clamp", "metal"),
)

# Fixed weight used when a caller does not supply one. The paper says keyword
# priors use a fixed weight and does not publish the number.
DEFAULT_KEYWORD_WEIGHT = 1.0


def keyword_material(name, weight=DEFAULT_KEYWORD_WEIGHT):
    """Return a keyword-prior proposal for an object name.

    If no rule matches, the class is ``unknown``. The vote scales that class.
    """
    text = (name or "").lower()
    material = "unknown"
    matched = None
    for needle, cls in _KEYWORD_RULES:
        if needle in text:
            material = cls
            matched = needle
            break
    if material not in MATERIAL_CLASSES:
        raise ValueError(f"keyword rule produced {material!r}")
    return {
        "source": "keyword",
        "class": material,
        "weight": float(weight),
        "matched": matched,
        "dictionary": "sceneagent.taxonomy._KEYWORD_RULES",
    }


def require_material(name):
    if name not in MATERIAL_CLASSES:
        raise ValueError(f"unknown material class {name!r}")
    return name


def require_body(name):
    if name not in BODY_TYPES:
        raise ValueError(f"unknown body type {name!r}")
    return name


def require_collision(name):
    if name not in COLLISION_PRIMITIVES:
        raise ValueError(f"unknown collision primitive {name!r}")
    return name
