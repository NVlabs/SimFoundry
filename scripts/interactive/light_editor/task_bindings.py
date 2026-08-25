# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Which scene objects a task config binds, and whether each binding still holds.

A task config under ``scripts/cfg/task/`` binds scene objects through
``og_task_config.semantic_group_mapping``, a map from a group name to a list
of keys. A key matches an object by instance name or by category, so every
group is matched both ways. Nothing checks the result at load time: a group
that resolves to no objects or to several breaks the run later, or silently
changes the task. :data:`EFFECTS` names each failure and its severity.

Pairing a config with a scene is heuristic; each association reports how it
was made. In descending order of reliability: ``run_config`` (a top-level run
config in ``scripts/cfg/*.yaml`` pairs the two outright), ``task_dir`` (the
config sits in ``scripts/cfg/task/<scene name>/``), ``task_name`` (the task
name is, or starts with, the scene name), ``content`` (every group resolves
in this scene; instance-name matches are ``possible``, category-only matches
are ``weak`` and off by default). Association is always computed from the
unedited scene, so deleting an object cannot erase the evidence needed to
warn about that deletion.

Everything here is offline: YAML plus a scene JSON document. No OmniGibson,
no Isaac Sim, no GPU.

Usage:
    python task_bindings.py [--scene <scene_state.json>] [--remove NAME] [--json]
"""

import json
from pathlib import Path

import yaml

from scene_io import SceneEditError

#: Task configs, one per task, in a tree of subdirectories.
TASK_CFG_SUBDIR = "scripts/cfg/task"

#: The one subdirectory under it that does not hold authored configs: Review &
#: Export writes config snapshots into `<group>/exports/`. Mirrors
#: `export_bundle.SNAPSHOT_SUBDIR`; a test keeps the two in sync.
SNAPSHOT_SUBDIR = "exports"

#: Top-level run configs, which pair a task config with a scene.
RUN_CFG_SUBDIR = "scripts/cfg"

#: Where scene directories live; each is named for its scene.
SCENES_SUBDIR = "assets/scenes"

#: Category OmniGibson assigns every robot; task configs bind it with
#: ``robot: [agent]``. Robots serialize no category in the scene JSON.
ROBOT_CATEGORY = "agent"

#: OmniGibson's default when init_info names no category.
DEFAULT_CATEGORY = "object"

#: Predicate sections, and the aggregation applied to the subject group.
PREDICATE_SECTIONS = (
    ("init_predicates_all", "init", "all"),
    ("init_predicates_any", "init", "any"),
    ("init_predicates_specific", "init", "specific"),
    ("goal_predicates_all", "goal", "all"),
    ("goal_predicates_any", "goal", "any"),
    ("goal_predicates_specific", "goal", "specific"),
)

#: Sections keyed by group name rather than listing predicates.
GROUP_KEYED_SECTIONS = (
    "group_xyz_randomization",
    "group_z_rot_randomization",
    "group_init_relative_poses",
    "group_init_joint_positions",
    "group_predicate_placement",
)

#: What each failure does to a run, worst first. The order is the ranking used
#: to summarize a group that is referenced in several places.
EFFECTS = (
    ("assertion_error", "breaks",
     "an assert fires during the first reset, minutes into the run"),
    ("vacuous_success", "breaks",
     "all() over no objects is True, so every episode terminates successful on step 1"),
    ("unreachable_goal", "breaks",
     "any() over no objects is False, so the goal can never be reached"),
    ("placement_overlap", "changes_task",
     "both objects are placed at the same predicate-derived pose"),
    ("applies_to_all", "changes_task",
     "the predicate now requires every matched object to satisfy it"),
    ("applies_to_any", "changes_task",
     "the predicate now accepts any matched object satisfying it"),
    ("milestone_skipped", "changes_task",
     "the milestone is logged as unresolvable and never fires"),
    ("init_skipped", "changes_task",
     "the initial-state setup for this group silently does nothing"),
    ("placement_skipped", "changes_task",
     "predicate placement for this group silently does nothing"),
    ("unused", "no_effect",
     "only randomization reads this group, so nothing observable changes"),
)

_EFFECT_RANK = {name: rank for rank, (name, _, _) in enumerate(EFFECTS)}
_EFFECT_SEVERITY = {name: severity for name, severity, _ in EFFECTS}
_EFFECT_DETAIL = {name: detail for name, _, detail in EFFECTS}

#: Association methods, most reliable first.
CONFIDENCE_ORDER = ("certain", "likely", "possible", "weak")


def _rank_confidence(confidence):
    """Sort key for a confidence label; unknown labels sort last."""
    try:
        return CONFIDENCE_ORDER.index(confidence)
    except ValueError:
        return len(CONFIDENCE_ORDER)


# --- the scene side --------------------------------------------------------

def scene_object_categories(scene):
    """Map every object in a scene document to the category OmniGibson gives it.

    Matching must reproduce the runtime ``obj.category``, so this does not
    reuse ``scene_io``'s display category: a robot serializes no category at
    all, and OmniGibson assigns it ``agent``.

    Args:
        scene (dict): Parsed scene JSON.

    Returns:
        dict: ``{object name: category}``, in scene-document order.
    """
    categories = {}
    for name, info in (scene.get("objects_info", {}).get("init_info", {}) or {}).items():
        if str(info.get("class_module", "")).startswith("omnigibson.robots."):
            categories[name] = ROBOT_CATEGORY
            continue
        category = (info.get("args") or {}).get("category")
        categories[name] = str(category) if category else DEFAULT_CATEGORY
    return categories


def apply_edit(categories, removed=(), added=None):
    """Return the object index a proposed edit would leave behind.

    Args:
        categories (dict): ``{name: category}`` from
            :func:`scene_object_categories`.
        removed (iterable[str]): Objects the edit deletes. Names not in the
            scene are ignored (deleting a not-yet-saved import is legal).
        added (dict or None): ``{name: category}``, or ``{name: spec}`` using
            the specs ``asset_library.object_spec`` builds.

    Returns:
        dict: A new ``{name: category}``; the input is not modified.

    Raises:
        SceneEditError: If a name is not a string, an added object collides
            with one already in the scene, or an added value carries no
            category.
    """
    result = dict(categories)
    for name in removed:
        if not isinstance(name, str):
            raise SceneEditError(f"removed object names must be strings, got {name!r}")
        result.pop(name, None)

    for name, value in (added or {}).items():
        if not isinstance(name, str):
            raise SceneEditError(f"added object names must be strings, got {name!r}")
        if name in result:
            raise SceneEditError(f"cannot add {name}: the scene already has an object by that name")
        if isinstance(value, str):
            category = value
        elif isinstance(value, dict):
            category = value.get("category") or (
                (value.get("init_info") or {}).get("args") or {}).get("category")
        else:
            category = None
        if not category:
            raise SceneEditError(f"added object {name!r} has no category")
        result[name] = str(category)
    return result


def known_scene_names(repo_root):
    """Scene names available in this checkout, from ``assets/scenes/*/``.

    Used only to break ties in the ``task_name`` heuristic when one scene
    name is a prefix of another.

    Args:
        repo_root (str or Path): Repository root.

    Returns:
        set[str]: Directory names under :data:`SCENES_SUBDIR`.
    """
    scenes_dir = Path(repo_root) / SCENES_SUBDIR
    if not scenes_dir.is_dir():
        return set()
    return {child.name for child in scenes_dir.iterdir() if child.is_dir()}


def scene_name_for(scene_json_path):
    """Scene name for a scene-state JSON: the stem before ``_scene_state_``."""
    return Path(scene_json_path).stem.split("_scene_state_")[0]


# --- the task side ---------------------------------------------------------

def _predicate_entries(section):
    """Yield ``(index, entry)`` for a predicate section that may be null."""
    if not isinstance(section, list):
        return
    for index, entry in enumerate(section):
        if isinstance(entry, dict):
            yield index, entry


def _use(where, role, *, empty, many, state=None):
    """One place a group is referenced, and what each cardinality does there."""
    return {"where": where, "role": role, "state": state,
            "effect_when_empty": empty, "effect_when_many": many}


def _collect_uses(og_cfg):
    """Find every reference to a group name, and what breaks at each one.

    ``where`` is the path inside the config, so a warning can point at the
    YAML; the effects say what the wrong number of objects does at that use.

    Args:
        og_cfg (dict): ``og_task_config``.

    Returns:
        dict: ``{group name: [use, ...]}`` for every group referenced,
        including groups with no ``semantic_group_mapping`` entry.
    """
    uses = {}

    def add(group, use):
        if isinstance(group, str) and group:
            uses.setdefault(group, []).append(use)

    for key, phase, kind in PREDICATE_SECTIONS:
        for index, entry in _predicate_entries(og_cfg.get(key)):
            state = entry.get("state")
            where = f"{key}[{index}]"
            if phase == "init":
                # Init predicates only set state: an empty subject group skips
                # the setup, while a missing `other` is still a hard assert.
                add(entry.get("group"), _use(where, "subject", state=state,
                                             empty="init_skipped", many="applies_to_all"))
            elif kind == "all":
                add(entry.get("group"), _use(where, "subject", state=state,
                                             empty="vacuous_success", many="applies_to_all"))
            elif kind == "any":
                add(entry.get("group"), _use(where, "subject", state=state,
                                             empty="unreachable_goal", many="applies_to_any"))
            else:
                # SPECIFIC asserts exactly one object at step time.
                add(entry.get("group"), _use(where, "subject", state=state,
                                             empty="assertion_error", many="assertion_error"))
            add(entry.get("other_group"), _use(where, "other", state=state,
                                               empty="assertion_error", many="assertion_error"))

    # Milestones degrade instead of asserting: a bad group logs a warning and
    # drops the milestone.
    for index, entry in _predicate_entries(og_cfg.get("milestone_predicates")):
        where = f"milestone_predicates[{index}]"
        name = entry.get("name")
        if name:
            where += f" ({name})"
        add(entry.get("group"), _use(where, "subject", state=entry.get("state"),
                                     empty="milestone_skipped", many="milestone_skipped"))
        add(entry.get("other_group"), _use(where, "other", state=entry.get("state"),
                                           empty="milestone_skipped", many="milestone_skipped"))

    for key in GROUP_KEYED_SECTIONS:
        section = og_cfg.get(key)
        if not isinstance(section, dict):
            continue
        for group, value in section.items():
            if key == "group_init_relative_poses":
                # PickPlaceTask asserts exactly one object per pose group.
                add(group, _use(key, "pose", empty="assertion_error", many="assertion_error"))
            elif key == "group_init_joint_positions":
                add(group, _use(key, "joints", empty="init_skipped", many="applies_to_all"))
            elif key == "group_predicate_placement":
                add(group, _use(key, "placed",
                                empty="placement_skipped", many="placement_overlap"))
                references = None
                if isinstance(value, dict):
                    references = value.get("reference_groups", value.get("reference_group"))
                for reference in ([references] if isinstance(references, str) else (references or [])):
                    # The placement reference must match exactly one object.
                    add(reference, _use(f"{key}.{group}.reference_group", "reference",
                                        empty="assertion_error", many="assertion_error"))
            else:
                add(group, _use(key, "randomization", empty="unused", many="unused"))

    extra = og_cfg.get("additional_objects")
    if isinstance(extra, dict):
        # A missing or oversized group only costs the distractors their
        # placement centre.
        add(extra.get("reference_group"),
            _use("additional_objects.reference_group", "reference",
                 empty="unused", many="unused"))

    return uses


def read_task_config(path):
    """Parse one task config, or decline it.

    Args:
        path (str or Path): A YAML file under :data:`TASK_CFG_SUBDIR`.

    Returns:
        dict or None: ``path``, ``name`` (file stem), ``task_name``,
        ``task_dir`` (the containing directory name), ``instruction``,
        ``groups`` (``{group: [key, ...]}``) and ``uses`` (from
        :func:`_collect_uses`); or None, with no error, when the file is not a
        checkable task config: unparseable YAML, no ``og_task_config``, or a
        ``semantic_group_mapping`` that binds nothing.
    """
    path = Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        # Unparseable files contribute no bindings; the pipeline reports them.
        return None
    if not isinstance(document, dict):
        return None
    og_cfg = document.get("og_task_config")
    if not isinstance(og_cfg, dict):
        return None

    mapping = og_cfg.get("semantic_group_mapping")
    groups = {}
    if isinstance(mapping, dict):
        for group, keys in mapping.items():
            if isinstance(keys, str):
                keys = [keys]
            if not isinstance(keys, list):
                continue
            groups[str(group)] = [str(key) for key in keys if isinstance(key, (str, int))]
    if not any(groups.values()):
        return None

    return {
        "path": path,
        "name": path.stem,
        "task_name": str(document.get("task_name") or path.stem),
        "task_dir": path.parent.name,
        "instruction": document.get("language_instruction"),
        "groups": groups,
        "uses": _collect_uses(og_cfg),
    }


def run_config_scene_pairs(repo_root):
    """Read the task-to-scene pairings the top-level run configs state outright.

    A run config selects its task in Hydra's ``defaults`` list and names the
    scene JSON the eval and teleop stages load. The two stages can name
    different scenes, so both are kept.

    Args:
        repo_root (str or Path): Repository root.

    Returns:
        dict: ``{task config stem: {scene name: [source, ...]}}``, where each
        source is ``"<run config>.<field>"``.
    """
    pairs = {}
    cfg_dir = Path(repo_root) / RUN_CFG_SUBDIR
    for path in sorted(cfg_dir.glob("*.yaml")):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(document, dict):
            continue

        tasks = []
        for entry in document.get("defaults") or []:
            if isinstance(entry, dict) and entry.get("task"):
                tasks.append(str(entry["task"]))
        if not tasks:
            continue

        scenes = {}
        for section, field in (("s15_eval", "scene_json"),
                               ("s14_teleop", "scene_json_name")):
            value = (document.get(section) or {}).get(field) if isinstance(
                document.get(section), dict) else None
            # The field may hold a path or a bare scene name.
            if isinstance(value, str) and value and "${" not in value:
                scenes.setdefault(scene_name_for(Path(value).name), []).append(
                    f"{path.name}.{section}.{field}")

        for task in tasks:
            for scene, sources in scenes.items():
                pairs.setdefault(task, {}).setdefault(scene, []).extend(sources)
    return pairs


def discover_task_configs(repo_root, *, pairs=None):
    """Parse every task config in the repo and attach its scene hints.

    Args:
        repo_root (str or Path): Repository root.
        pairs (dict or None): Pre-read :func:`run_config_scene_pairs` output,
            so a caller checking many scenes reads the run configs once.

    Returns:
        list[dict]: One entry per checkable config, sorted by path, each as
        :func:`read_task_config` returns plus ``scene_hints``: a list of
        ``{"scene", "how", "evidence"}`` for the scenes the repo names outright.
        Files that are not checkable task configs are omitted silently; see
        :func:`read_task_config`.
    """
    repo_root = Path(repo_root)
    pairs = run_config_scene_pairs(repo_root) if pairs is None else pairs
    configs = []
    root = repo_root / TASK_CFG_SUBDIR
    for path in sorted(root.rglob("*.yaml")):
        # Export snapshots are byte-copies of configs already in this list;
        # skip them so the task picker offers only authored configs.
        if SNAPSHOT_SUBDIR in path.relative_to(root).parts:
            continue
        config = read_task_config(path)
        if config is None:
            continue
        config["scene_hints"] = [
            {"scene": scene, "how": "run_config", "evidence": ", ".join(sorted(set(sources)))}
            for scene, sources in sorted(pairs.get(config["name"], {}).items())
        ]
        configs.append(config)
    return configs


# --- association -----------------------------------------------------------

def _name_association(config, scene_name, known_scenes):
    """Association from naming conventions alone, or None."""
    if config["task_dir"] == scene_name:
        return {"how": "task_dir", "confidence": "certain",
                "evidence": f"config lives in {TASK_CFG_SUBDIR}/{scene_name}/"}
    task_name = config["task_name"]
    if task_name == scene_name or config["name"] == scene_name:
        return {"how": "task_name", "confidence": "certain",
                "evidence": f"task name is the scene name ({scene_name})"}
    for candidate in (task_name, config["name"]):
        if not candidate.startswith(scene_name + "_"):
            continue
        # When one scene name prefixes another, only the most specific scene
        # owns the task.
        longer = [other for other in (known_scenes or ())
                  if len(other) > len(scene_name)
                  and (candidate == other or candidate.startswith(other + "_"))]
        if longer:
            continue
        return {"how": "task_name", "confidence": "likely",
                "evidence": f"task name {candidate} starts with the scene name"}
    for candidate in (task_name, config["name"]):
        if scene_name.startswith(candidate + "_"):
            # The other direction: scene variants are named by suffixing the
            # scene they came from, and remain legitimate targets for the task.
            return {"how": "scene_variant", "confidence": "possible",
                    "evidence": f"scene is a variant of {candidate}"}
    return None


def associate(config, scene_name, categories=None, *, known_scenes=None):
    """Decide whether a task config is meant for this scene, and how sure we are.

    Args:
        config (dict): One entry from :func:`discover_task_configs`.
        scene_name (str): Scene being edited, as :func:`scene_name_for` returns.
        categories (dict or None): The **unedited** scene index, for the
            content heuristic; an edited index can erase the evidence for the
            association that would have warned about the edit.
        known_scenes (set[str] or None): Scene names in the checkout, used to
            stop a task being claimed by a shorter scene name that is also a
            prefix of it.

    Returns:
        dict or None: ``{"how", "confidence", "evidence"}``, or None if nothing
        connects this config to this scene.
    """
    for hint in config.get("scene_hints", ()):
        if hint["scene"] == scene_name and hint["how"] == "run_config":
            return {"how": "run_config", "confidence": "certain",
                    "evidence": hint["evidence"]}

    by_name = _name_association(config, scene_name, known_scenes)
    if by_name is not None:
        return by_name

    if not categories:
        return None
    names = set(categories)
    values = set(categories.values())
    by_instance_name = []
    for keys in config["groups"].values():
        if not any(key in names or key in values for key in keys):
            return None
        by_instance_name.extend(key for key in keys if key in names)
    if by_instance_name:
        return {"how": "content", "confidence": "possible",
                "evidence": "binds objects present here by instance name: "
                            + ", ".join(sorted(set(by_instance_name))[:3])}
    return {"how": "content", "confidence": "weak",
            "evidence": "every group matches something here, but only by category"}


# --- binding state ---------------------------------------------------------

def _match(keys, categories):
    """Objects a group's keys select, and which identifier selected each one."""
    matched = []
    for name, category in categories.items():
        if name in keys:
            matched.append({"object": name, "by": "name"})
        elif category in keys:
            matched.append({"object": name, "by": "category"})
    return matched


def _group_state(group, keys, uses, categories):
    """Resolve one group and rank what its cardinality does to the run."""
    matched = _match(keys, categories)
    count = len(matched)
    if count == 1:
        effect = None
    else:
        field = "effect_when_empty" if count == 0 else "effect_when_many"
        effects = [use[field] for use in uses] or ["unused"]
        effect = min(effects, key=lambda name: _EFFECT_RANK.get(name, len(EFFECTS)))
    return {
        "group": group,
        "keys": list(keys),
        "objects": [entry["object"] for entry in matched],
        "matched_by": {entry["object"]: entry["by"] for entry in matched},
        "count": count,
        "uses": [use["where"] for use in uses],
        "effect": effect,
    }


def groups_for_scene(config, categories):
    """Resolve one config's groups against one scene, association set aside.

    Unlike :func:`bindings_for_scene`, this does not require the config to be
    associated with the scene: given that the caller is looking at this
    config, it reports what each of its groups matches here.

    Args:
        config (dict): One entry from :func:`discover_task_configs`, or the
            result of :func:`read_task_config`.
        categories (dict): ``{object name: category}`` for the scene as it
            stands, from :func:`scene_object_categories` and :func:`apply_edit`.

    Returns:
        dict: ``{group: state}`` for every group the config names, each state
        the same shape :func:`bindings_for_scene` reports per group.
    """
    return {
        group: _group_state(group, keys, config["uses"].get(group, []), categories)
        for group, keys in config["groups"].items()
    }


def severity_of(effect):
    """How badly an effect breaks a run: one of the labels in :data:`EFFECTS`.

    Unknown effects rank as `changes_task`: an unrecognised name is not
    evidence of safety.
    """
    return _EFFECT_SEVERITY.get(effect, "changes_task")


def bindings_for_scene(scene, configs, *, scene_name, known_scenes=None,
                       min_confidence="possible", categories=None):
    """Resolve every applicable task config's bindings against one scene.

    Args:
        scene (dict): Parsed scene JSON.
        configs (list[dict]): From :func:`discover_task_configs`.
        scene_name (str): Scene being edited.
        known_scenes (set[str] or None): See :func:`associate`.
        min_confidence (str): Weakest association to include, one of
            :data:`CONFIDENCE_ORDER`. The default drops ``weak``
            (category-only content matches).
        categories (dict or None): Override the scene index, for callers that
            have already applied an edit. Association still uses ``scene``.

    Returns:
        list[dict]: One entry per applicable config: ``task``, ``path``,
        ``task_name``, ``instruction``, ``association`` and ``groups`` (one
        :func:`_group_state` per bound group, plus one per group that
        predicates reference but ``semantic_group_mapping`` never defines,
        marked ``undefined``).

    Raises:
        SceneEditError: If ``min_confidence`` is not a known confidence.
    """
    if min_confidence not in CONFIDENCE_ORDER:
        raise SceneEditError(
            f"min_confidence must be one of {', '.join(CONFIDENCE_ORDER)}, got {min_confidence!r}")

    authored = scene_object_categories(scene)
    current = authored if categories is None else categories
    limit = _rank_confidence(min_confidence)

    reports = []
    for config in configs:
        association = associate(config, scene_name, authored, known_scenes=known_scenes)
        if association is None or _rank_confidence(association["confidence"]) > limit:
            continue

        groups = [
            _group_state(group, keys, config["uses"].get(group, []), current)
            for group, keys in config["groups"].items()
        ]
        for group, uses in config["uses"].items():
            if group in config["groups"]:
                continue
            # A predicate naming an undefined group raises KeyError while the
            # scene loads, before a single step.
            groups.append({
                "group": group, "keys": [], "objects": [], "matched_by": {},
                "count": 0, "uses": [use["where"] for use in uses],
                "effect": "assertion_error", "undefined": True,
            })

        reports.append({
            "task": config["name"],
            "path": str(config["path"]),
            "task_name": config["task_name"],
            "instruction": config.get("instruction"),
            "association": association,
            "groups": groups,
        })
    return reports


def _message(report, group):
    """One line stating the defect, in the terms the task config uses."""
    task = report["task"]
    name = group["group"]
    keys = ", ".join(group["keys"])
    if group.get("undefined"):
        return (f"{task}: predicates reference group '{name}', which "
                f"semantic_group_mapping never defines")
    if group["count"] == 0:
        return f"{task}: group '{name}' ({keys}) matches no object in this scene"
    return (f"{task}: group '{name}' ({keys}) matches {group['count']} objects: "
            + ", ".join(group["objects"]))


def check_bindings(scene, configs, removed=(), added=None, *, scene_name,
                   known_scenes=None, min_confidence="possible",
                   include_no_effect=False):
    """Warn about task bindings a proposed edit would leave unsatisfied.

    Warnings only: nothing here is a reason to refuse a save. The association
    may be wrong, and a task config is a text file anyone can fix afterwards.

    Args:
        scene (dict): Parsed scene JSON, before the edit.
        configs (list[dict]): From :func:`discover_task_configs`.
        removed (iterable[str]): Objects the edit deletes.
        added (dict or None): ``{name: category}`` or ``{name: spec}``; see
            :func:`apply_edit`.
        scene_name (str): Scene being edited.
        known_scenes (set[str] or None): See :func:`associate`.
        min_confidence (str): See :func:`bindings_for_scene`.
        include_no_effect (bool): Also report groups whose cardinality nothing
            observable depends on.

    Returns:
        list[dict]: Warnings, worst first, each carrying ``task``, ``path``,
        ``group``, ``keys``, ``objects``, ``count``, ``count_before``,
        ``kind`` (``empty_group`` / ``ambiguous_group`` / ``undefined_group``),
        ``effect``, ``severity``, ``detail``, ``uses``, ``association``,
        ``caused_by_edit`` and ``message``.

    Raises:
        SceneEditError: If the edit is malformed or ``min_confidence`` is not a
            known confidence.
    """
    authored = scene_object_categories(scene)
    edited = apply_edit(authored, removed, added)

    # Keyed by path, not by task stem: configs in different directories can
    # share a stem, and a collision would hand one config the other's
    # pre-edit count.
    before = {
        (report["path"], group["group"]): group["count"]
        for report in bindings_for_scene(
            scene, configs, scene_name=scene_name, known_scenes=known_scenes,
            min_confidence=min_confidence)
        for group in report["groups"]
    }

    warnings = []
    for report in bindings_for_scene(
            scene, configs, scene_name=scene_name, known_scenes=known_scenes,
            min_confidence=min_confidence, categories=edited):
        for group in report["groups"]:
            effect = group["effect"]
            if effect is None:
                continue
            severity = _EFFECT_SEVERITY.get(effect, "changes_task")
            if severity == "no_effect" and not include_no_effect:
                continue
            count_before = before.get((report["path"], group["group"]), group["count"])
            warnings.append({
                "task": report["task"],
                "path": report["path"],
                "task_name": report["task_name"],
                "group": group["group"],
                "keys": group["keys"],
                "objects": group["objects"],
                "count": group["count"],
                "count_before": count_before,
                "kind": ("undefined_group" if group.get("undefined")
                         else "empty_group" if group["count"] == 0 else "ambiguous_group"),
                "effect": effect,
                "severity": severity,
                "detail": _EFFECT_DETAIL.get(effect, ""),
                "uses": group["uses"],
                "association": report["association"],
                # Whether this edit broke the binding or merely inherited a
                # pre-existing mismatch.
                "caused_by_edit": count_before != group["count"],
                "message": _message(report, group),
            })

    warnings.sort(key=lambda w: (
        _EFFECT_RANK.get(w["effect"], len(EFFECTS)),
        not w["caused_by_edit"],
        _rank_confidence(w["association"]["confidence"]),
        w["task"], w["group"],
    ))
    return warnings


def main():
    """Report every scene's task bindings from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--scene", default=None,
                        help="One scene-state JSON. Default: every scene's _latest.")
    parser.add_argument("--remove", action="append", default=[],
                        help="Simulate deleting an object. Repeatable.")
    parser.add_argument("--add", action="append", default=[],
                        help="Simulate adding <name>=<category>. Repeatable.")
    parser.add_argument("--min-confidence", default="possible", choices=CONFIDENCE_ORDER)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    configs = discover_task_configs(repo_root)
    scenes = set(known_scene_names(repo_root))

    if args.scene:
        paths = [Path(args.scene)]
    else:
        paths = sorted(
            path for name in scenes
            for path in [repo_root / SCENES_SUBDIR / name / f"{name}_scene_state_latest.json"]
            if path.exists()
        )

    added = {}
    for spec in args.add:
        name, _, category = spec.partition("=")
        added[name] = category or name.rsplit("_", 1)[0]

    payload = []
    for path in paths:
        scene = json.loads(path.read_text(encoding="utf-8"))
        name = scene_name_for(path)
        warnings = check_bindings(
            scene, configs, removed=args.remove, added=added, scene_name=name,
            known_scenes=scenes, min_confidence=args.min_confidence)
        applicable = bindings_for_scene(
            scene, configs, scene_name=name, known_scenes=scenes,
            min_confidence=args.min_confidence)
        payload.append({"scene": name, "tasks": len(applicable), "warnings": warnings})
        if args.json:
            continue
        print(f"{name}: {len(applicable)} task config(s), {len(warnings)} warning(s)")
        for warning in warnings:
            print(f"  [{warning['severity']}/{warning['effect']}] {warning['message']}")
            print(f"      {warning['detail']}; used at {', '.join(warning['uses'])}")
    if args.json:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
