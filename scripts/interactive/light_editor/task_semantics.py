# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Whether a task config could run at all, asked without starting Isaac Sim.

``task_bindings`` checks whether a config's groups match objects in a given
scene; this module checks the config by itself: the states it names exist,
every predicate references a group the mapping defines, a goal is stated, an
instruction is present. Cardinality ("does `other_group` match exactly one
object") is a scene question and stays in ``task_bindings``; the server runs
both and merges the results.

Generated configs are model output, so a made-up state, an undefined group or
a missing goal must be caught here rather than minutes into a run.

Usage:
    python task_semantics.py <config.yaml> [...] [--all]
"""

from pathlib import Path

import yaml

import task_bindings

#: States that take a second object -- every OmniGibson ``RelativeObjectState``
#: subclass. ``PickPlaceTask`` reads ``other_group`` for these, so a predicate
#: that omits it raises KeyError. OmniGibson cannot be imported in this env,
#: so the list is checked in; re-derive it from
#: deps/BEHAVIOR-1K/OmniGibson/omnigibson/object_states when OmniGibson is
#: upgraded.
OG_BINARY_STATES = frozenset({
    "AttachedTo", "ContactParticles", "ContainedParticles", "Contains",
    "Covered", "Draped", "Filled", "Inside", "IsGrasping", "ModifiedParticles",
    "NextTo", "OnTop", "Overlaid", "Saturated", "Touching", "Under",
})

#: The ``AbsoluteObjectState`` half: no ``other_group``, and naming one is
#: simply ignored.
OG_UNARY_STATES = frozenset({
    "AABB", "Burnt", "ContactBodies", "Cooked", "Folded", "FoldedLevel",
    "Frozen", "HeatSourceOrSink", "Heated", "HorizontalAdjacency", "Joint",
    "MaxTemperature", "ObjectsInFOVOfRobot", "OnFire", "Open", "Pose",
    "ParticleApplier", "ParticleModifier", "ParticleRemover", "ParticleSink",
    "ParticleSource", "SlicerActive", "Temperature", "TensorizedValueState",
    "ToggledOn", "Unfolded", "VerticalAdjacency",
})

#: States SimFoundry handles itself (``SPECIAL_STATE_NAMES`` in
#: ``simfoundry/tasks/predicates.py``). All but ``Lifted`` read
#: ``other_group``; ``IsGrasping`` is OmniGibson's own and is already binary
#: above.
SPECIAL_BINARY_STATES = frozenset({
    "PlaceOnTop", "InsideAABB", "OnTopAABB", "AboveAABB",
})
SPECIAL_UNARY_STATES = frozenset({"Lifted"})

BINARY_STATES = OG_BINARY_STATES | SPECIAL_BINARY_STATES
UNARY_STATES = OG_UNARY_STATES | SPECIAL_UNARY_STATES
KNOWN_STATES = BINARY_STATES | UNARY_STATES

#: States whose ``value:`` must be a boolean: the check compares with ``==``,
#: and a string "true" never equals True, so the predicate is silently never
#: satisfied. Numeric states such as ``Temperature`` are excluded. Same
#: provenance and test as the arity sets above.
BOOLEAN_STATES = frozenset({
    "AttachedTo", "Burnt", "Contains", "Cooked", "Covered", "Draped", "Filled",
    "Folded", "Frozen", "Heated", "Inside", "IsGrasping", "NextTo", "OnTop",
    "Open", "Overlaid", "Saturated", "SlicerActive", "ToggledOn", "Touching",
    "Under", "Unfolded",
    # The SimFoundry states, which all compare a bool. PlaceOnTop ignores the
    # value but still reads the key.
    "PlaceOnTop", "InsideAABB", "OnTopAABB", "AboveAABB", "Lifted",
})

#: The goal sections. A config with nothing in any of them is not a task:
#: ``all([])`` is True, so every episode ends successful on step one.
GOAL_SECTIONS = ("goal_predicates_all", "goal_predicates_any",
                 "goal_predicates_specific")
INIT_SECTIONS = ("init_predicates_all", "init_predicates_any",
                 "init_predicates_specific")

#: The task ``type`` these rules describe. Other types are reported as
#: unjudged rather than silently passed.
VALIDATED_TASK_TYPE = "PickPlaceTask"

#: Task types this repo can actually run: ``PickPlaceTask`` is the only class
#: ``simfoundry/tasks/`` registers. Any other name fails a registry check
#: minutes into a run.
SUPPORTED_TASK_TYPES = frozenset({VALIDATED_TASK_TYPE})

#: Keys ``PickPlaceTask`` cannot be constructed or evaluated without.
#: ``activity_name`` is a positional parameter, so a config without one raises
#: TypeError while the environment is built; ``termination_config`` is indexed
#: directly by the eval stage, so a config without one raises KeyError before
#: Isaac Sim is even asked for anything.
REQUIRED_PICK_PLACE_KEYS = ("activity_name", "termination_config")

#: Milestone entry keys ``_create_milestone_predicates`` indexes directly;
#: each missing one is a KeyError during load.
REQUIRED_MILESTONE_KEYS = ("name", "group", "state", "value")

#: Effects this module reports that ``task_bindings.EFFECTS`` has no name for,
#: because they are not about cardinality. Same ``(severity, detail)`` shape,
#: looked up after that table so the shared names keep their shared meaning.
_EXTRA_EFFECTS = {
    "load_error": ("breaks",
                   "the task raises while the scene is being loaded, before step one"),
    # Not `breaks`: the run config can supply the prompt. What is lost is the
    # export, which cannot pin a blank instruction.
    "unpinned_prompt": ("changes_task",
                        "the exported command cannot pin s15_eval.prompt, so the run "
                        "config's own prompt -- another task's sentence -- stands"),
    "never_satisfied": ("breaks",
                        "the comparison can never hold, so the goal is unreachable"),
    # Not a defect and not a pass: reported rather than returned as silence,
    # because an empty problem list is read as "this config is runnable".
    "unjudged": ("no_effect",
                 "these checks describe PickPlaceTask, so nothing here has been "
                 "verified for this task type -- check it by hand"),
}


def _effect(name):
    """``(severity, detail)`` for an effect name, from either table."""
    severity = task_bindings._EFFECT_SEVERITY.get(name)
    if severity is not None:
        return severity, task_bindings._EFFECT_DETAIL.get(name, "")
    return _EXTRA_EFFECTS.get(name, ("changes_task", ""))


#: Worst first, slotted into ``task_bindings``' own ranking: ``load_error``
#: sorts ahead of everything, ``never_satisfied`` alongside
#: ``assertion_error``, ``unpinned_prompt`` among the ``changes_task`` effects.
_EXTRA_RANK = {"load_error": -1, "never_satisfied": 0, "unpinned_prompt": 4.5,
               # Last: context for everything else, not a finding.
               "unjudged": 99}


def _rank(effect):
    if effect in _EXTRA_RANK:
        return _EXTRA_RANK[effect]
    return task_bindings._EFFECT_RANK.get(effect, len(task_bindings.EFFECTS))


def _problem(kind, effect, where, message, **extra):
    """One reason a config cannot run, in the shape every warning here takes.

    ``kind`` is what ``server.wire_warning`` renames to ``code``; ``where`` is
    the path inside the YAML, e.g. ``goal_predicates_all[0].state``.
    """
    severity, detail = _effect(effect)
    return {"kind": kind, "effect": effect, "severity": severity,
            "where": where, "message": message, "detail": detail, **extra}


def blocking(problems):
    """The problems that mean "this config cannot run", not "check this".

    Args:
        problems (list[dict]): From :func:`validate_task`.

    Returns:
        list[dict]: Those with severity ``breaks``.
    """
    return [p for p in problems if p.get("severity") == "breaks"]


def summarize(problems):
    """One line naming the worst problem and how many others there are."""
    if not problems:
        return ""
    lead = problems[0]["message"]
    return lead + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else "")


def validate_task_yaml(yaml_text):
    """:func:`validate_task` for text that has not been parsed yet.

    Args:
        yaml_text (str): The config as it would be written.

    Returns:
        list[dict]: Problems; text that is not YAML, or not a mapping, is
        itself a problem rather than an exception.
    """
    try:
        document = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return [_problem("unparseable", "load_error", "",
                         f"the config is not valid YAML: {e}")]
    return validate_task(document)


def validate_task(document):
    """Every reason *document* could not run as a task config.

    Scene-independent: this says whether the config is internally coherent;
    ``task_bindings.check_bindings`` says whether it fits the open scene.
    Neither implies the other.

    Args:
        document (dict): A parsed task config: ``task_name``,
            ``language_instruction`` and ``og_task_config``.

    Returns:
        list[dict]: Problems, worst first. Each carries ``kind``, ``effect``,
        ``severity`` (``breaks`` / ``changes_task`` / ``no_effect``), ``where``
        (the path inside the YAML), ``message`` and ``detail``. Empty means the
        config is runnable as far as anything can tell without a scene.
    """
    if not isinstance(document, dict):
        return [_problem("unparseable", "load_error", "",
                         "the config is not a YAML mapping")]

    og_cfg = document.get("og_task_config")
    if not isinstance(og_cfg, dict):
        return [_problem("no_og_task_config", "load_error", "og_task_config",
                         "the config has no og_task_config section, so there is "
                         "no task to run")]

    task_type = og_cfg.get("type")
    if task_type is None:
        # task_config["type"] is a plain index with no default; the
        # environment raises KeyError while loading.
        return [_problem(
            "no_task_type", "load_error", "og_task_config.type",
            "og_task_config names no type, and the environment reads "
            "task_config['type'] directly -- the run raises KeyError while "
            "loading, before step one")]
    task_type = str(task_type).strip()
    if task_type not in SUPPORTED_TASK_TYPES:
        # Unjudged, not valid: these rules only describe PickPlaceTask.
        return [_problem(
            "unsupported_task_type", "unjudged", "og_task_config.type",
            f"type is {task_type!r}; these checks describe "
            f"{', '.join(sorted(SUPPORTED_TASK_TYPES))} and cannot say whether "
            f"this config is runnable. Nothing about it has been verified.",
            task_type=task_type)]

    problems = []
    problems += _check_required_keys(og_cfg)
    problems += _check_instruction(document)
    mapping, mapping_problems = _check_mapping(og_cfg)
    problems += mapping_problems
    if mapping:
        # Only when there is something to state a goal about: a config that
        # binds nothing has already been reported for that.
        problems += _check_goal_present(og_cfg)
    problems += _check_predicates(og_cfg, mapping)
    problems += _check_milestones(og_cfg, mapping)
    problems += _check_group_references(og_cfg, mapping)

    problems.sort(key=lambda p: (_rank(p["effect"]), p["where"], p["kind"]))
    return problems


def _check_required_keys(og_cfg):
    """The fields whose absence is a crash rather than a defaulted value."""
    problems = []
    activity = og_cfg.get("activity_name")
    if "activity_name" not in og_cfg:
        problems.append(_problem(
            "no_activity_name", "load_error", "og_task_config.activity_name",
            "activity_name is required: PickPlaceTask takes it positionally, so "
            "the environment raises TypeError while constructing the task"))
    elif not isinstance(activity, str) or not activity.strip():
        problems.append(_problem(
            "blank_activity_name", "load_error", "og_task_config.activity_name",
            f"activity_name is {activity!r}; it names the activity the task "
            f"reports and logs under, and blank is not a name"))

    termination = og_cfg.get("termination_config")
    if "termination_config" not in og_cfg:
        problems.append(_problem(
            "no_termination_config", "load_error",
            "og_task_config.termination_config",
            "termination_config is required: the eval stage writes "
            "task_cfg['termination_config']['max_steps'] directly, so a config "
            "without one raises KeyError before the scene is loaded"))
    elif not isinstance(termination, dict):
        problems.append(_problem(
            "bad_termination_config", "load_error",
            "og_task_config.termination_config",
            f"termination_config is {type(termination).__name__}, not a mapping; "
            f"the eval stage assigns max_steps into it"))
    return problems


def _check_milestones(og_cfg, mapping):
    """Milestone predicates.

    They follow the same rules as goal predicates, plus one of their own:
    ``requires`` names other milestones, and a name nothing defines means the
    gated milestone is never checked at all.
    """
    section = og_cfg.get("milestone_predicates")
    if section is None:
        return []
    where = "og_task_config.milestone_predicates"
    if not isinstance(section, list):
        return [_problem(
            "bad_milestone_section", "load_error", where,
            f"milestone_predicates is {type(section).__name__}, not a list")]

    problems = []
    names = []
    for index, entry in enumerate(section):
        at = f"{where}[{index}]"
        if not isinstance(entry, dict):
            problems.append(_problem(
                "bad_milestone", "load_error", at,
                f"{at} is {type(entry).__name__}, not a milestone mapping"))
            continue
        missing = [key for key in REQUIRED_MILESTONE_KEYS if key not in entry]
        if missing:
            problems.append(_problem(
                "milestone_missing_key", "load_error", at,
                f"{at} has no {', '.join(missing)}; _create_milestone_predicates "
                f"indexes {'them' if len(missing) > 1 else 'it'} directly and "
                f"raises KeyError"))
        name = entry.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name)
        # Same state/arity/value rules as goal predicates, reused so the two
        # cannot drift.
        problems += _check_predicate(entry, mapping, at)

    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        problems.append(_problem(
            "duplicate_milestone", "milestone_skipped", where,
            f"more than one milestone is called {duplicates[0]!r}; they share one "
            f"entry in milestones_achieved, so all but the last are unreportable",
            group=duplicates[0]))

    defined = set(names)
    for index, entry in enumerate(section):
        if not isinstance(entry, dict):
            continue
        requires = entry.get("requires") or []
        if isinstance(requires, str):
            requires = [requires]
        if not isinstance(requires, list):
            problems.append(_problem(
                "bad_milestone_requires", "load_error",
                f"{where}[{index}].requires",
                f"requires is {type(requires).__name__}; it has to be a list of "
                f"milestone names, or one name"))
            continue
        for unknown in [r for r in requires if r not in defined]:
            problems.append(_problem(
                "undefined_milestone_requirement", "milestone_skipped",
                f"{where}[{index}].requires",
                f"{entry.get('name')!r} requires milestone {unknown!r}, which no "
                f"milestone defines; prerequisites_satisfied reads "
                f"milestones_achieved.get({unknown!r}, 0), so this milestone is "
                f"never checked at all", group=unknown))
    return problems


def _check_instruction(document):
    """The sentence the policy is given. Blank is not a neutral default."""
    instruction = document.get("language_instruction")
    if isinstance(instruction, str) and instruction.strip():
        return []
    return [_problem(
        "no_instruction", "unpinned_prompt", "language_instruction",
        "language_instruction is empty, so an export of this task cannot pin "
        "the sentence the policy is given")]


def _check_mapping(og_cfg):
    """The group table, and what a predicate is allowed to reference.

    Returns:
        tuple: ``(mapping, problems)`` where ``mapping`` is the usable
        ``{group: [key, ...]}`` -- groups whose value is unusable are left out,
        so a predicate naming one is reported once as an unusable group rather
        than twice.
    """
    raw = og_cfg.get("semantic_group_mapping")
    where = "og_task_config.semantic_group_mapping"
    if not isinstance(raw, dict) or not raw:
        return {}, [_problem(
            "no_group_mapping", "load_error", where,
            "semantic_group_mapping is empty, so no predicate can name a group "
            "and no object is ever bound")]

    mapping = {}
    problems = []
    for group, keys in raw.items():
        at = f"{where}.{group}"
        if not isinstance(group, str) or not group.strip():
            problems.append(_problem(
                "bad_group_name", "load_error", at,
                f"group name {group!r} is not a name a predicate can reference"))
            continue
        if isinstance(keys, str):
            # set(...) over a bare string yields its characters, so a bare
            # string binds nothing.
            problems.append(_problem(
                "group_keys_not_a_list", "load_error", at,
                f"group '{group}' maps to the string {keys!r}; it has to be a "
                f"list -- [{keys}] -- or set(...) reads it one character at a time"))
            continue
        if not isinstance(keys, list) or not keys:
            problems.append(_problem(
                "group_binds_nothing", "vacuous_success", at,
                f"group '{group}' lists no object name or category, so it binds "
                f"nothing in any scene"))
            continue
        bad = [k for k in keys if not isinstance(k, str) or not k.strip()]
        if bad:
            problems.append(_problem(
                "bad_group_key", "vacuous_success", at,
                f"group '{group}' lists {bad[0]!r}, which is not an object name "
                f"or category and cannot match anything"))
            continue
        mapping[group] = [str(k) for k in keys]
    return mapping, problems


def _check_goal_present(og_cfg):
    """Report a config that states no goal predicate."""
    for key in GOAL_SECTIONS:
        section = og_cfg.get(key)
        if isinstance(section, list) and any(isinstance(e, dict) for e in section):
            return []
    return [_problem(
        "no_goal", "vacuous_success", "og_task_config.goal_predicates_all",
        "the config states no goal predicate, so MultiPredicate holds an empty "
        "list -- all([]) is True and every episode ends successful on step 1")]


def _check_predicates(og_cfg, mapping):
    """Every predicate entry, against what PickPlaceTask does with it."""
    problems = []
    for key in INIT_SECTIONS + GOAL_SECTIONS:
        section = og_cfg.get(key)
        if section is None:
            continue
        if not isinstance(section, list):
            problems.append(_problem(
                "bad_predicate_section", "load_error", f"og_task_config.{key}",
                f"{key} is {type(section).__name__}, not a list of predicates"))
            continue
        for index, entry in enumerate(section):
            at = f"og_task_config.{key}[{index}]"
            if not isinstance(entry, dict):
                problems.append(_problem(
                    "bad_predicate", "load_error", at,
                    f"{key}[{index}] is {type(entry).__name__}, not a predicate mapping"))
                continue
            problems += _check_predicate(entry, mapping, at)
    return problems


def _check_predicate(entry, mapping, at):
    """One predicate: its group, its state, its value, and its other group."""
    problems = []

    group = entry.get("group")
    if not isinstance(group, str) or not group.strip():
        problems.append(_problem(
            "missing_group", "load_error", f"{at}.group",
            f"{at} names no group; PickPlaceTask reads predicate_info['group'] "
            f"and raises KeyError without one"))
        group = None
    elif group not in mapping:
        problems.append(_problem(
            "undefined_group", "assertion_error", f"{at}.group",
            f"{at} references group '{group}', which semantic_group_mapping "
            f"never defines", group=group))

    state = entry.get("state")
    if not isinstance(state, str) or not state.strip():
        problems.append(_problem(
            "missing_state", "load_error", f"{at}.state",
            f"{at} names no state"))
        return problems

    state = state.strip()
    if state not in KNOWN_STATES:
        problems.append(_problem(
            "unknown_state", "load_error", f"{at}.state",
            f"{at} names state '{state}', which is neither a SimFoundry state "
            f"({', '.join(sorted(SPECIAL_BINARY_STATES | SPECIAL_UNARY_STATES))}) "
            f"nor one OmniGibson registers -- _resolve_registered_state raises "
            f"ValueError for it", state=state))
        # Arity and value are judged only for a state that exists.
        return problems

    if "value" not in entry:
        problems.append(_problem(
            "missing_value", "load_error", f"{at}.value",
            f"{at} has no value; PickPlaceTask reads predicate_info['value'] "
            f"and raises KeyError without one"))
    elif state in BOOLEAN_STATES and not isinstance(entry.get("value"), bool):
        value = entry.get("value")
        problems.append(_problem(
            "non_boolean_value", "never_satisfied", f"{at}.value",
            f"{at} sets value: {value!r} for boolean state '{state}'; the check "
            f"is `get_value() == value`, and {value!r} never equals True or False",
            state=state))

    kwargs = entry.get("state_kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        problems.append(_problem(
            "bad_state_kwargs", "load_error", f"{at}.state_kwargs",
            f"{at} sets state_kwargs to {type(kwargs).__name__}; it is splatted "
            f"as **state_kwargs and has to be a mapping or null"))

    other = entry.get("other_group")
    has_other = isinstance(other, str) and other.strip()
    if state in BINARY_STATES:
        if not has_other:
            problems.append(_problem(
                "missing_other_group", "load_error", f"{at}.other_group",
                f"{at} uses binary state '{state}' with no other_group; "
                f"PickPlaceTask reads predicate_info['other_group'] for it and "
                f"raises KeyError without one", state=state))
        elif other not in mapping:
            problems.append(_problem(
                "undefined_group", "assertion_error", f"{at}.other_group",
                f"{at} references group '{other}', which semantic_group_mapping "
                f"never defines", group=other))
        elif group is not None and other == group:
            problems.append(_problem(
                "self_referential", "unreachable_goal", f"{at}.other_group",
                f"{at} asks whether '{group}' is {state} itself; the target is "
                f"drawn from the same group as the subject, so the predicate "
                f"can never hold", group=group, state=state))
    elif has_other:
        problems.append(_problem(
            "unused_other_group", "unused", f"{at}.other_group",
            f"{at} names other_group '{other}' for unary state '{state}', which "
            f"ignores it", group=other, state=state))

    return problems


def _check_group_references(og_cfg, mapping):
    """Sections outside the predicate lists that name a group, e.g. randomization.

    Walked with ``task_bindings._collect_uses`` so the list of reference sites
    has a single source. Predicate references are already reported per
    predicate above.
    """
    problems = []
    predicate_keys = tuple(f"{key}[" for key in INIT_SECTIONS + GOAL_SECTIONS)
    for group, uses in task_bindings._collect_uses(og_cfg).items():
        if group in mapping:
            continue
        elsewhere = [use for use in uses
                     if not str(use["where"]).startswith(predicate_keys)]
        if not elsewhere:
            continue
        worst = min((use["effect_when_empty"] for use in elsewhere), key=_rank)
        where = elsewhere[0]["where"]
        problems.append(_problem(
            "undefined_group", worst, f"og_task_config.{where}",
            f"{where} names group '{group}', which semantic_group_mapping never "
            f"defines", group=group))
    return problems


def main():
    """Report on task configs named on the command line."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("configs", nargs="+", help="Task config YAML files")
    parser.add_argument("--all", action="store_true",
                        help="Also print configs with nothing to report")
    args = parser.parse_args()

    worst = 0
    for name in args.configs:
        path = Path(name)
        try:
            problems = validate_task_yaml(path.read_text(encoding="utf-8"))
        except OSError as e:
            print(f"{path}: could not read ({e.strerror})")
            worst = max(worst, 1)
            continue
        if not problems and not args.all:
            continue
        print(path)
        for problem in problems:
            print(f"  {problem['severity']:<12} {problem['message']}")
        if blocking(problems):
            worst = max(worst, 1)
    return worst


if __name__ == "__main__":
    import sys

    sys.exit(main())
