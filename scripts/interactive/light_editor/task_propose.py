# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Turn a plain-English prompt into a task config yaml for the light editor's
"Generate task" panel: Gemini gets a scene screenshot and object list and
returns goal predicates over OmniGibson-style states.

Needs only google-genai and pyyaml (no torch); google-genai is imported
lazily, so on a venv without it the rest of the editor still works and only
this panel is unavailable.
"""

import os
import re
import time
from pathlib import Path

import yaml

import task_semantics

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


# Allowed goal-predicate states offered to the model.
OG_OBJECT_STATES = "OnTop, OnTopAABB, InsideAABB, AboveAABB, Lifted"

ROBOT_CONSTRAINTS = {
    "franka": (
        "The robot gripper has a maximum opening length of 85 mm. When "
        "proposing a task that involves grasping, choose objects that can "
        "fit within this grasp (e.g. avoid very large or wide objects that "
        "would exceed 85 mm)."
    ),
}


def _robot_constraint(robot_type):
    """The prompt's constraint paragraph for *robot_type*, or "" when none exists.

    Args:
        robot_type (str): As the panel sends it.

    Returns:
        str: The paragraph with the blank lines around it, or "" for an arm
        with no entry in :data:`ROBOT_CONSTRAINTS`.
    """
    text = ROBOT_CONSTRAINTS.get(str(robot_type or "").strip().lower())
    return f"\nImportant: {text}\n" if text else ""


PROMPT_TEMPLATE = """You are a professional robotic expert in sim-to-real, robot learning, and VLAs.

Look at the provided image of a tabletop scene in a simulation (OmniGibson).
Using ONLY the objects listed below, produce exactly ONE task that a {robot_type}
arm can perform in this scene, matching this instruction from the user:

    {user_prompt}

If the user's instruction cannot be carried out with the objects listed below --
it names something that is not there, or asks for something a {robot_type} arm
in this scene cannot do -- then output exactly one line:

    cannot: <one sentence naming what is missing or impossible>

and nothing else. Do NOT substitute a different task, and do NOT pick the
nearest thing these objects allow. A task the user did not ask for is worse
than no task: it is saved, run, and reported on as though it were theirs.

Ensure the task's goal requires a change from the current configuration of
the scene -- for example, do not propose putting a pear in a bowl if it is
already in the bowl. The task should have clear goal conditions expressible
with object states.
{robot_constraint}
Scene object list (id -> category, name; use these names as group identifiers in predicates):
{object_list}

Output format: one YAML block that can be merged into a task config. Use this
exact structure:

1. task_name: short snake_case name (e.g. stack_cup_on_plate)
2. language_instruction: one sentence telling a person what to do, phrased as an
   order to the robot. Follow the house style of the existing configs exactly:
   "Pick up the <object> and put it on the <target>" / "... and put it in the
   <target>". Name the objects the way a person would say them ("the red marker",
   "the yellow bowl"), never by their scene id.
3. semantic_group_mapping: map each logical group name you use (e.g. cup, plate) to a list containing exactly one scene object "name" from the list above (e.g. cup: [blue_cup_syvtml_19], plate: [teal_plate_qdudop_16]).
4. goal_predicates_all: list of predicates that must ALL be true for success. Each predicate has: state, state_kwargs (null if not needed), value (true/false), group, other_group (only for binary states like OnTop, Touching).
5. goal_predicates_any: list of predicates where ANY being true yields success (optional; use null if not needed).

Allowed states (from OmniGibson / SimFoundry): {states}
For binary relations use group and other_group. For unary states use group and set other_group to null.

State usage guidance (emit state_kwargs exactly as shown, or null):
- OnTop: OmniGibson kinematic on-top check. Binary (group on top of other_group). state_kwargs: null.
- OnTopAABB: bounding-box on-top check, more reliable for scanned/custom meshes; prefer it over OnTop. Binary. Optional state_kwargs keys: z_tolerance (meters, default 0.03), xy_overlap_threshold (fraction 0-1 of the top object's footprint, default 0.5).
- InsideAABB: bounding-box containment; use for putting an object in a bowl, cup, box, or drawer. Binary. Recommended state_kwargs: volume_threshold (fraction 0-1 of the inner object's volume that must be inside; use 0.5).
- AboveAABB: object held entirely above the reference object's top surface (hovering, not resting). Binary. Optional state_kwargs keys: min_clearance (meters, default 0.0), xy_overlap_threshold (fraction 0-1, or null to skip the alignment check).
- Lifted: object raised at least min_height above its height at the start of the episode. Unary (set other_group to null). state_kwargs: min_height (meters, default 0.05).
Do not invent other states or state_kwargs keys.

Example predicate format:
  goal_predicates_all:
    - state: OnTopAABB
      state_kwargs:
        z_tolerance: 0.03
        xy_overlap_threshold: 0.5
      value: true
      group: cup
      other_group: plate
  goal_predicates_any: null

Output ONLY the YAML block -- no prose, no headings. It must be valid YAML with keys: task_name, language_instruction, semantic_group_mapping, goal_predicates_all, goal_predicates_any.
The single `cannot:` line above is the one permitted alternative to that block."""


class TaskProposeError(RuntimeError):
    """Anything about the request or the response that a UI should show verbatim."""


def _load_api_keys():
    """Load KEY=VALUE pairs from api_keys.txt into os.environ (no-op if already set)."""
    here = Path(__file__).resolve()
    candidate = None
    for parent in here.parents:
        maybe = parent / "api_keys.txt"
        if maybe.exists():
            candidate = maybe
            break
    if candidate is None:
        return
    # utf-8-sig strips a BOM that would otherwise become part of the first key's name.
    with open(candidate, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                # Strip shell-style quotes so they are not sent as part of the key.
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


_GEMINI_API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
_DEFAULT_MODEL = "gemini-2.5-pro"
# Gemini 2.5-pro spends output tokens on thinking before writing; a budget
# sized for the yaml alone truncates the reply into an empty candidate.
_MAX_OUTPUT_TOKENS = 65535

# Throttling is the one 4xx worth retrying; other client errors fail
# identically every attempt.
_RATE_LIMIT_MARKERS = (
    "429", "resource_exhausted", "resourceexhausted", "rate limit", "ratelimit",
    "quota exceeded", "too many requests",
)
_NON_RETRYABLE_MARKERS = (
    "400", "401", "403", "404", "invalid_argument", "permission_denied",
    "unauthenticated", "not_found", "api key not valid",
)


def _is_non_retryable(exc):
    """True for a client error that every retry would hit again."""
    text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return True
    return any(marker in text for marker in _NON_RETRYABLE_MARKERS)


def _client():
    """Build a genai Client: an API key (Developer API) if one is set,
    otherwise a Vertex project with Application Default Credentials.
    """
    if genai is None:
        raise TaskProposeError(
            "google-genai is not installed in this env -- re-run "
            "`bash scripts/installation/install_light_editor.sh` "
            "to enable task generation."
        )
    _load_api_keys()
    key = next((os.environ[k] for k in _GEMINI_API_KEY_ENVS if os.environ.get(k)), None)
    if key:
        return genai.Client(api_key=key)
    project = os.environ.get("GCLOUD_PROJECT")
    if project:
        return genai.Client(vertexai=True, project=project, location="global")
    raise TaskProposeError(
        "No Gemini credentials found. Set GEMINI_API_KEY / GOOGLE_API_KEY, or "
        "GCLOUD_PROJECT with `gcloud auth application-default login` -- see "
        "api_keys.txt."
    )


def _call_gemini(prompt_text, image_bytes, *, model=_DEFAULT_MODEL, n_retries=3):
    client = _client()
    parts = [genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
             genai_types.Part.from_text(text=prompt_text)]
    contents = [genai_types.Content(role="user", parts=parts)]
    config = genai_types.GenerateContentConfig(
        temperature=0.4, max_output_tokens=_MAX_OUTPUT_TOKENS)

    last_exc = None
    for attempt in range(n_retries):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            text = getattr(response, "text", None)
            if text:
                return text
            last_exc = TaskProposeError("Gemini returned an empty response.")
        except Exception as e:  # noqa: BLE001 - surfaced to the UI, not swallowed
            if _is_non_retryable(e):
                raise TaskProposeError(f"Gemini rejected the request: {e}") from None
            last_exc = e
        # Back off between attempts only; no sleep after the final one.
        if attempt + 1 < n_retries:
            time.sleep(min(2 ** attempt, 8))
    raise TaskProposeError(f"Gemini call failed after {n_retries} attempt(s): {last_exc}")


def _format_object_list(objects):
    lines = []
    for i, obj in enumerate(objects):
        name = obj.get("name") or f"object_{i}"
        category = obj.get("category") or "?"
        lines.append(f'  "{i}": {{"category": "{category}", "name": "{name}"}}')
    return "{\n" + ",\n".join(lines) + "\n}"


_FENCE_RE = re.compile(r"```(?:ya?ml)?\s*\n(.*?)```", re.DOTALL)


def _clean_yaml_block(text):
    """Strip a markdown code fence and any prose before the yaml payload."""
    block = text.strip()
    if block.startswith("```"):
        lines = block.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        block = "\n".join(lines).strip()
    match = re.search(r"^task_name:", block, re.MULTILINE)
    if match:
        block = block[match.start():]
    return block.strip()


def _yaml_candidates(text):
    """Yield the payloads worth handing to the yaml parser, best first.

    Gemini may wrap the block in a fence with prose around it, so fenced
    payloads are tried first; the whole reply is the fallback for a bare block.
    """
    for block in _FENCE_RE.findall(text):
        candidate = _clean_yaml_block(block)
        if candidate:
            yield candidate
    candidate = _clean_yaml_block(text)
    if candidate:
        yield candidate


# A refusal is a line of its own; it is only honoured when the reply carries
# no `task_name:`, so a proposal that mentions "cannot" is still a proposal.
_REFUSAL_RE = re.compile(r"^\s*cannot\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)


def _refusal(text):
    """The reason Gemini gave for declining, or None if it proposed something."""
    if re.search(r"^\s*task_name\s*:", text, re.MULTILINE):
        return None
    match = _REFUSAL_RE.search(text)
    if not match:
        return None
    reason = match.group(1).strip().strip("`\"'").strip()
    return reason or "the objects in this scene cannot do that"


def _parse_task(text):
    scan_error = None
    parsed_something = False
    for block in _yaml_candidates(text):
        try:
            parsed = yaml.safe_load(block)
        except yaml.YAMLError as e:
            scan_error = scan_error or e
            continue
        parsed_something = True
        if isinstance(parsed, dict) and parsed.get("task_name"):
            return parsed
    # Only blame the yaml when nothing in the reply parsed at all.
    if scan_error is not None and not parsed_something:
        raise TaskProposeError(
            f"Gemini's response wasn't valid YAML: {scan_error}"
        ) from None
    raise TaskProposeError(
        "Gemini's response didn't include a task_name. Raw response:\n" + text[:500]
    )


def _base_task_template():
    """Same shape as scripts/cfg/task/example.yaml, minus the comments."""
    return {
        "task_name": "proposed_task",
        # Dict order is file order under safe_dump(sort_keys=False).
        "language_instruction": "",
        "og_task_config": {
            "type": "PickPlaceTask",
            "activity_name": "proposed_task",
            "semantic_group_mapping": {},
            "init_predicates_all": None,
            "init_predicates_any": None,
            "init_predicates_specific": None,
            "goal_predicates_all": None,
            "goal_predicates_any": None,
            "goal_predicates_specific": None,
            "robot_pose": None,
            "robot_xyz_randomization": None,
            "robot_z_rot_randomization": None,
            "robot_joint_randomization": None,
            "group_xyz_randomization": None,
            "group_z_rot_randomization": None,
            "termination_config": {"max_steps": 500},
            "reward_config": {"r_potential": 1.0},
            "include_obs": False,
        },
    }


# Default per-group randomization: 5 cm of jitter in the table plane, none in
# Z, and +-10 degrees of yaw (metres / radians). Adjust per group in the range
# editor.
_DEFAULT_GROUP_XYZ = [0.05, 0.05, 0.0]
_DEFAULT_GROUP_Z_ROT = 0.174  # pi / 18

# The arm's group randomizes through robot_xyz_randomization, not as a prop.
_ROBOT_GROUP_NAMES = ("robot", "agent")


def _is_robot_group(name, members):
    """True for the group that binds the arm rather than a prop."""
    names = [name] + list(members or [])
    return any(str(n).strip().lower() in _ROBOT_GROUP_NAMES for n in names)


# Preposition per binary state, for the fallback instruction below.
_STATE_PREPOSITION = {
    "insideaabb": "in",
    "inside": "in",
    "ontop": "on",
    "ontopaabb": "on",
    "placeontop": "on",
    "aboveaabb": "above",
    "touching": "against",
}


def _spoken(group):
    """A group name as a person would say it: `blue_cup` -> `the blue cup`."""
    words = str(group).replace("_", " ").strip()
    return f"the {words}" if words else "it"


def _derive_instruction(mapping, *goal_sections):
    """Build a house-style instruction from the goal, for when the model omits one.

    Policy evaluation reads ``language_instruction`` as the prompt, so a
    sentence derived from the first usable goal predicate beats a blank one.

    Args:
        mapping (dict): ``semantic_group_mapping``.
        *goal_sections: The goal predicate lists, best first.

    Returns:
        str: The instruction, or "" when there is nothing to build one from.
    """
    first = None
    for section in goal_sections:
        if not isinstance(section, list):
            continue
        first = next((p for p in section
                      if isinstance(p, dict) and p.get("group")), None)
        if first:
            break
    if not first:
        groups = [g for g in (mapping or {}) if not _is_robot_group(g, (mapping or {}).get(g))]
        return f"Pick up {_spoken(groups[0])}" if groups else ""

    group = first.get("group")
    other = first.get("other_group")
    state = str(first.get("state") or "").strip().lower()
    if not other:
        # Unary state: no target to name.
        return f"Pick up {_spoken(group)}"
    preposition = _STATE_PREPOSITION.get(state, "on")
    return f"Pick up {_spoken(group)} and put it {preposition} {_spoken(other)}"


def _normalized_mapping(mapping):
    """``semantic_group_mapping`` with one-object groups written as lists.

    Gemini sometimes returns ``cup: blue_cup_19`` instead of the requested
    ``cup: [blue_cup_19]``; ``PickPlaceTask`` turns a bare string into a set of
    its characters, so the group would match no object. Anything else passes
    through untouched for ``task_semantics`` to report.

    Args:
        mapping (dict): As proposed.

    Returns:
        dict: A new mapping; the input is not modified.
    """
    return {group: ([keys] if isinstance(keys, str) else keys)
            for group, keys in mapping.items()}


def _build_full_task(proposed, scene_name):
    raw_name = proposed.get("task_name", "proposed_task")
    task_name = f"{scene_name}_{raw_name}" if scene_name else raw_name
    full = _base_task_template()
    full["task_name"] = task_name
    full["og_task_config"]["activity_name"] = task_name
    mapping = proposed.get("semantic_group_mapping")
    mapping = _normalized_mapping(mapping if isinstance(mapping, dict) else {})
    full["og_task_config"]["semantic_group_mapping"] = mapping
    prop_groups = [g for g, members in mapping.items() if not _is_robot_group(g, members)]
    if prop_groups:
        full["og_task_config"]["group_xyz_randomization"] = {
            group: list(_DEFAULT_GROUP_XYZ) for group in prop_groups
        }
        full["og_task_config"]["group_z_rot_randomization"] = {
            group: _DEFAULT_GROUP_Z_ROT for group in prop_groups
        }
    full["og_task_config"]["goal_predicates_all"] = proposed.get("goal_predicates_all")
    full["og_task_config"]["goal_predicates_any"] = proposed.get("goal_predicates_any")
    instruction = proposed.get("language_instruction")
    instruction = instruction.strip() if isinstance(instruction, str) else ""
    full["language_instruction"] = instruction or _derive_instruction(
        mapping,
        full["og_task_config"]["goal_predicates_all"],
        full["og_task_config"]["goal_predicates_any"],
    )
    return task_name, full


def propose_task(prompt, objects, image_bytes, scene_name, robot_type="franka"):
    """Ask Gemini for one task yaml matching *prompt*, scoped to *objects*.

    Args:
        prompt (str): The user's free-text task description.
        objects (list[dict]): ``{"name": ..., "category": ...}`` for every
            editable prop currently in the scene.
        image_bytes (bytes): A PNG of the current viewport.
        scene_name (str): Used as the task_name prefix.
        robot_type (str): Selects the constraint text folded into the prompt;
            an arm with no :data:`ROBOT_CONSTRAINTS` entry gets none.

    Returns:
        dict: ``{"refused": True, "reason": ...}`` when the instruction cannot
        be carried out with these objects -- an answer, not an error.
        Otherwise ``{"task_name", "filename", "yaml_text", "problems"}``, where
        ``problems`` is :func:`task_semantics.validate_task`'s verdict, so a
        flawed proposal can still be shown and edited.
    """
    if not prompt or not prompt.strip():
        raise TaskProposeError("Type a description of the task first.")
    if not objects:
        raise TaskProposeError("This scene has no editable objects to build a task from.")

    prompt_text = PROMPT_TEMPLATE.format(
        robot_type=robot_type or "franka",
        user_prompt=prompt.strip(),
        robot_constraint=_robot_constraint(robot_type or "franka"),
        object_list=_format_object_list(objects),
        states=OG_OBJECT_STATES,
    )

    response_text = _call_gemini(prompt_text, image_bytes)
    # Check for a refusal before parsing, which would report it as a missing task_name.
    declined = _refusal(response_text)
    if declined:
        return {"refused": True, "reason": declined}
    proposed = _parse_task(response_text)
    task_name, full = _build_full_task(proposed, scene_name)

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", task_name)
    yaml_text = yaml.safe_dump(full, sort_keys=False)
    return {
        "task_name": task_name,
        "filename": f"{safe_name}.yaml",
        "yaml_text": yaml_text,
        # Validated on the assembled document, not on Gemini's fragment.
        "problems": task_semantics.validate_task(full),
    }
