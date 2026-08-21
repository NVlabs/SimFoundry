# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Propose tabletop tasks for a scene using a VLM (Gemini 3 Pro).

Reads scene.png and objects_info from {scene_name}_scene_state_latest.json in
assets/scenes/{scene_name}/, prompts the VLM to propose tasks that a robot arm
can perform in OmniGibson, and writes task configs to
scripts/cfg/task/{scene_name}_{task_name}.yaml.

Usage (from repo root or scripts/):
  python scripts/pipeline/B_augmentation/stages/7_propose_scene_tasks.py scene_name=nv_desk

Requires: GOOGLE_CLOUD_PROJECT or gcloud_project in config; google-genai / vertexai.
"""
import re
import json
import hydra
from omegaconf import OmegaConf
from pathlib import Path

from simfoundry.models.vlm import Gemini


# OmniGibson object states (from omnigibson/object_states) useful for tabletop tasks
# OG_OBJECT_STATES = (
#     "OnTop, Touching, Inside, NextTo, Under, Open, ToggledOn, AttachedTo, "
#     "Covered, Filled, Cooked, Heated, Frozen, Burnt, OnFire, HorizontalAdjacency, "
#     "VerticalAdjacency, Contains, Draped, Overlaid, IsGrasping"
# )

OG_OBJECT_STATES = (
    "OnTop, OnTopAABB, InsideAABB, AboveAABB, Lifted"
)

# Robot constraints (gripper info, etc.) for different robot types
ROBOT_CONSTRAINTS = {
    "franka": {
        "constraint": "The robot gripper has a maximum opening length of 85 mm. When proposing tasks that involve grasping, choose objects that can fit within this grasp (e.g. avoid proposing to grasp very large or wide objects that would exceed 85 mm).",
    }
}

PROMPT_TEMPLATE = """You are a professional robotic expert in sim-to-real, robot learning, and VLAs.

Look at the provided image of a tabletop scene in a simulation (OmniGibson). Using ONLY the objects listed below, propose exactly {num_tasks} distinct tasks that a {robot_type} arm can perform in this scene, ensure that each task goal requires a change from the current configuration of the scene, for example, do not propose tasks to put a pear in bowl if it is already in the bowl. Each task should have clear goal conditions expressible with object states.

Important: {robot_constraint}
{object_constraints}

Scene object list (id -> category, name; use these names as group identifiers in predicates):
{object_list}

Output format: for each of the {num_tasks} tasks, output a YAML block that can be merged into a task config. Use this exact structure for each task.

For each task provide:
1. task_name: short snake_case name (e.g. stack_cup_on_plate)
2. semantic_group_mapping: map each logical group name you use (e.g. cup, plate) to a list containing exactly one scene object "name" from the list above (e.g. cup: [blue_cup_syvtml_19], plate: [teal_plate_qdudop_16]).
3. goal_predicates_all: list of predicates that must ALL be true for success. Each predicate has: state, state_kwargs (null if not needed), value (true/false), group, other_group (only for binary states like OnTop, Touching).
4. goal_predicates_any: list of predicates where ANY being true yields success (optional; use null if not needed).

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
    - state: InsideAABB
      state_kwargs:
        volume_threshold: 0.5
      value: true
      group: pear
      other_group: bowl
  goal_predicates_any: null

Output exactly {num_tasks} tasks. Separate each task with "---" and a task number (e.g. "--- Task 2"). Each task block must be valid YAML with keys: task_name, semantic_group_mapping, goal_predicates_all, goal_predicates_any."""


def load_scene_objects(scene_state_path: str) -> dict:
    """Load objects from {scene_name}_scene_state_latest.json.

    Reads objects_info.init_info and returns dict of
    name -> {category, name} for non-robot, non-background objects.
    """
    with open(scene_state_path, "r") as f:
        data = json.load(f)
    init_info = data.get("objects_info", {}).get("init_info", {})
    objects = {}
    for obj_key, obj_data in init_info.items():
        args = obj_data.get("args", {})
        category = args.get("category", "")
        name = args.get("name", obj_key)
        # Skip robot and background entries
        if obj_data.get("class_name", "").lower().startswith("franka") or \
           obj_data.get("class_name", "").lower().startswith("robot") or \
           "robot" in obj_key.lower() or \
           "background" in category.lower():
            continue
        objects[obj_key] = {"category": category, "name": name}
    return objects


def format_object_list_for_prompt(objects: dict) -> str:
    """Format scene objects as a string for the VLM prompt."""
    lines = []
    for obj_id, info in sorted(objects.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]):
        lines.append(
            f'  "{obj_id}": {{"category": "{info.get("category", "?")}", "name": "{info.get("name", "?")}"}}'
        )
    return "{\n" + ",\n".join(lines) + "\n}"


def resolve_specifiers_to_object_names(objects: dict, specifiers: list) -> list[str]:
    """
    Resolve a list of specifiers to scene object names.
    Each specifier can be an object name (e.g. iter_0) or a category (e.g. banana, white_plate).
    Categories are expanded to all object names that have that category (case-insensitive match).
    Returns a deduplicated list of object names in scene order.
    """
    if not specifiers:
        return []
    name_to_id = {}
    category_to_names = {}  # key: category.lower() -> list of object names
    for obj_id, info in objects.items():
        name = info.get("name") or str(obj_id)
        cat = (info.get("category") or "").strip()
        name_to_id[name] = obj_id
        if cat:
            category_to_names.setdefault(cat.lower(), []).append(name)
    seen = set()
    result = []
    for spec in specifiers:
        spec = (spec or "").strip()
        if not spec:
            continue
        # Exact object name
        if spec in name_to_id and spec not in seen:
            seen.add(spec)
            result.append(spec)
            continue
        # Category (case-insensitive)
        spec_lower = spec.lower()
        if spec_lower in category_to_names:
            for n in category_to_names[spec_lower]:
                if n not in seen:
                    seen.add(n)
                    result.append(n)
    return result


def format_object_constraints(include_any: list, include_all: list, exclude_all: list) -> str:
    """Build prompt text for include_any, include_all, exclude_all object constraints."""
    parts = []
    if include_any:
        names = ", ".join(include_any)
        parts.append(
            f"Object inclusion (include_any): The following objects must each appear in at least one of the proposed tasks (across all tasks, every listed object must be used at least once): [{names}]."
        )
    if include_all:
        names = ", ".join(include_all)
        parts.append(
            f"Object inclusion (include_all): The following objects must appear in every proposed task's goal_predicates (each task must involve all of them): [{names}]."
        )
    if exclude_all:
        names = ", ".join(exclude_all)
        parts.append(
            f"Object exclusion (exclude_all): Do not use any of the following objects in any proposed task: [{names}]."
        )
    if not parts:
        return ""
    return "Object constraints (follow strictly):\n- " + "\n- ".join(parts) + "\n\n"


def build_full_task_yaml(proposed: dict, base_template: dict, scene_name: str) -> str:
    """Merge VLM-proposed goal/config into a full task YAML (OmegaConf-friendly)."""
    raw_task_name = proposed.get("task_name", "proposed_task")
    task_name = f"{scene_name}_{raw_task_name}"
    semantic_group_mapping = proposed.get("semantic_group_mapping") or {}
    goal_predicates_all = proposed.get("goal_predicates_all")
    goal_predicates_any = proposed.get("goal_predicates_any")

    # Start from load_scene-like template
    full = dict(base_template)
    full["task_name"] = task_name
    full["og_task_config"] = full.get("og_task_config", {})
    full["og_task_config"]["activity_name"] = task_name
    full["og_task_config"]["semantic_group_mapping"] = semantic_group_mapping
    full["og_task_config"]["goal_predicates_all"] = goal_predicates_all
    full["og_task_config"]["goal_predicates_any"] = goal_predicates_any
    return full


def parse_vlm_tasks(text: str) -> list[dict]:
    """Parse VLM response into a list of task dicts (task_name, semantic_group_mapping, goal_predicates_*)."""
    def _clean_block(block: str) -> str:
        block = block.strip()
        if block.startswith("```"):
            lines = block.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            block = "\n".join(lines).strip()

        # Trim any prose, headings, or task labels before the actual YAML payload.
        task_name_match = re.search(r"^task_name:", block, re.MULTILINE)
        if task_name_match:
            block = block[task_name_match.start():]
        return block.strip()

    def _parse_block(block: str) -> dict | None:
        block = _clean_block(block)
        if len(block) < 10:
            return None
        try:
            parsed = OmegaConf.create(block)
            if isinstance(parsed, dict) or "task_name" in parsed:
                task = OmegaConf.to_container(parsed, resolve=True)
                if isinstance(task, dict) and task.get("task_name"):
                    return task
        except Exception:
            pass
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict) and parsed.get("task_name"):
                return parsed
        except Exception:
            pass
        return None

    def _append_task(task: dict | None):
        if task is None:
            return
        signature = json.dumps(task, sort_keys=True)
        if signature not in seen:
            seen.add(signature)
            tasks.append(task)

    tasks = []
    seen = set()

    # Extract YAML/JSON from markdown code blocks first
    code_blocks = re.findall(r"```(?:yaml)?\s*\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        for block in code_blocks:
            block = block.strip()
            if len(block) < 15:
                continue
            # Single block may contain multiple YAML docs separated by ---.
            sub_blocks = re.split(r"(?:^|\n)---[^\n]*\n", block)
            for sub in sub_blocks:
                _append_task(_parse_block(sub))
        if tasks:
            return tasks

    # Fallback: split by standard YAML document separators, including variants
    # such as "--- Task 2" and "---\n# Task 2".
    raw_blocks = re.split(r"(?:^|\n)---[^\n]*\n", text)
    for block in raw_blocks:
        _append_task(_parse_block(block))

    if tasks:
        return tasks

    # Last resort: split responses that emit repeated task_name blocks without
    # a YAML document separator.
    raw_blocks = re.split(r"(?=^task_name:)", text, flags=re.MULTILINE)
    for block in raw_blocks:
        _append_task(_parse_block(block))
    return tasks


def get_base_task_template() -> dict:
    """Default task template (same structure as load_scene / stack_dishware)."""
    return {
        "task_name": "proposed_task",
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


from simfoundry import ASSET_DIR as SIMFOUNDRY_ASSET_DIR, CFG_DIR


@hydra.main(version_base=None, config_path=CFG_DIR, config_name="real2sim_cfg")
def main(cfg):
    scene_name = cfg.scene_name
    scene_dir = (Path(SIMFOUNDRY_ASSET_DIR) / "scenes" / scene_name).resolve()

    # Scene image: assets/scenes/{scene_name}/scene.png
    scene_image_path = scene_dir / "scene.png"

    # Scene state JSON: assets/scenes/{scene_name}/{scene_name}_scene_state_latest.json
    scene_state_path = scene_dir / f"{scene_name}_scene_state_latest.json"

    # Output directory: defaults to config-controlled augmentation output.
    out_dir = Path(cfg.propose_scene_task.out_dir).resolve()

    if not scene_state_path.exists():
        scene_state_path = Path(cfg.s14_og.out_dir) / "reconstructed_og_scene.json"
    if not scene_image_path.exists():
        fallback_images = [
            Path(cfg.s14_og.out_dir) / "auto_generation" / "scene_000.png",
            Path(cfg.s1_video.out_dir) / "frames_all" / "frame_0001.png",
        ]
        scene_image_path = next((p for p in fallback_images if p.exists()), scene_image_path)

    if not scene_state_path.exists():
        raise FileNotFoundError(f"Scene state JSON not found: {scene_state_path}")
    if not scene_image_path.exists():
        raise FileNotFoundError(f"Scene image not found: {scene_image_path}")

    num_tasks = cfg.propose_scene_task.num_tasks

    objects = load_scene_objects(str(scene_state_path))
    print(f"Loaded {len(objects)} objects from {scene_state_path}")
    for obj_id, info in objects.items():
        print(f"  {obj_id}: {info.get('category', '?')} {info.get('name', '?')}")
    object_list_str = format_object_list_for_prompt(objects)
    raw_include_any = cfg.propose_scene_task.get("include_any") or []
    raw_include_all = cfg.propose_scene_task.get("include_all") or []
    raw_exclude_all = cfg.propose_scene_task.get("exclude_all") or []
    include_any = resolve_specifiers_to_object_names(objects, raw_include_any)
    include_all = resolve_specifiers_to_object_names(objects, raw_include_all)
    exclude_all = resolve_specifiers_to_object_names(objects, raw_exclude_all)
    object_constraints = format_object_constraints(include_any, include_all, exclude_all)

    # Get robot type from config (default: franka)
    robot_type = cfg.propose_scene_task.get("robot_type", "franka")
    robot_constraint = cfg.propose_scene_task.get("robot_constraint")
    if robot_constraint is None:
        robot_info = ROBOT_CONSTRAINTS.get(robot_type.lower(), ROBOT_CONSTRAINTS["franka"])
        robot_constraint = robot_info["constraint"]

    prompt = PROMPT_TEMPLATE.format(
        object_list=object_list_str,
        states=OG_OBJECT_STATES,
        num_tasks=num_tasks,
        object_constraints=object_constraints,
        robot_type=robot_type,
        robot_constraint=robot_constraint,
    )
    print("Prompt:")
    print(prompt)

    vlm = Gemini(
        project=cfg.gcloud_project,
        location="global",
        model=cfg.propose_scene_task.vlm_model,
    )
    print(f"Calling VLM with scene image and object list...")
    temperature = cfg.propose_scene_task.get("temperature", 0.7)
    remote_retries = int(cfg.propose_scene_task.get("remote_retries", 3))
    result = vlm(
        prompt=prompt,
        image_paths=[str(scene_image_path)],
        temperature=temperature,
        n_retries=remote_retries,
    )
    if result is None:
        raise RuntimeError(
            f"Task proposal VLM returned no response. "
            f"Check propose_scene_task.vlm_model={cfg.propose_scene_task.vlm_model} and API access."
        )
    response_text = vlm.get_result_text(result)
    print("VLM response (first 1500 chars):")
    print(response_text[:1500])
    if len(response_text) > 1500:
        print("...")

    tasks = parse_vlm_tasks(response_text)
    if not tasks:
        print("No tasks parsed from VLM response. Saving raw response for debugging.")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "propose_raw_response.txt", "w") as f:
            f.write(response_text)
        return

    base_template = get_base_task_template()
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, task_dict in enumerate(tasks[:num_tasks], start=1):
        full = build_full_task_yaml(task_dict, base_template, scene_name)
        task_name = task_dict.get("task_name", f"proposed_task_{i}")
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", task_name)
        out_path = out_dir / f"{scene_name}_{safe_name}.yaml"
        OmegaConf.save(OmegaConf.create(full), out_path)
        print(f"Wrote {out_path}")
    if len(tasks) > num_tasks:
        print(f"Ignored {len(tasks) - num_tasks} extra task(s) beyond {num_tasks}.")


if __name__ == "__main__":
    main()
