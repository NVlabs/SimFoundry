# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Cousin hot-swap for the interactive scene editor.

Ported verbatim from ``interactive_scene_editor_cousin_swap.py``, a 3,191-line
fork that had drifted 2,300 lines behind the main editor and had never been
maintained separately. The 440 lines of cousin logic were the only thing in it
that the main editor did not already have — and increasingly the only thing
worth keeping, since the fork lacked the HUD, the non-blocking scale dialog, the
callback guard, the keymap table and pre-flight validation.

Kept as a mixin in its own module so it is completely inert unless
``--cousins_combinations`` is passed: nothing here runs, and nothing is bound,
when the feature is off.

Inputs come from the B_augmentation pipeline:

* ``combinations.json`` from
  ``B_augmentation/stages/2_generate_cousin_combinations.py``
* the ``custom-assets`` dataset from
  ``B_augmentation/stages/5_import_cousin_usd.py``, laid out as
  ``deps/BEHAVIOR-1K/datasets/custom-assets/objects/<category>/<model>/usd/<model>.usd``

**Unverified.** This module has not been exercised end-to-end: it compiles and
degrades gracefully, but has never swapped an actual cousin. Run B_augmentation,
then exercise this once before trusting it — and before deleting the fork it
came from.
"""

import json
import os
import re
from pathlib import Path

import omnigibson as og
import omnigibson.lazy as lazy
import torch as th
from omnigibson.objects import USDObject
from omnigibson.utils.ui_utils import KeyboardEventHandler


class CousinSwapMixin:
    """Hot-swap scene objects for generated digital cousins.

    Expects the host class to provide: ``scene``, ``objects``, ``object_names``,
    ``selected_idx``, ``initial_poses``, ``initial_scales``, ``usd_object_paths``
    and ``scene_objects_info_names``.
    """

    def init_cousin_swap(self, combinations_path=None, dataset_name="custom-assets",
                         swap_key="H", settle_steps=0):
        """Configure cousin swapping. Call from the host's __init__.

        Args:
            combinations_path (str or None): Path to combinations.json. None disables
                the feature entirely.
            dataset_name (str): Dataset under deps/BEHAVIOR-1K/datasets/.
            swap_key (str): Key that advances to the next combination.
            settle_steps (int): Physics steps to run after a swap.
        """
        self.cousins_combinations_path = combinations_path
        self.cousins_dataset_name = dataset_name
        self.cousins_swap_key = swap_key
        self.cousins_settle_steps = max(0, int(settle_steps))
        self.swap_combinations = []
        self.curr_cousin_idx = -1
        self.pending_cousin_swap = False
        self.is_swapping_cousins = False

    @property
    def cousin_swap_enabled(self):
        return bool(getattr(self, "cousins_combinations_path", None))

    def service_pending_cousin_swap(self):
        """Run a requested swap. Call once per frame from the host's run loop.

        The keypress only sets a flag: swapping tears the scene down and rebuilds
        it, which must not happen inside a carb input callback.
        """
        if getattr(self, "pending_cousin_swap", False):
            self.pending_cousin_swap = False
            self.hot_swap_cousins()

    def _parse_cousin_from_path(self, usd_path):
        """
        Parse a cousin category folder to get category and model.

        Example:
            usd_path = ../../deps/BEHAVIOR-1K/datasets/{DATASET_NAME}/objects/blue_bowl_cousin_003_v3
            category = blue_bowl_cousin_003_v3
            model = wuujbj
        """
        parts = Path(usd_path).parts
        category = parts[-1]
        model = next(p.name for p in Path(usd_path).iterdir() if p.is_dir())
        return category, model

    def _find_object_by_prefix(self, prefix):
        matches = [
            obj for obj in self.scene.objects
            if obj.name == prefix or obj.name.startswith(prefix + "_")
        ]

        # Backward compatibility:
        # combinations.json often uses keys like "iter_7", while objects restored
        # from saved scene JSON are named with long descriptors ending in "_7".
        # If direct prefix matching fails, map iter_<idx> to any object whose name
        # ends with _<idx>.
        if len(matches) == 0 and prefix.startswith("iter_"):
            idx = prefix.split("iter_", 1)[1]
            if idx.isdigit():
                matches = [
                    obj for obj in self.scene.objects
                    if obj.name.endswith(f"_{idx}")
                ]

        if len(matches) == 0:
            print(f"[HOT SWAP] Nothing found to swap for prefix={prefix}")
            return None

        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple objects found for prefix {prefix}: "
                f"{[o.name for o in matches]}"
            )

        return matches[0]

    def _get_cousins_dataset_root(self):
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "deps" / "BEHAVIOR-1K" / "datasets" / self.cousins_dataset_name / "objects"

    def _normalize_category_for_match(self, text):
        """
        Normalize category names so variants like:
        - a___b, a_b
        - trader_joe's vs trader_joe_s
        - head_&_shoulders vs head_shoulders
        can still match.
        """
        normalized = re.sub(r"[^a-z0-9]+", "_", text.lower())
        normalized = re.sub(r"_+", "_", normalized)
        return normalized.strip("_")

    def _find_matching_cousin_folders(self, dataset_root, cousin_category):
        # 1) Fast path exact match
        exact_matches = [
            p for p in dataset_root.iterdir()
            if p.is_dir() and p.name == cousin_category
        ]
        if exact_matches:
            return exact_matches

        # 2) Robust normalized match
        target_norm = self._normalize_category_for_match(cousin_category)
        normalized_matches = [
            p for p in dataset_root.iterdir()
            if p.is_dir() and self._normalize_category_for_match(p.name) == target_norm
        ]
        return normalized_matches

    def _usd_has_single_root_link(self, usd_path):
        """
        Best-effort static check that mirrors OmniGibson's root-link logic.

        Returns:
            bool or None:
                - True/False when check succeeds
                - None if pxr is unavailable or USD parsing fails
        """
        try:
            from pxr import Usd
        except Exception:
            return None

        try:
            stage = Usd.Stage.Open(str(usd_path))
            if stage is None:
                return False

            root_prim = stage.GetDefaultPrim()
            if not root_prim or not root_prim.IsValid():
                active_children = [p for p in stage.GetPseudoRoot().GetChildren() if p.IsActive()]
                if len(active_children) != 1:
                    return False
                root_prim = active_children[0]

            links_to_create = set()
            joint_children = set()
            for prim in root_prim.GetChildren():
                if prim.GetTypeName() != "Xform":
                    continue

                link_name = prim.GetName()
                links_to_create.add(link_name)

                for child_prim in prim.GetChildren():
                    if "joint" not in child_prim.GetTypeName().lower():
                        continue

                    rels = {r.GetName(): r for r in child_prim.GetRelationships()}
                    body0 = rels.get("physics:body0")
                    body1 = rels.get("physics:body1")
                    if body0 is None or body1 is None:
                        continue

                    body0_targets = body0.GetTargets()
                    body1_targets = body1.GetTargets()
                    if not body0_targets or not body1_targets:
                        continue

                    joint_children.add(body1_targets[0].pathString.split("/")[-1])

            valid_root_links = list(links_to_create - joint_children)
            return len(valid_root_links) == 1
        except Exception as e:
            print(f"[HOT SWAP] Warning: failed to parse USD for root-link check ({usd_path}): {e}")
            return None

    def _select_valid_cousin_asset(self, matching_folders, min_usd_size_bytes=4096):
        """
        Pick the best cousin asset candidate from matched category folders.

        We sometimes have multiple normalized matches (e.g., apostrophe vs underscore
        category variants), and some generated USDs are tiny / incomplete. Prefer
        candidates with an OG-compatible root-link structure, then existing / larger USDs.
        """
        candidates = []
        for folder in sorted(matching_folders):
            model_dirs = sorted([p for p in folder.iterdir() if p.is_dir()])
            for model_dir in model_dirs:
                model = model_dir.name
                usd_path = model_dir / "usd" / f"{model}.usd"
                if not usd_path.exists():
                    continue
                try:
                    size = usd_path.stat().st_size
                except OSError:
                    continue
                has_single_root_link = self._usd_has_single_root_link(usd_path)
                candidates.append((has_single_root_link, size, folder, model, usd_path))

        if not candidates:
            return None, None, None

        # First prefer OG-compatible root-link assets (when check is available),
        # then prefer non-tiny USDs (usually complete exports), then largest size.
        root_valid = [c for c in candidates if c[0] is True]
        root_unknown = [c for c in candidates if c[0] is None]
        root_invalid = [c for c in candidates if c[0] is False]

        if root_valid:
            pool_by_root = root_valid
        elif root_unknown:
            pool_by_root = root_unknown
        else:
            pool_by_root = root_invalid
            print("[HOT SWAP] Warning: all candidate cousins failed root-link precheck; using best-effort fallback.")

        non_tiny = [c for c in pool_by_root if c[1] >= min_usd_size_bytes]
        pool = non_tiny if non_tiny else pool_by_root
        has_single_root_link, size, folder, model, usd_path = max(pool, key=lambda c: c[1])
        if len(candidates) > 1:
            print(
                f"[HOT SWAP] Selected cousin candidate model={model} "
                f"(usd_size={size} bytes, root_link_ok={has_single_root_link}) from {folder.name}"
            )
        return folder, model, usd_path

    def _load_cousins_combinations(self):
        if self.cousins_combinations_path is None:
            return False
        if not os.path.exists(self.cousins_combinations_path):
            print(f"[HOT SWAP] combinations.json not found: {self.cousins_combinations_path}")
            return False
        with open(self.cousins_combinations_path, "r") as f:
            self.swap_combinations = json.load(f)
        if not self.swap_combinations:
            print("[HOT SWAP] combinations.json is empty. Hot-swap disabled.")
            return False
        print(f"[HOT SWAP] Loaded {len(self.swap_combinations)} combinations.")
        return True

    def _resolve_keyboard_input(self, key_name):
        if key_name is None:
            return None
        key_name = key_name.strip().upper()
        mapping = {
            "SPACE": "SPACE",
            "ENTER": "ENTER",
            "BACKSPACE": "BACKSPACE",
            "ESC": "ESCAPE",
            "ESCAPE": "ESCAPE",
            "TAB": "TAB",
            "GRAVE": "GRAVE",
            "UP": "UP",
            "DOWN": "DOWN",
            "LEFT": "LEFT",
            "RIGHT": "RIGHT",
            "PAGE_UP": "PAGE_UP",
            "PAGE_DOWN": "PAGE_DOWN",
            "APOSTROPHE": "APOSTROPHE",
            "SLASH": "SLASH",
            "COMMA": "COMMA",
            "PERIOD": "PERIOD",
            "MINUS": "MINUS",
            "EQUAL": "EQUAL",
        }
        if len(key_name) == 1 and key_name.isalpha():
            attr = key_name
        elif len(key_name) == 1 and key_name.isdigit():
            attr = f"KEY_{key_name}"
        elif key_name.startswith("F") and key_name[1:].isdigit():
            attr = key_name
        else:
            attr = mapping.get(key_name, key_name)
        if not hasattr(lazy.carb.input.KeyboardInput, attr):
            print(f"[HOT SWAP] Unknown key '{key_name}' for cousins swap. Skipping hot-swap key binding.")
            return None
        return getattr(lazy.carb.input.KeyboardInput, attr)

    def _setup_cousins_hot_swap_key(self):
        if not self._load_cousins_combinations():
            return
        key = self._resolve_keyboard_input(self.cousins_swap_key)
        if key is None:
            return
        KeyboardEventHandler.add_keyboard_callback(
            key=key,
            callback_fn=self.request_cousins_hot_swap
        )
        print(f"[HOT SWAP] Press '{self.cousins_swap_key}' to swap cousins.")

    def request_cousins_hot_swap(self):
        self.pending_cousin_swap = True

    def _remove_object_tracking(self, obj_name):
        old_index = None
        was_selected = False
        if obj_name in self.object_names:
            old_index = self.object_names.index(obj_name)
            was_selected = (self.selected_idx == old_index)
            self.object_names.remove(obj_name)
            if self.selected_idx > old_index:
                self.selected_idx -= 1
        self.objects.pop(obj_name, None)
        self.initial_poses.pop(obj_name, None)
        self.initial_scales.pop(obj_name, None)
        self.usd_object_paths.pop(obj_name, None)
        if obj_name in self.scene_objects_info_names:
            self.scene_objects_info_names.remove(obj_name)
        return old_index, was_selected

    def _add_object_tracking(self, obj_name, obj, pos, ori, usd_path, old_index=None, was_selected=False):
        if old_index is not None and old_index <= len(self.object_names):
            self.object_names.insert(old_index, obj_name)
            if was_selected:
                self.selected_idx = old_index
        else:
            self.object_names.append(obj_name)
            if was_selected:
                self.selected_idx = len(self.object_names) - 1
        self.objects[obj_name] = obj
        self.initial_poses[obj_name] = (pos.clone(), ori.clone())
        self.initial_scales[obj_name] = obj.scale.clone()
        self.usd_object_paths[obj_name] = usd_path

    def hot_swap_cousins(self):
        """
        Hot-swap cousins by fully reloading the scene.

        Process:
        1. Pause simulation
        2. Collect all objects to swap and their new cousin info
        3. Remove all old objects from scene
        4. Add all new cousin objects
        5. Set all poses
        6. Initialize physics
        7. Re-enable simulation if it was playing
        """
        if self.is_swapping_cousins or not self.swap_combinations:
            return

        self.is_swapping_cousins = True
        self.curr_cousin_idx = (self.curr_cousin_idx + 1) % len(self.swap_combinations)
        combo = self.swap_combinations[self.curr_cousin_idx]

        print(f"[HOT SWAP] Starting cousin swap, combo idx = {self.curr_cousin_idx}")

        dataset_root = self._get_cousins_dataset_root()
        if not dataset_root.exists():
            print(f"[HOT SWAP] Dataset root not found: {dataset_root}")
            self.is_swapping_cousins = False
            return

        # Step 1: Pause simulation
        was_playing = og.sim.is_playing()
        if was_playing:
            print("[HOT SWAP] Pausing simulation...")
            og.sim.stop()

        # Step 2: Collect all swap information
        print("[HOT SWAP] Collecting swap information...")
        swap_info = []  # List of dicts with old_obj, new_usd_path, pos, ori, scale, etc.

        for obj_prefix, cousin_path in combo.items():
            print(f"[HOT SWAP] Processing obj_prefix={obj_prefix}, cousin_path={cousin_path}")

            old_obj = self._find_object_by_prefix(obj_prefix)
            if old_obj is None:
                print(f"[HOT SWAP] {obj_prefix} not found, skipping")
                continue

            # Parse cousin path to find the new USD
            old_category = old_obj.category
            base_category = old_category.split("_cousin_")[0] if "_cousin_" in old_category else old_category

            filestem = Path(cousin_path).stem
            if filestem.endswith("_transparent"):
                cousin_suffix = filestem[:-len("_transparent")]
            else:
                cousin_suffix = filestem

            cousin_category = f"{base_category}_{cousin_suffix}"

            matching_folders = self._find_matching_cousin_folders(dataset_root, cousin_category)

            if not matching_folders:
                print(f"[HOT SWAP] No folder found matching '{cousin_category}'")
                continue

            folder, model, usd_file = self._select_valid_cousin_asset(matching_folders)
            if folder is None:
                print(f"[HOT SWAP] No valid USD found for '{cousin_category}'")
                continue
            usd_path = os.path.abspath(str(usd_file))

            # Store swap info
            pos, orn = old_obj.get_position_orientation()
            swap_info.append({
                "old_obj": old_obj,
                "old_name": old_obj.name,
                "old_index": self.object_names.index(old_obj.name) if old_obj.name in self.object_names else None,
                "was_selected": (self.selected_idx == self.object_names.index(old_obj.name)) if old_obj.name in self.object_names else False,
                "was_in_scene_group": old_obj.name in self.scene_objects_info_names,
                "obj_prefix": obj_prefix,
                "new_model": model,
                "new_usd_path": usd_path,
                "old_category": old_category,
                "pos": pos.clone(),
                "ori": orn.clone(),
                "scale": old_obj.scale.clone(),
            })

        if not swap_info:
            print("[HOT SWAP] No valid objects to swap")
            self.is_swapping_cousins = False
            if was_playing:
                og.sim.play()
            return

        # Step 3: Remove all old objects
        print(f"[HOT SWAP] Removing {len(swap_info)} old objects...")
        with og.sim.stopped():
            for info in swap_info:
                print(f"  └─ Removing {info['old_name']}")
                self.scene.remove_object(info['old_obj'])
                self._remove_object_tracking(info['old_name'])

        # Step 4: Add all new objects
        print(f"[HOT SWAP] Adding {len(swap_info)} new cousin objects...")
        with og.sim.stopped():
            for info in swap_info:
                new_name = f"{info['obj_prefix']}_{info['new_model']}"
                print(f"  └─ Adding {new_name}")

                new_obj = USDObject(
                    name=new_name,
                    usd_path=info['new_usd_path'],
                    category=info['old_category'],
                    model=info['new_model'],
                    dataset_name=self.cousins_dataset_name,
                    scale=[1.0, 1.0, 1.0],
                )

                self.scene.add_object(new_obj)
                og.sim.step()

                # Store new object info for pose setting
                info['new_obj'] = new_obj
                info['new_name'] = new_name

        # Step 5: Set all poses
        print(f"[HOT SWAP] Setting poses for {len(swap_info)} objects...")
        with og.sim.stopped():
            for info in swap_info:
                print(f"  └─ Setting pose for {info['new_name']}")
                info['new_obj'].set_position_orientation(info['pos'], info['ori'])
                info['new_obj'].scale = info['scale']

                # Update tracking
                self._add_object_tracking(
                    obj_name=info['new_name'],
                    obj=info['new_obj'],
                    pos=info['pos'],
                    ori=info['ori'],
                    usd_path=info['new_usd_path'],
                    old_index=info['old_index'],
                    was_selected=info['was_selected'],
                )

                if info['was_in_scene_group']:
                    self.scene_objects_info_names.append(info['new_name'])

        # Step 6: Initialize physics
        print("[HOT SWAP] Initializing physics...")
        with og.sim.stopped():
            og.sim.initialize_physics()
            self.scene.update_initial_file()

        # Step 7: Step physics while keeping objects still to reduce post-swap "explosive" rebound energy
        if self.cousins_settle_steps > 0:
            print(f"[HOT SWAP] Settling objects for {self.cousins_settle_steps} physics steps...")
            for _ in range(self.cousins_settle_steps):
                og.sim.step_physics()
                for obj in self.scene.objects:
                    if hasattr(obj, "keep_still"):
                        obj.keep_still()

        # Step 8: Re-enable simulation if it was playing
        if was_playing:
            print("[HOT SWAP] Resuming simulation...")
            og.sim.play()

        print(f"[HOT SWAP] Swap complete! Swapped {len(swap_info)} objects.")
        self.is_swapping_cousins = False
