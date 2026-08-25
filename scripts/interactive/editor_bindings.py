# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The single source of truth for the interactive scene editor's keymap.

This table drives all three consumers — the carb callback registration, the HUD
legend, and ``print_controls()``. Before it existed the same keymap was written
out by hand in four places (module docstring, registration, printed help, HUD),
and they drifted: the docstring documented TAB for cycling and W/S for vertical
motion long after the bindings had become ``[``/``]`` and ``W``/``Q``.

Each entry names a key and an *action id*. The editor supplies the callables in
a dict keyed by the same ids, so a binding without an implementation — or an
implementation nobody bound — is a startup error rather than a silent gap.
"""

from collections import OrderedDict

# (key_name, action_id, help_text). key_name is a carb.input.KeyboardInput
# attribute, resolved at registration time so this module stays importable
# without Isaac Sim.
BINDINGS = [
    # --- Selection ---
    ("LEFT_BRACKET", "cycle_forward", "Cycle through objects (forward)"),
    ("RIGHT_BRACKET", "cycle_backward", "Cycle through objects (backward)"),

    # --- Translation, robot base frame (world if no robot) ---
    ("UP", "translate_x_plus", "Move +X"),
    ("DOWN", "translate_x_minus", "Move -X"),
    ("LEFT", "translate_y_plus", "Move +Y"),
    ("RIGHT", "translate_y_minus", "Move -Y"),
    ("W", "translate_z_plus", "Move +Z"),
    ("Q", "translate_z_minus", "Move -Z"),

    # --- Rotation, local ---
    ("N", "rotate_x_plus", "Rotate +X (pitch)"),
    ("M", "rotate_x_minus", "Rotate -X"),
    ("C", "rotate_y_plus", "Rotate +Y (roll)"),
    ("V", "rotate_y_minus", "Rotate -Y"),
    ("SLASH", "rotate_z_plus", "Rotate +Z (yaw)"),
    ("APOSTROPHE", "rotate_z_minus", "Rotate -Z"),

    # --- Rotation, global ---
    ("KEY_1", "rotate_global_x_plus", "Rotate +X (global)"),
    ("KEY_2", "rotate_global_x_minus", "Rotate -X (global)"),
    ("KEY_3", "rotate_global_y_plus", "Rotate +Y (global)"),
    ("KEY_4", "rotate_global_y_minus", "Rotate -Y (global)"),
    ("Z", "rotate_global_z_plus", "Rotate +Z (global)"),
    ("X", "rotate_global_z_minus", "Rotate -Z (global)"),

    # --- Scale ---
    ("EQUAL", "scale_up", "Scale up"),
    ("MINUS", "scale_down", "Scale down"),
    ("NUMPAD_ADD", "scale_up", "Scale up (numpad)"),
    ("NUMPAD_SUBTRACT", "scale_down", "Scale down (numpad)"),
    ("S", "set_scale", "Set exact scale (dialog)"),

    # --- Step sizes ---
    ("KEY_5", "translation_delta_up", "Increase translation step"),
    ("KEY_6", "translation_delta_down", "Decrease translation step"),
    ("KEY_7", "rotation_delta_up", "Increase rotation step"),
    ("KEY_8", "rotation_delta_down", "Decrease rotation step"),
    ("KEY_9", "scale_delta_up", "Increase scale step"),
    ("KEY_0", "scale_delta_down", "Decrease scale step"),

    # --- Simulation ---
    ("SPACE", "toggle_sim", "Toggle play/stop"),
    ("O", "stop_sim", "Stop simulation"),
    ("P", "play_sim", "Play simulation"),

    # --- Skybox ---
    ("R", "skybox_x_plus", "Rotate skybox +X"),
    ("F", "skybox_x_minus", "Rotate skybox -X"),
    ("Y", "skybox_y_plus", "Rotate skybox +Y"),
    ("H", "skybox_y_minus", "Rotate skybox -Y"),
    ("J", "skybox_z_plus", "Rotate skybox +Z"),
    ("L", "skybox_z_minus", "Rotate skybox -Z"),
    ("I", "skybox_brighter", "Increase skybox intensity"),
    ("K", "skybox_dimmer", "Decrease skybox intensity"),

    # --- State ---
    ("ENTER", "save", "Save scene state"),
    ("BACKSPACE", "reset_object", "Reset object to initial pose"),
    ("U", "undo", "Undo"),
    ("D", "delete_object", "Delete selected object (soft)"),
    ("G", "toggle_group", "Toggle group mode"),
    ("F1", "print_pose", "Print current pose"),

    # --- Display ---
    ("F2", "toggle_hud", "Toggle HUD panel"),
    ("F3", "toggle_highlight", "Toggle selection outline"),

    # --- System ---
    ("B", "debug_shell", "Debug shell (needs --debug_shell)"),
    ("ESCAPE", "exit", "Exit"),
]

# Display grouping for the printed help and the HUD legend, in order.
CATEGORIES = OrderedDict([
    ("Selection", ["cycle_forward", "cycle_backward"]),
    ("Translation", ["translate_x_plus", "translate_x_minus", "translate_y_plus",
                     "translate_y_minus", "translate_z_plus", "translate_z_minus"]),
    ("Rotation (local)", ["rotate_x_plus", "rotate_x_minus", "rotate_y_plus",
                          "rotate_y_minus", "rotate_z_plus", "rotate_z_minus"]),
    ("Rotation (global)", ["rotate_global_x_plus", "rotate_global_x_minus",
                           "rotate_global_y_plus", "rotate_global_y_minus",
                           "rotate_global_z_plus", "rotate_global_z_minus"]),
    ("Scale", ["scale_up", "scale_down", "set_scale"]),
    ("Step sizes", ["translation_delta_up", "translation_delta_down",
                    "rotation_delta_up", "rotation_delta_down",
                    "scale_delta_up", "scale_delta_down"]),
    ("Simulation", ["toggle_sim", "stop_sim", "play_sim"]),
    ("Lighting", ["skybox_x_plus", "skybox_x_minus", "skybox_y_plus", "skybox_y_minus",
                  "skybox_z_plus", "skybox_z_minus", "skybox_brighter", "skybox_dimmer"]),
    ("State", ["save", "reset_object", "undo", "delete_object", "toggle_group", "print_pose"]),
    ("Display", ["toggle_hud", "toggle_highlight"]),
    ("System", ["debug_shell", "exit"]),
])

# Friendlier names than the carb constants for anything not self-evident.
KEY_LABELS = {
    "LEFT_BRACKET": "[", "RIGHT_BRACKET": "]", "SLASH": "/", "APOSTROPHE": "'",
    "EQUAL": "+", "MINUS": "-", "NUMPAD_ADD": "num +", "NUMPAD_SUBTRACT": "num -",
    "ESCAPE": "ESC", "SPACE": "SPACE", "ENTER": "ENTER", "BACKSPACE": "BACKSPACE",
    **{f"KEY_{d}": str(d) for d in range(10)},
}


def key_label(key_name):
    """Human-readable label for a carb key name."""
    return KEY_LABELS.get(key_name, key_name)


def keys_for(action_id):
    """Every key bound to an action, in table order."""
    return [key for key, action, _ in BINDINGS if action == action_id]


def help_for(action_id):
    """Help text for an action."""
    for _, action, text in BINDINGS:
        if action == action_id:
            return text
    return ""


def validate(handlers):
    """Cross-check the table against the editor's implemented actions.

    Args:
        handlers (dict): action id -> callable.

    Raises:
        RuntimeError: If a key is bound twice, an action has no handler, or a
            handler is never bound. Any of those is a wiring bug that would
            otherwise show up as a key that silently does nothing.
    """
    problems = []

    seen = {}
    for key, action, _ in BINDINGS:
        if key in seen:
            problems.append(f"key {key!r} is bound twice: {seen[key]!r} and {action!r}")
        seen[key] = action

    bound = {action for _, action, _ in BINDINGS}
    for action in sorted(bound - set(handlers)):
        problems.append(f"action {action!r} is bound to a key but has no handler")
    for action in sorted(set(handlers) - bound):
        problems.append(f"handler {action!r} exists but no key is bound to it")

    for name, actions in CATEGORIES.items():
        for action in actions:
            if action not in bound:
                problems.append(f"category {name!r} lists unknown action {action!r}")

    if problems:
        raise RuntimeError("keymap is inconsistent:\n  " + "\n  ".join(problems))


def format_controls(extra_notes=None):
    """Render the keymap as printable help.

    Args:
        extra_notes (list[str] or None): Lines appended after the table.

    Returns:
        str
    """
    lines = ["=" * 60, "INTERACTIVE SCENE EDITOR - KEYBOARD CONTROLS", "=" * 60]
    for category, actions in CATEGORIES.items():
        lines.append(f"\n{category}:")
        for action in dict.fromkeys(actions):          # de-dup, keep order
            keys = " / ".join(key_label(k) for k in keys_for(action))
            lines.append(f"  {keys:<14s} - {help_for(action)}")
    lines.append("=" * 60)
    for note in extra_notes or []:
        lines.append(note)
    return "\n".join(lines)
