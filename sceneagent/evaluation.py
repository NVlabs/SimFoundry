"""Scripted place-success grid (SceneAgent sections 3.10 and figure 14).

The paper's expert uses simulator state: approach, descend, grasp, lift, carry,
lower, release, retreat. A success check asks whether the object ends inside
the target. The website's experiment view crosses object sisters with nine
lighting setups and reports successes over ten trials plus a 95% Wilson interval.

Phase 1 runs that check geometrically. It does not roll out pi0.5 or any other
learned policy, and it does not reproduce the website's trial counts. Lighting
names are the nine columns on that page. The geometric check does not depend
on them, and each cell says so.

A trial succeeds when the sister passed the mesh gates and the scripted start
pose is inside the reach disc around the task anchor. Starts outside that disc
are unsuccessful episodes, which section 3.10 says are discarded from the
training set. Here they remain in the evaluation count so the cell can report
k of n.
"""

from __future__ import annotations

import numpy as np

from sceneagent.metrics import wilson_interval

# Column labels from the project website's "Policy performance across digital
# sisters" grid. They are conditions, not a claim that those lights were rendered.
LIGHTING_SETUPS = (
    "Neutral",
    "Strong warm",
    "Strong cool",
    "Very dim",
    "Very bright",
    "Hard left",
    "Hard right",
    "Backlit",
    "Overhead spot",
)


def trial_starts(n_trials, half_extent_m, seed):
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-half_extent_m, half_extent_m, size=(n_trials, 2))
    return offsets


def score_trials(starts_xy, reach_radius_m, sister_accepted):
    starts = np.asarray(starts_xy, dtype=float)
    reach = float(reach_radius_m)
    rows = []
    successes = 0
    for index, (x, y) in enumerate(starts):
        distance = float(np.hypot(x, y))
        reached = distance <= reach
        success = bool(sister_accepted and reached)
        successes += int(success)
        rows.append({
            "trial": index,
            "start_xy_m": [float(x), float(y)],
            "distance_m": distance,
            "reached": reached,
            "success": success,
        })
    return successes, rows


def single_rollout_place(final_qpos, bowl_xyz, bowl_half_extent_m, nstep):
    """One MuJoCo rollout, not the ten-trial rate in figure 14.

    The box is inside the bowl when its free-joint position lies in the bowl
    geom's axis-aligned box. That containment test is this runner's. The paper
    does not publish it. One rollout cannot fill the nine lighting columns, so
    the grid stays partial and those columns are not given a success rate.
    The Wilson interval is for this single trial only.
    """
    qpos = [float(value) for value in final_qpos]
    center = qpos[:3]
    half = float(bowl_half_extent_m)
    bowl = [float(value) for value in bowl_xyz]
    inside = all(abs(center[index] - bowl[index]) <= half for index in range(3))
    low, high = wilson_interval(1 if inside else 0, 1)
    return {
        "section": "3.8",
        "figure": 14,
        "status": "partial",
        "reason": "One MuJoCo rollout cannot support the ten-trial place rate in figure 14.",
        "policy": "mujoco_single_rollout",
        "not_a_learned_policy": True,
        "not_a_ten_trial_rate": True,
        "rate_reported": False,
        "n_rollouts": 1,
        "nstep": int(nstep),
        "lighting": list(LIGHTING_SETUPS),
        "lighting_simulated": False,
        "cells_ran": False,
        "trial": {
            "success": inside,
            "containment": "box free-joint position inside the bowl geom AABB",
            "containment_published": False,
            "final_qpos": qpos,
            "box_center_m": center,
            "bowl_xyz_m": bowl,
            "bowl_half_extent_m": half,
        },
        "wilson95_one_trial": [low, high],
        "wilson_trials": 1,
    }


def sister_grid(sisters, starts_xy, reach_radius_m, lighting=LIGHTING_SETUPS):
    """Object-sister rows by lighting columns.

    Every lighting column repeats the geometric result. ``lighting_simulated``
    is false so a reader cannot mistake the repeat for a rendered illumination study.
    """
    n_trials = len(starts_xy)
    rows = []
    for sister in sisters:
        successes, trials = score_trials(starts_xy, reach_radius_m, sister["accepted"])
        low, high = wilson_interval(successes, n_trials)
        cell = {
            "successes": successes,
            "trials": n_trials,
            "rate": successes / n_trials,
            "wilson95": [low, high],
            "lighting_simulated": False,
            "policy": "scripted_geometric_place",
            "sister_accepted": sister["accepted"],
        }
        rows.append({
            "sister": sister["name"],
            "kind": sister["kind"],
            "accepted_for_training": sister["accepted"],
            "reasons": sister["reasons"],
            "cells": {name: dict(cell) for name in lighting},
            "trials": trials,
        })
    return {
        "task": "place_inside",
        "policy": "scripted_geometric_place",
        "not_a_learned_policy": True,
        "lighting": list(lighting),
        "lighting_simulated": False,
        "rows": rows,
    }
