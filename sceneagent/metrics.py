"""Reconstruction and evaluation statistics named by SceneAgent.

Chamfer distance and F1 at 1 cm follow equations (4) and (5) in section 4.1.
Points are in meters. The paper's threshold tau is 1 cm, so the default tau
is 0.01 m. F1 is 0 when precision + recall is 0, and a missing object scores 0
when a scene is summarized the way GraspClutter6D is summarized in that section.

The Wilson interval is the 95% interval the project website prints on each
place-success cell (for example 1 of 10 trials is reported as 1.8% to 40.4%).
The paper's figure 14 reports the same cells as successes over ten trials.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

TAU_1CM_M = 0.01
WILSON_Z_95 = 1.959963984540054


def wilson_interval(successes, trials, z=WILSON_Z_95):
    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must lie in [0, trials]")
    phat = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (phat + z2 / (2.0 * trials)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / trials + z2 / (4.0 * trials * trials)) / denom
    return center - half, center + half


def _as_points(points):
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    return arr


def _min_distances(source, target):
    delta = source[:, None, :] - target[None, :, :]
    return np.linalg.norm(delta, axis=2).min(axis=1)


def symmetric_chamfer(points_x, points_y):
    """One-half the sum of the two directed mean nearest-point distances."""
    x = _as_points(points_x)
    y = _as_points(points_y)
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Chamfer distance needs points on both clouds")
    return 0.5 * (float(_min_distances(x, y).mean()) + float(_min_distances(y, x).mean()))


def f1_at_threshold(points_x, points_y, tau=TAU_1CM_M):
    x = _as_points(points_x)
    y = _as_points(points_y)
    if len(x) == 0 or len(y) == 0:
        return 0.0
    precision = float((_min_distances(x, y) <= tau).mean())
    recall = float((_min_distances(y, x) <= tau).mean())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _hungarian(cost):
    """Minimum-cost assignment. Exact enumeration; scenes here are small."""
    cost = np.asarray(cost, dtype=float)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []
    if n_rows > 8 or n_cols > 8:
        raise ValueError("phase-1 matcher supports at most 8 objects on a side")
    best = None
    best_cost = None
    columns = range(n_cols)
    for choice in itertools.permutations(columns, n_rows):
        total = float(sum(cost[i, choice[i]] for i in range(n_rows)))
        if best_cost is None or total < best_cost:
            best_cost = total
            best = [(i, choice[i]) for i in range(n_rows)]
    return best


def score_reconstruction(delivered, reference, tau=TAU_1CM_M):
    """Match delivered clouds to reference clouds by Chamfer cost.

    ``delivered`` and ``reference`` map a name to an (n, 3) cloud in meters.
    F1 is averaged over every reference object, with unmatched objects at 0.
    Chamfer is averaged over matched objects only, matching section 4.1's
    GraspClutter6D summary. This function does not load that dataset.
    """
    ref_names = list(reference)
    del_names = list(delivered)
    if not ref_names:
        raise ValueError("reference is empty")
    f1_by_ref = {name: 0.0 for name in ref_names}
    if not del_names:
        return {
            "f1_mean": 0.0,
            "chamfer_mean_matched_m": None,
            "matches": [],
            "f1_by_reference": f1_by_ref,
            "tau_m": tau,
        }
    cost = np.zeros((len(ref_names), len(del_names)), dtype=float)
    for i, ref_name in enumerate(ref_names):
        for j, del_name in enumerate(del_names):
            cost[i, j] = symmetric_chamfer(reference[ref_name], delivered[del_name])
    pairs = _hungarian(cost)
    matches = []
    chamfers = []
    for i, j in pairs:
        ref_name = ref_names[i]
        del_name = del_names[j]
        chamfer = float(cost[i, j])
        f1 = f1_at_threshold(delivered[del_name], reference[ref_name], tau=tau)
        f1_by_ref[ref_name] = f1
        chamfers.append(chamfer)
        matches.append({
            "reference": ref_name,
            "delivered": del_name,
            "chamfer_m": chamfer,
            "f1": f1,
        })
    return {
        "f1_mean": float(np.mean([f1_by_ref[name] for name in ref_names])),
        "chamfer_mean_matched_m": float(np.mean(chamfers)) if chamfers else None,
        "matches": matches,
        "f1_by_reference": f1_by_ref,
        "tau_m": tau,
    }
