"""Section 3.3 trimmed similarity ICP.

Umeyama (1991) solves the scale, rotation, and translation between paired
points. Each iteration keeps the closest fraction of those pairs. That trim
is the overlap used by trimmed ICP. ``TRIM_KEEP`` is this implementation's
overlap, not a number published by SceneAgent.

The section 3.1 gate still needs the unpublished acceptance thresholds.
A residual file records the measured distances. It does not accept the pose.
"""

from __future__ import annotations

import numpy as np

# Closest-pair fraction retained each iteration. Not a SceneAgent threshold.
TRIM_KEEP = 0.7


def umeyama(source, target):
    """Similarity that maps source points onto paired target points.

    Returns scale, a 3x3 rotation, and a translation. The transform is
    ``scale * rotation @ point + translation``.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must be matching Nx3 arrays")
    if len(source) < 3:
        raise ValueError("Umeyama needs at least 3 pairs")
    count = float(len(source))
    mean_source = source.mean(axis=0)
    mean_target = target.mean(axis=0)
    centered_source = source - mean_source
    centered_target = target - mean_target
    covariance = centered_target.T @ centered_source / count
    left, singular, right = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        sign[2, 2] = -1.0
    rotation = left @ sign @ right
    variance = float((centered_source ** 2).sum() / count)
    if variance <= 0:
        raise ValueError("source points are coincident")
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = mean_target - scale * (rotation @ mean_source)
    return scale, rotation, translation


def apply_similarity(points, scale, rotation, translation):
    points = np.asarray(points, dtype=np.float64)
    transformed = scale * (np.asarray(rotation, dtype=np.float64) @ points.T)
    return transformed.T + np.asarray(translation, dtype=np.float64)


def _rodrigues(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return cosine * np.eye(3) + sine * cross + (1.0 - cosine) * np.outer(axis, axis)


def rotation_seeds():
    """Export-frame identity plus quarter-turns about the coordinate axes.

    The stored SAM 3D rotation vector is absent, so these seeds are the
    search. They are not acceptance thresholds.
    """
    seeds = [np.eye(3)]
    for axis in np.eye(3):
        for turn in (1, 2, 3):
            seeds.append(_rodrigues(axis, turn * np.pi / 2.0))
    return seeds


def _nearest(query, target, tree):
    """Nearest neighbors. A KD-tree is used when scipy is installed."""
    query = np.asarray(query, dtype=np.float64)
    if tree is not None:
        distance, index = tree.query(query, k=1)
        return np.asarray(distance, dtype=np.float64), np.asarray(index, dtype=np.int64)
    target = np.asarray(target, dtype=np.float64)
    nearest_distance = np.empty(len(query), dtype=np.float64)
    nearest_index = np.empty(len(query), dtype=np.int64)
    chunk = 256
    for start in range(0, len(query), chunk):
        block = query[start:start + chunk]
        squared = ((block[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
        index = squared.argmin(axis=1)
        nearest_index[start:start + len(block)] = index
        nearest_distance[start:start + len(block)] = np.sqrt(squared[np.arange(len(block)), index])
    return nearest_distance, nearest_index


def initial_similarity(source, target):
    """Place the exported mesh on the target centroid. Rotation starts as identity.

    The SAM 3D export frame is the source. Scale matches the two clouds' RMS
    sizes so the first correspondence is not a unit mesh against a scene-sized
    splat. That scale is a measured ratio, not an acceptance threshold.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_rms = float(np.sqrt(((source - source.mean(axis=0)) ** 2).sum(axis=1).mean()))
    target_rms = float(np.sqrt(((target - target.mean(axis=0)) ** 2).sum(axis=1).mean()))
    if source_rms <= 0:
        raise ValueError("source points are coincident")
    scale = target_rms / source_rms
    rotation = np.eye(3)
    translation = target.mean(axis=0) - scale * source.mean(axis=0)
    return scale, rotation, translation


def trimmed_similarity_icp(source, target, trim_keep=TRIM_KEEP, iterations=30):
    """Refine a similarity from the source frame onto the target points.

    ``trim_keep`` is the fraction of nearest pairs retained. The paper does
    not publish this overlap.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not 0 < float(trim_keep) <= 1:
        raise ValueError("trim_keep must be in (0, 1]")
    tree = None
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(target)
    except Exception:
        tree = None
    base_scale, _, _ = initial_similarity(source, target)
    keep_count = max(3, int(round(float(trim_keep) * len(source))))
    best = None
    for seed_index, seed in enumerate(rotation_seeds()):
        scale = base_scale
        rotation = np.asarray(seed, dtype=np.float64)
        translation = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
        for _ in range(int(iterations)):
            transformed = apply_similarity(source, scale, rotation, translation)
            distance, index = _nearest(transformed, target, tree)
            chosen = np.argsort(distance)[:keep_count]
            scale, rotation, translation = umeyama(source[chosen], target[index[chosen]])
        transformed = apply_similarity(source, scale, rotation, translation)
        distance, index = _nearest(transformed, target, tree)
        chosen = np.argsort(distance)[:keep_count]
        score = float(np.median(distance[chosen]))
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "seed_index": seed_index,
                "scale": float(scale),
                "rotation": np.asarray(rotation, dtype=np.float64),
                "translation": np.asarray(translation, dtype=np.float64),
                "distance": distance,
                "inlier_index": chosen,
                "target_index": index,
                "aligned": transformed,
            }
    inlier_distance = best["distance"][best["inlier_index"]]
    return {
        "scale": best["scale"],
        "rotation": best["rotation"],
        "translation": best["translation"],
        "rotation_seed_index": best["seed_index"],
        "trim_keep": float(trim_keep),
        "iterations": int(iterations),
        "inlier_count": int(keep_count),
        "source_count": int(len(source)),
        "target_count": int(len(target)),
        "distance": best["distance"],
        "inlier_index": best["inlier_index"],
        "target_index": best["target_index"],
        "inlier_distance": inlier_distance,
        "aligned": best["aligned"],
    }


def measure_residuals(result, target):
    """Measured residuals. No pass/fail cutoff is applied.

    ``normalized_median`` divides the median nearest distance by the target
    cloud's RMS size. ``coverage`` is the fraction of target points that are
    the neighbor of at least one trimmed inlier. ``silhouette`` is filled by
    the caller when a camera is available; this function leaves it null.
    """
    target = np.asarray(target, dtype=np.float64)
    distance = np.asarray(result["distance"], dtype=np.float64)
    inlier_distance = np.asarray(result["inlier_distance"], dtype=np.float64)
    target_rms = float(np.sqrt(((target - target.mean(axis=0)) ** 2).sum(axis=1).mean()))
    if target_rms <= 0:
        raise ValueError("target points are coincident")
    median = float(np.median(distance))
    p90 = float(np.percentile(distance, 90))
    covered = len(set(int(index) for index in result["target_index"][result["inlier_index"]]))
    return {
        "normalized_median": median / target_rms,
        "p90": p90,
        "median": median,
        "p90_inlier": float(np.percentile(inlier_distance, 90)),
        "median_inlier": float(np.median(inlier_distance)),
        "normalizer": target_rms,
        "normalizer_definition": "RMS distance of the target points from their centroid",
        "coverage": covered / float(len(target)),
        "coverage_definition": "fraction of target points chosen by at least one trimmed inlier",
        "silhouette": None,
        "thresholds_published": False,
    }
