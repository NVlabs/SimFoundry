"""Alignment acceptance interface from SceneAgent section 3.1.

A candidate is accepted only when all four residuals pass fixed thresholds:
normalized median distance, 90th-percentile distance, coverage, and silhouette
agreement. The paper does not publish the four numbers. Phase 1 therefore
accepts a candidate only when the caller supplies a threshold for every
residual. Missing thresholds leave the candidate undecided, which is not an
acceptance.

Distance residuals pass at or below ``max``. Coverage and silhouette pass at
or above ``min``. That inequality direction is the usual reading of those
quantities; the paper does not write the inequalities out.
"""

RESIDUAL_NAMES = ("normalized_median", "p90", "coverage", "silhouette")
DISTANCE_RESIDUALS = ("normalized_median", "p90")
AGREEMENT_RESIDUALS = ("coverage", "silhouette")


def accept_alignment(residuals, thresholds):
    residuals = dict(residuals or {})
    thresholds = dict(thresholds or {})
    missing = [name for name in RESIDUAL_NAMES if name not in residuals or name not in thresholds]
    if missing:
        return {
            "decision": "undecided",
            "accepted": False,
            "missing": missing,
            "reason": "SceneAgent states that fixed thresholds exist and does not publish them.",
        }
    failed = []
    for name in DISTANCE_RESIDUALS:
        limit = thresholds[name]
        if "max" not in limit:
            return {
                "decision": "undecided",
                "accepted": False,
                "missing": [name],
                "reason": f"{name} needs a max threshold",
            }
        if float(residuals[name]) > float(limit["max"]):
            failed.append(name)
    for name in AGREEMENT_RESIDUALS:
        limit = thresholds[name]
        if "min" not in limit:
            return {
                "decision": "undecided",
                "accepted": False,
                "missing": [name],
                "reason": f"{name} needs a min threshold",
            }
        if float(residuals[name]) < float(limit["min"]):
            failed.append(name)
    return {
        "decision": "reject" if failed else "accept",
        "accepted": not failed,
        "failed": failed,
        "residuals": {name: float(residuals[name]) for name in RESIDUAL_NAMES},
    }
