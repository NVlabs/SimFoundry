"""Skill identifiers from SceneAgent figure 7.

Phase 1 runs the skills whose work is a file contract, a vote, a mesh check,
or a scripted success count. The remaining identifiers are listed so later
phases have a stable name. Their status is ``not_implemented``.
"""

PHASE1_SKILLS = (
    "SCENEAGENT-RUN-PIPELINE",
    "SCENEAGENT-INGEST-DATA",
    "SCENEAGENT-RUN-STEP",
    "SCENEAGENT-OBJECT-SISTERS",
    "SCENEAGENT-SCENE-SISTERS",
    "SCENEAGENT-TASK-SISTERS",
    "SCENEAGENT-GENERATE-SISTERS",
    "SCENEAGENT-PACKAGE-DATA-PROCESSED",
    "SCENEAGENT-POLICY-EVAL",
)

# Trimmed similarity ICP writes measured residuals. The four acceptance
# numbers are still unpublished, so the gate does not accept a pose from
# those residuals alone.
GATE_SKILLS = (
    (
        "SCENEAGENT-ALIGN-SCENE",
        "Trimmed similarity ICP can write measured residuals. The pose stays undecided when the unpublished thresholds are absent.",
    ),
)

# Isaac opened phase1_place.usda on Quebec. That is a stage fact. It is not
# a fix report written by a coding model, so the status is not "implemented"
# in the sense of a model review.
STAGE_OPENED_SKILLS = (
    (
        "SCENEAGENT-INSPECT-ISAAC",
        "stage_opened",
        "Isaac opened phase1_place.usda (opened=True; prims /World, /World/box, /World/bowl). No coding model wrote a fix report.",
        False,
    ),
)

LATER_SKILLS = (
    "SCENEAGENT-VIDEO-TO-4D",
    "SCENEAGENT-ORIENT-BACKGROUND",
    "SCENEAGENT-ARTICULATE-NATIVE-ASSETS",
    "SCENEAGENT-SIMULATE-FRACTURE",
    "SCENEAGENT-TIMESERIES-4D-SCENE",
    "SCENEAGENT-POLICY-AUTORESEARCH",
    "SCENEAGENT-ADAPT-VLA-POLICIES",
)


def plan_steps():
    steps = []
    for skill in PHASE1_SKILLS:
        steps.append({"skill": skill, "status": "implemented_phase1"})
    for skill, reason in GATE_SKILLS:
        steps.append({"skill": skill, "status": "gate_only", "reason": reason})
    for skill, status, reason, fix_report in STAGE_OPENED_SKILLS:
        steps.append({
            "skill": skill,
            "status": status,
            "reason": reason,
            "fix_report_from_model": fix_report,
        })
    for skill in LATER_SKILLS:
        steps.append({
            "skill": skill,
            "status": "not_implemented",
            "reason": "Requires reconstruction, a simulator, or a learned policy that phase 1 does not run.",
        })
    return steps
