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

# The acceptance rule is implemented. The trimmed similarity ICP that produces
# the four residuals is not.
GATE_SKILLS = (
    (
        "SCENEAGENT-ALIGN-SCENE",
        "Phase 1 accepts or rejects caller-supplied residuals. It does not run trimmed similarity ICP.",
    ),
)

LATER_SKILLS = (
    "SCENEAGENT-VIDEO-TO-4D",
    "SCENEAGENT-ORIENT-BACKGROUND",
    "SCENEAGENT-ARTICULATE-NATIVE-ASSETS",
    "SCENEAGENT-SIMULATE-FRACTURE",
    "SCENEAGENT-TIMESERIES-4D-SCENE",
    "SCENEAGENT-INSPECT-ISAAC",
    "SCENEAGENT-POLICY-AUTORESEARCH",
    "SCENEAGENT-ADAPT-VLA-POLICIES",
)


def plan_steps():
    steps = []
    for skill in PHASE1_SKILLS:
        steps.append({"skill": skill, "status": "implemented_phase1"})
    for skill, reason in GATE_SKILLS:
        steps.append({"skill": skill, "status": "gate_only", "reason": reason})
    for skill in LATER_SKILLS:
        steps.append({
            "skill": skill,
            "status": "not_implemented",
            "reason": "Requires reconstruction, a simulator, or a learned policy that phase 1 does not run.",
        })
    return steps
