"""Behavior added so sections 3.1–3.4 match the paper where the code can."""

import json
from pathlib import Path

from sceneagent.agent_loop import (
    ALLOWED_STEPS,
    EXTERNAL_HARNESSES,
    STEP_SKILL,
    harness_status,
    record_and_repair,
    run_agent_loop,
)
from sceneagent.alignment import (
    FRUITS_ICP_RESIDUAL,
    OURS_COVERAGE_MIN,
    OURS_GATE_NOTE,
    OURS_NORMALIZED_MEDIAN_MAX,
    OURS_P90_MAX,
    OURS_SILHOUETTE_MIN,
    accept_alignment,
    ours_iteration1_thresholds,
)
from sceneagent.fruits_agent import PLAN_PROMPT, order_plan, plan_with_client
from sceneagent.infill import complete_background, lama_inpaint, plane_fill_from_association, plane_infill, write_rgb_png
from sceneagent.ingest_assets import (
    OURS_COLOR_HEURISTIC,
    color_calibration_directory,
    color_calibration_report,
    ingest_asset,
    read_still_image_sequence,
)
from sceneagent.reviewer_views import REQUIRED_VIEWS, write_view_manifest
from sceneagent.section34 import (
    DIFFUSERS_039_FORWARD,
    REPO_POS_EMBED_CALL,
    articulate_row,
    choose_qwen_embed_rope_call,
    cubepart_status,
    equation2_vote,
    inspect_parts,
    parse_material_properties,
    partsam_summary_prompt,
    request_audit,
    request_material_properties,
)
from sceneagent.skills import plan_steps
from sceneagent.sam2_user import LABEL as SAM2_LABEL
from sceneagent.sam2_user import run_sam2, sam2_status
from sceneagent.website_step02 import LABEL as GROUNDING_LABEL
from sceneagent.website_step02 import groundingdino_status, run_groundingdino


class _Client:
    def __init__(self, text):
        self.text = text
        self.prompts = []

    def generate_flash(self, prompt, image_bytes=None):
        self.prompts.append(prompt)
        return {
            "text": self.text,
            "model_id": "fake",
            "http_status": 200,
            "used_paper_model": False,
            "paper_model_id": "gemini-2.5-flash",
        }


def _ppm(path, rgb, size=2):
    header = f"P6\n{size} {size}\n255\n".encode("ascii")
    path.write_bytes(header + bytes(rgb) * (size * size))


def test_ledger_skill_follows_figure_7_and_harnesses_stay_uncalled(tmp_path):
    summary = run_agent_loop(
        tmp_path / "loop",
        {
            "isaac_inspector": lambda _ctx: {"ok": True, "artifact": "stage.txt", "reason": "", "detail": {}},
            "icp": lambda _ctx: {"ok": False, "artifact": None, "reason": "no pose", "detail": {}},
            "articulate": lambda _ctx: {"ok": False, "artifact": None, "reason": "no urdf", "detail": {}},
        },
        ["isaac_inspector", "icp", "articulate"],
    )
    skills = {entry["item"]: entry["skill"] for entry in summary["entries"]}
    assert skills["isaac_inspector"] == "SCENEAGENT-INSPECT-ISAAC"
    assert skills["icp"] == "SCENEAGENT-ALIGN-SCENE"
    assert skills["articulate"] == "SCENEAGENT-ARTICULATE-NATIVE-ASSETS"
    assert STEP_SKILL["material_vote"] == "SCENEAGENT-RUN-STEP"
    assert len(set(skills.values())) == 3
    status = harness_status()
    assert status["called"] == []
    assert status["status"] == "still_open"
    assert list(EXTERNAL_HARNESSES) == [
        "Codex", "Claude Code", "Antigravity", "Muse", "Stirrup", "Grok CLI",
    ]


def test_plan_asks_from_input_state_and_falls_back_when_empty():
    for step_id in ALLOWED_STEPS:
        assert step_id not in PLAN_PROMPT
    assert '{"steps"' not in PLAN_PROMPT
    for name in EXTERNAL_HARNESSES:
        assert name not in PLAN_PROMPT
    empty = plan_with_client(_Client(""))
    assert empty["used_fallback"] is True
    assert empty["fallback"] == "default_plan"
    assert empty["steps"] == order_plan([])
    ordered = plan_with_client(_Client('{"steps": ["sam3d", "not-a-block", "icp"]}'))
    assert ordered["used_fallback"] is False
    assert ordered["steps"] == ["sam3d", "icp"]
    assert ordered["steps"][0] != "material_vote"


def test_repair_hook_reruns_only_a_registered_tool_and_invents_no_pose(tmp_path):
    rejected = record_and_repair(
        tmp_path / "bare",
        "icp",
        {"ok": False, "artifact": None, "reason": "residual rejected", "detail": {}},
        tools={},
    )
    assert rejected["calls"] == []
    assert rejected["entries"][0]["decision"] == "reject"
    assert rejected["entries"][0]["skill"] == "SCENEAGENT-ALIGN-SCENE"
    manifest = json.loads((tmp_path / "bare" / "manifests" / "icp-1.json").read_text(encoding="utf-8"))
    assert "pose" not in manifest["detail"]

    def repair(_context):
        return {"ok": False, "artifact": None, "reason": "still no pose", "detail": {}}

    rerun = record_and_repair(
        tmp_path / "hook",
        "icp",
        {"ok": False, "artifact": None, "reason": "residual rejected", "detail": {}},
        tools={"icp_repair": repair},
    )
    assert rerun["calls"] == ["icp_repair"]
    assert [entry["attempt"] for entry in rerun["entries"]] == [1, 2]
    second = json.loads((tmp_path / "hook" / "manifests" / "icp-2.json").read_text(encoding="utf-8"))
    assert second["detail"]["pose"] is None


def test_reviewer_manifest_names_the_paper_views_and_does_not_draw_them(tmp_path):
    body = write_view_manifest(tmp_path, "apple")
    assert [name for name, _group in REQUIRED_VIEWS] == body["required"]
    groups = {}
    for view in body["views"]:
        groups[view["group"]] = groups.get(view["group"], 0) + 1
        assert view["present"] is False
        assert not Path(view["path"]).exists()
    assert groups == {"registration_overlay": 1, "axis": 6, "oblique": 2, "assembled": 2}
    assert body["status"] == "missing"
    assert body["images_generated"] is False
    assert body["missing"] == body["required"]


def test_alignment_stays_undecided_without_thresholds():
    residuals = {
        "normalized_median": 0.033103985722272695,
        "p90": 0.21563629180764,
        "coverage": 0.14616666666666667,
        "silhouette": None,
    }
    undecided = accept_alignment(residuals, None)
    assert undecided["decision"] == "undecided"
    assert undecided["accepted"] is False
    assert "silhouette" in undecided["missing"]


def test_inspect_isaac_skill_matches_the_opened_stage():
    step = next(item for item in plan_steps() if item["skill"] == "SCENEAGENT-INSPECT-ISAAC")
    assert step["status"] == "stage_opened"
    assert step["status"] != "not_implemented"
    assert step["fix_report_from_model"] is False
    assert "No coding model wrote a fix report." in step["reason"]


def test_still_images_are_a_sequence_and_existing_assets_need_a_file(tmp_path):
    folder = tmp_path / "stills"
    folder.mkdir()
    (folder / "0001.jpg").write_bytes(b"\xff")
    (folder / "0002.png").write_bytes(b"\x89")
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    sequence = read_still_image_sequence(folder)
    assert sequence["count"] == 2
    assert sequence["paths"] == [str(folder / "0001.jpg"), str(folder / "0002.png")]
    assert sequence["colmap_ran"] is False
    assert sequence["reconstruction"] == "not_run"
    assert sequence["paper_site_disagree"] is False
    ply = tmp_path / "scan.ply"
    ply.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    accepted = ingest_asset(ply, "existing_3dgs", out_dir=tmp_path / "ledger")
    assert accepted["decision"] == "accept"
    assert accepted["skill"] == "SCENEAGENT-INGEST-DATA"
    assert accepted["downloaded"] is False
    ledger = json.loads((tmp_path / "ledger" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["entries"][0]["decision"] == "accept"
    missing = ingest_asset(tmp_path / "nope.ply", "photogrammetry")
    assert missing["decision"] == "reject"
    remote = ingest_asset("https://example.test/marble.ply", "generated_3dgs", source_name="World Labs Marble")
    assert remote["decision"] == "reject"
    assert remote["downloaded"] is False
    assert remote["generators_contacted"] == []
    obj = tmp_path / "cap.obj"
    obj.write_text("o cap\n", encoding="utf-8")
    assert ingest_asset(obj, "photogrammetry")["decision"] == "accept"
    las = tmp_path / "room.las"
    las.write_bytes(b"LASF")
    assert ingest_asset(las, "lidar")["decision"] == "accept"


def test_color_heuristic_is_labeled_ours(tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    _ppm(folder / "a.ppm", (10, 10, 10))
    _ppm(folder / "b.ppm", (12, 11, 10))
    _ppm(folder / "c.ppm", (80, 10, 10))
    report = color_calibration_directory(folder)
    assert report["paper_method"] is False
    assert report["heuristic"] == OURS_COLOR_HEURISTIC
    assert report["limit_l1"] == 30.0
    assert report["colmap_ran"] is False
    assert "ours" in report["label"]
    flagged = {Path(path).name for path in report["flagged_paths"]}
    assert flagged == {"c.ppm"}
    direct = color_calibration_report([
        ("a", (10, 10, 10)),
        ("b", (12, 11, 10)),
        ("c", (80, 10, 10)),
    ])
    assert direct["flagged_paths"] == ["c"]


def test_groundingdino_is_website_step_02_and_not_run():
    status = groundingdino_status()
    assert status["label"] == "website step 02"
    assert status["not_section_3_3"] is True
    assert status["status"] == "not_run"
    assert status["weights_loaded"] is False


def test_plane_infill_refuses_to_emit_without_a_plane_render(tmp_path):
    destination = tmp_path / "infill.png"
    refused = plane_infill(tmp_path / "missing.png", destination, use_lama=True)
    assert refused["emitted"] is False
    assert refused["artifact"] is None
    assert refused["status"] == "open"
    assert refused["reason"] == "no plane render exists"
    assert not destination.exists()
    present = tmp_path / "plane.png"
    present.write_bytes(b"png")
    held = plane_infill(present, destination, use_lama=True)
    assert held["emitted"] is False
    assert held["lama"] == "not_started"
    assert not destination.exists()


def test_equation2_keeps_a_missing_prior_out_of_the_mlp_claim():
    voted = equation2_vote("widget", vlm_class="plastic", physics_prior=None, vlm_confidence=0.5)
    assert voted["physics_prior"] == "absent"
    assert voted["claimed_mlp"] is False
    assert voted["class_is_mlp"] is False
    assert voted["unknown_scale"] == 0.4
    assert voted["material_class"] == "plastic"
    assert voted["scores"]["unknown"] == 0.4
    prior = equation2_vote(
        "widget",
        vlm_class="wood",
        physics_prior={"class": "metal", "confidence": 2.0},
        vlm_confidence=1.0,
    )
    assert prior["physics_prior"] == "present"
    assert prior["claimed_mlp"] is False
    assert prior["material_class"] == "metal"
    audit = request_audit("widget", voted["material_class"], voted["sources"], api_key="super-secret-value", live=False)
    assert audit["called"] is False
    assert audit["model_id"] is None
    assert audit["decision"] == "pending"
    assert "super-secret-value" not in audit["prompt"]
    assert "gemini-2.5-flash" in audit["prompt"]
    assert audit["historical"][0]["http_status"] == 404
    assert audit["historical"][1]["material_class"] == "wood"


def test_part_prompt_stays_pending_and_does_not_call_cubepart():
    pending = inspect_parts(["red cap", "blue barrel", "white tip"])
    assert pending["decision"] == "pending"
    assert pending["call_cubepart"] is False
    assert pending["inspected"] is False
    assert "hidden geometry is missing" in pending["prompt"]
    assert "red cap" in pending["prompt"]
    needed = inspect_parts(["cap"], model_reply="yes, completion is needed")
    assert needed["call_cubepart"] is True
    skipped = inspect_parts(["cap"], model_reply="no")
    assert skipped["call_cubepart"] is False


def test_ours_alignment_gate_rejects_fruits_and_accepts_a_passing_residual():
    assert OURS_GATE_NOTE == (
        "SceneAgent did not publish these cutoffs. These are our iteration-1 gates."
    )
    gate = ours_iteration1_thresholds()
    assert gate["paper_published"] is False
    assert gate["label"] == "ours_iteration_1"
    assert gate["normalized_median"]["max"] == OURS_NORMALIZED_MEDIAN_MAX == 0.05
    assert gate["coverage"]["min"] == OURS_COVERAGE_MIN == 0.5
    assert gate["p90"]["max"] == OURS_P90_MAX == 0.25
    assert gate["silhouette"]["min"] == OURS_SILHOUETTE_MIN == 0.5
    fruits = accept_alignment(FRUITS_ICP_RESIDUAL, gate)
    assert fruits["decision"] == "reject"
    assert fruits["accepted"] is False
    assert fruits["gate"] == "ours_iteration_1"
    assert fruits["paper_published"] is False
    assert set(fruits["failed"]) == {"coverage", "silhouette"}
    assert "normalized_median" not in fruits["failed"]
    assert "p90" not in fruits["failed"]
    passing = accept_alignment(
        {
            "normalized_median": 0.05,
            "p90": 0.25,
            "coverage": 0.5,
            "silhouette": 0.5,
        },
        "ours_iteration_1",
    )
    assert passing["decision"] == "accept"
    assert passing["accepted"] is True
    assert passing["failed"] == []


def test_groundingdino_and_sam2_labels_do_not_close_section_3_3(tmp_path):
    assert GROUNDING_LABEL == "website step 02"
    idle = groundingdino_status()
    assert idle["label"] == "website step 02"
    assert idle["not_section_3_3"] is True
    assert idle["closes_section_3_3"] is False
    assert idle["status"] == "not_run"
    assert idle["weights_loaded"] is False
    busy = groundingdino_status(gpu_busy=True)
    assert busy["status"] == "not_run_gpu_busy"
    assert busy["closes_section_3_3"] is False
    assert "section 3.3" not in busy["label"]
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"\xff\xd8")
    ran = run_groundingdino(image, "fruit", gpu_busy=True)
    assert ran["status"] == "not_run_gpu_busy"
    assert ran["label"] == "website step 02"
    assert SAM2_LABEL == "user-requested SAM 2, not the paper's SAM vit_h"
    sam = sam2_status()
    assert sam["label"] == SAM2_LABEL
    assert sam["paper_sam"] == "vit_h"
    assert sam["not_section_3_3"] is True
    assert sam["closes_section_3_3"] is False
    assert sam["not_paper_sam_vit_h"] is True
    assert sam["status"] == "not_run"
    held = run_sam2(image, box=[0, 0, 2, 2], gpu_busy=True)
    assert held["status"] == "not_run_gpu_busy"
    assert held["tool"] == "SAM 2"
    assert "vit_h" not in held["tool"]


def test_lama_refuses_without_forging_an_image(tmp_path):
    missing_out = tmp_path / "forged.png"
    missing = lama_inpaint(tmp_path / "absent.png", missing_out)
    assert missing["emitted"] is False
    assert missing["forged"] is False
    assert missing["status"] == "no_image"
    assert missing["artifact"] is None
    assert not missing_out.exists()
    hole = write_rgb_png(
        tmp_path / "hole.png",
        [
            [0, 0, 0, 255, 255, 255],
            [255, 255, 255, 200, 180, 160],
        ],
    )
    painted = lama_inpaint(hole, tmp_path / "inpaint.png")
    assert painted["forged"] is False
    if painted["status"] == "weights_absent":
        assert painted["emitted"] is False
        assert painted["weights_installed"] is False
        assert painted["artifact"] is None
        assert "not installed" in painted["reason"]
        assert not (tmp_path / "inpaint.png").exists()
    else:
        assert painted["status"] == "inpainted"
        assert painted["emitted"] is True
        assert Path(painted["artifact"]).is_file()
        assert Path(painted["artifact"]).stat().st_size > 0
    empty = complete_background(association=None, hole_image=None, out_path=tmp_path / "none.png", use_lama=True)
    assert empty["plane_completion"] == "missing"
    assert empty["emitted"] is False
    assert not (tmp_path / "none.png").exists()
    placeholder = plane_fill_from_association(
        {"kept": 27548, "total": 84993, "centers": [[0.0, 1.0, 0.0], [0.1, 1.0, 0.2], [0.2, 1.01, -0.1]]},
        tmp_path / "plane.png",
    )
    assert placeholder["kind"] == "plane_fill_placeholder"
    assert placeholder["lama"] is False
    assert placeholder["plane"]["normal_axis"] == "y"
    assert Path(placeholder["artifact"]).is_file()
    followed = complete_background(
        association={"n_passed_equation_1": 27548, "n_gaussians": 84993},
        out_path=tmp_path / "lama_out.png",
        use_lama=True,
    )
    assert followed["plane_completion"] == "placeholder"
    assert followed["forged"] is False
    if followed.get("status") == "weights_absent":
        assert not (tmp_path / "lama_out.png").exists()


def test_material_properties_parse_without_calling(tmp_path):
    parsed = parse_material_properties(
        '{"material_class": "wood", "friction": 0.4, "mass_kg": 0.25, "density_kg_m3": 600}'
    )
    assert parsed["complete"] is True
    assert parsed["material_class"] == "wood"
    assert parsed["friction"] == 0.4
    assert parsed["mass_kg"] == 0.25
    assert parsed["density_kg_m3"] == 600.0
    held = request_material_properties("fruits", api_key="super-secret-value", live=False)
    assert held["called"] is False
    assert held["model_id"] is None
    assert "super-secret-value" not in json.dumps(held)
    assert "friction" in held["prompt"]
    assert "mass_kg" in held["prompt"]
    assert "density_kg_m3" in held["prompt"]
    summary = partsam_summary_prompt(1168, 15, [{"index": 0, "rgb": [255, 165, 0], "faces": 58883}])
    pending = inspect_parts([summary])
    assert pending["call_cubepart"] is False
    assert "1168" in pending["prompt"]
    assert "15" in pending["prompt"]
    decided = inspect_parts(["cap"], model_reply='{"completion_needed": false}')
    assert decided["completion_needed"] is False
    assert decided["call_cubepart"] is False


def test_cubepart_rope_call_shape_and_articulate_stay_open(tmp_path):
    diffusers = choose_qwen_embed_rope_call(("video_fhw", "device"))
    assert diffusers["signature"] == DIFFUSERS_039_FORWARD
    assert diffusers["positional"] == ("video_fhw",)
    repo = choose_qwen_embed_rope_call(("img_shapes", "txt_seq_lens", "device"))
    assert repo["signature"] == REPO_POS_EMBED_CALL
    assert repo["positional"] == ("img_shapes", "txt_seq_lens")
    assert cubepart_status(None)["status"] == "still_open"
    assert cubepart_status(tmp_path / "missing.ply")["status"] == "still_open"
    empty = tmp_path / "empty.ply"
    empty.write_bytes(b"")
    assert cubepart_status(empty)["status"] == "still_open"
    row = articulate_row(False, free_bytes=10**15)
    assert row["status"] == "still_open"
    assert row["download_started"] is False
    assert row["download_allowed"] is False
    assert row["urdf"] is None
