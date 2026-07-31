# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import os
import sys
import textwrap
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from simfoundry.models.remote_cache import RemoteCacheMissError, RemoteModelCache, image_digests


@pytest.fixture(autouse=True)
def clear_cache_env(monkeypatch):
    for name in ("CACHE_MODE", "TEST_MODE", "SIMFOUNDRY_MODEL_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)


def write_image(path, color=(10, 20, 30, 255)):
    image = Image.new("RGBA", (8, 8), color)
    image.save(path)
    return path


def png_base64(color=(100, 120, 140, 255)):
    buffer = BytesIO()
    Image.new("RGBA", (6, 6), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeGeminiInlineData:
    def __init__(self, image_base64):
        self.data = base64.b64decode(image_base64)
        self.mime_type = "image/png"

    def __bool__(self):
        return True


class FakeGeminiPart:
    def __init__(self, text=None, image_base64=None):
        self.text = text
        self.inline_data = None if image_base64 is None else FakeGeminiInlineData(image_base64)


class FakeGeminiChunk:
    def __init__(self, text, image_base64=None):
        self.text = text
        parts = []
        if image_base64 is not None:
            parts.append(FakeGeminiPart(image_base64=image_base64))
        self.candidates = [SimpleNamespace(content=SimpleNamespace(parts=parts))]


def make_fake_gemini_client(text="cached text", image_base64=None):
    class FakeGeminiClient:
        calls = 0

        def __init__(self, *args, **kwargs):
            self.models = self

        def generate_content_stream(self, *args, **kwargs):
            FakeGeminiClient.calls += 1
            return [FakeGeminiChunk(text=text, image_base64=image_base64)]

    return FakeGeminiClient


def test_cache_key_uses_image_content_not_absolute_path(tmp_path):
    image_a = write_image(tmp_path / "a.png")
    image_b = write_image(tmp_path / "nested-name.png")

    request_a = {"prompt": "describe", "image_inputs": image_digests(image_a)}
    request_b = {"prompt": "describe", "image_inputs": image_digests(image_b)}

    cache = RemoteModelCache(root_dir=tmp_path / "cache", mode="off")
    assert cache.key_for("gemini", "model", request_a) == cache.key_for("gemini", "model", request_b)


def test_gemini_cache_mode_writes_and_test_mode_replays(monkeypatch, tmp_path):
    from simfoundry.models import vlm

    image_path = write_image(tmp_path / "input.png")
    image_base64 = png_base64()
    FakeClient = make_fake_gemini_client(text="hello cache", image_base64=image_base64)
    monkeypatch.setattr(vlm.genai, "Client", FakeClient)
    monkeypatch.setenv("SIMFOUNDRY_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CACHE_MODE", "1")

    gemini = vlm.Gemini(project="test-project", model="gemini-2.5-flash")
    result = gemini(prompt="prompt", image_paths=image_path, temperature=0, top_p=0, seed=0)

    assert gemini.get_result_text(result) == "hello cache"
    assert gemini.get_result_images(result)[0].size == (6, 6)
    assert FakeClient.calls == 1
    assert len(list((tmp_path / "cache" / "gemini").glob("*.json"))) == 1

    monkeypatch.delenv("CACHE_MODE")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(
        vlm.genai,
        "Client",
        lambda *args, **kwargs: pytest.fail("TEST_MODE must not construct the Gemini client"),
    )

    replay_gemini = vlm.Gemini(project="test-project", model="gemini-2.5-flash")
    replay = replay_gemini(prompt="prompt", image_paths=image_path, temperature=0, top_p=0, seed=0)

    assert replay_gemini.get_result_text(replay) == "hello cache"
    assert replay_gemini.get_result_images(replay)[0].size == (6, 6)


def test_cache_mode_reuses_existing_entry_without_remote_call(monkeypatch, tmp_path):
    from simfoundry.models import vlm

    image_path = write_image(tmp_path / "input.png")
    FakeClient = make_fake_gemini_client(text="first")
    monkeypatch.setattr(vlm.genai, "Client", FakeClient)
    monkeypatch.setenv("SIMFOUNDRY_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CACHE_MODE", "1")

    gemini = vlm.Gemini(project="test-project", model="gemini-2.5-flash")
    first = gemini(prompt="same prompt", image_paths=image_path)
    assert gemini.get_result_text(first) == "first"
    assert FakeClient.calls == 1

    monkeypatch.setattr(
        vlm.genai,
        "Client",
        lambda *args, **kwargs: pytest.fail("CACHE_MODE should reuse existing matching entries"),
    )
    second_gemini = vlm.Gemini(project="test-project", model="gemini-2.5-flash")
    second = second_gemini(prompt="same prompt", image_paths=image_path)
    assert second_gemini.get_result_text(second) == "first"


def test_test_mode_cache_miss_fails_before_remote_call(monkeypatch, tmp_path):
    from simfoundry.models import vlm

    image_path = write_image(tmp_path / "input.png")
    monkeypatch.setenv("SIMFOUNDRY_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(
        vlm.genai,
        "Client",
        lambda *args, **kwargs: pytest.fail("TEST_MODE must not construct the Gemini client"),
    )

    gemini = vlm.Gemini(project="test-project", model="gemini-2.5-flash")
    with pytest.raises(RemoteCacheMissError):
        gemini(prompt="missing", image_paths=image_path)


def test_gpt_image_cache_write_and_replay(monkeypatch, tmp_path):
    from simfoundry.models import vlm

    image_path = write_image(tmp_path / "input.png")
    output_base64 = png_base64(color=(1, 2, 3, 255))

    class FakeOpenAI:
        calls = 0

        def __init__(self, *args, **kwargs):
            self.images = self

        def edit(self, *args, **kwargs):
            FakeOpenAI.calls += 1
            return SimpleNamespace(data=[SimpleNamespace(b64_json=output_base64)])

    monkeypatch.setattr(vlm, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("SIMFOUNDRY_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CACHE_MODE", "1")

    gpt = vlm.GPT(api_key="test")
    result = gpt(prompt="edit", image_path=image_path, image_shape="square")
    assert gpt.get_result_images(result)[0].size == (6, 6)
    assert FakeOpenAI.calls == 1

    monkeypatch.delenv("CACHE_MODE")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(vlm, "OpenAI", lambda *args, **kwargs: pytest.fail("TEST_MODE must not construct OpenAI"))

    replay_gpt = vlm.GPT(api_key="test")
    replay = replay_gpt(prompt="edit", image_path=image_path, image_shape="square")
    assert replay_gpt.get_result_images(replay)[0].size == (6, 6)


def file_manifest(root):
    manifest = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            manifest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


@pytest.mark.integration
def test_orchestrator_fast_stage_cache_and_test_modes(monkeypatch, tmp_path):
    repo_dir = Path(__file__).resolve().parents[1]
    stage_dir = tmp_path / "stages"
    stage_dir.mkdir()
    image_path = write_image(tmp_path / "fixture.png", color=(50, 60, 70, 255))
    cache_dir = tmp_path / "cache"

    (stage_dir / "stage_gemini.py").write_text(textwrap.dedent(
        """
        import os
        from pathlib import Path
        from types import SimpleNamespace
        from simfoundry.models import vlm

        class InlineData:
            data = b""
            mime_type = "image/png"
            def __bool__(self):
                return False

        class FakeChunk:
            text = "stage gemini text"
            candidates = [SimpleNamespace(content=SimpleNamespace(parts=[]))]

        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.models = self
            def generate_content_stream(self, *args, **kwargs):
                if os.environ.get("SIMFOUNDRY_REMOTE_SHOULD_FAIL") == "1":
                    raise RuntimeError("remote blocked")
                return [FakeChunk()]

        if os.environ.get("TEST_MODE") != "1":
            vlm.genai.Client = FakeClient

        out_dir = Path(os.environ["SIMFOUNDRY_TEST_OUTPUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        model = vlm.Gemini(project="test-project", model="gemini-2.5-flash")
        result = model(prompt="stage prompt", image_paths=os.environ["SIMFOUNDRY_TEST_IMAGE"])
        (out_dir / "gemini.txt").write_text(model.get_result_text(result), encoding="utf-8")
        """
    ), encoding="utf-8")

    (stage_dir / "stage_gpt.py").write_text(textwrap.dedent(
        """
        import base64
        import os
        from io import BytesIO
        from pathlib import Path
        from types import SimpleNamespace
        from PIL import Image
        from simfoundry.models import vlm

        def image_b64():
            buffer = BytesIO()
            Image.new("RGBA", (5, 5), (9, 8, 7, 255)).save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("ascii")

        class FakeOpenAI:
            def __init__(self, *args, **kwargs):
                self.images = self
            def edit(self, *args, **kwargs):
                if os.environ.get("SIMFOUNDRY_REMOTE_SHOULD_FAIL") == "1":
                    raise RuntimeError("remote blocked")
                return SimpleNamespace(data=[SimpleNamespace(b64_json=image_b64())])

        if os.environ.get("TEST_MODE") != "1":
            vlm.OpenAI = FakeOpenAI

        out_dir = Path(os.environ["SIMFOUNDRY_TEST_OUTPUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        model = vlm.GPT(api_key="test")
        result = model(prompt="stage edit", image_path=os.environ["SIMFOUNDRY_TEST_IMAGE"], image_shape="square")
        model.get_result_images(result)[0].save(out_dir / "gpt.png")
        """
    ), encoding="utf-8")

    import simfoundry.pipeline.orchestrator as orch
    from simfoundry.pipeline.orchestrator import StageSpec, run_pipeline

    def fake_plan(input_mode: str, **_kwargs):
        assert input_mode == "video"
        return [
            StageSpec("gemini", os.path.relpath(stage_dir / "stage_gemini.py", repo_dir), "fake", "simfoundry", "Fake Gemini stage"),
            StageSpec("gpt", os.path.relpath(stage_dir / "stage_gpt.py", repo_dir), "fake", "simfoundry", "Fake GPT stage"),
        ]

    monkeypatch.setattr(orch, "get_stage_plan", fake_plan)
    monkeypatch.setenv("PYTHONPATH", str(repo_dir))
    monkeypatch.setenv("SIMFOUNDRY_TEST_IMAGE", str(image_path))
    monkeypatch.setenv("SIMFOUNDRY_MODEL_CACHE_DIR", str(cache_dir))

    def run_fake_pipeline(scene_name: str):
        return run_pipeline(
            cwd=str(repo_dir),
            input_mode="video",
            include_ids_csv=None,
            exclude_ids_csv=None,
            exec_mode="direct",
            python_bin=sys.executable,
            env_map={"simfoundry": "simfoundry", "da3": "da3", "hunyuan": "hunyuan", "b1k": "b1k"},
            dry_run=False,
            stream_subseq_enabled=False,
            stream_start_stage=5,
            stream_end_stage=8,
            extra_overrides=[f"root_dir={tmp_path / 'Data'}", f"scene_name={scene_name}"],
        )

    run1 = tmp_path / "run1"
    monkeypatch.setenv("SIMFOUNDRY_TEST_OUTPUT_DIR", str(run1))
    monkeypatch.setenv("CACHE_MODE", "1")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("SIMFOUNDRY_REMOTE_SHOULD_FAIL", raising=False)
    run_fake_pipeline("cache_seed")

    run2 = tmp_path / "run2"
    monkeypatch.setenv("SIMFOUNDRY_TEST_OUTPUT_DIR", str(run2))
    monkeypatch.delenv("CACHE_MODE", raising=False)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SIMFOUNDRY_REMOTE_SHOULD_FAIL", "1")
    run_fake_pipeline("test_replay")

    run3 = tmp_path / "run3"
    monkeypatch.setenv("SIMFOUNDRY_TEST_OUTPUT_DIR", str(run3))
    monkeypatch.setenv("CACHE_MODE", "1")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("SIMFOUNDRY_REMOTE_SHOULD_FAIL", "1")
    run_fake_pipeline("cache_reuse")

    assert file_manifest(run1) == file_manifest(run2) == file_manifest(run3)
    assert len(list(cache_dir.rglob("*.json"))) == 2
