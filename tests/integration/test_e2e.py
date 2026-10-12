"""End-to-end renders on synthetic media. Slow (minutes): pytest -m slow."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from quillcut import jobs, media
from quillcut.render import render_plan
from quillcut.storage import read_json
from quillcut.types import EditPlan

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def isolated_projects(tmp_path, monkeypatch):
    """Projects and published clips go to a temp dir, not the real formats/ and outputs/."""
    import quillcut.storage as storage
    real = storage.FORMATS
    fake = tmp_path / "formats"
    for key in ("gaming", "podcast"):
        shutil.copytree(real / key, fake / key, ignore=shutil.ignore_patterns("projects", "input"))
    monkeypatch.setattr(storage, "FORMATS", fake)
    monkeypatch.setattr(jobs, "OUTPUTS", tmp_path / "outputs")
    yield


def _assert_publishable(result):
    assert result["qc"] == "PASS", result["failures"]
    info = media.video_info(Path(result["final"]))
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["duration"] >= 30


def test_auto_clips_pass_qc(media):
    results = jobs.run_auto(media["a_1080"], "gaming", "own_footage", max_clips=1, fast=True)
    _assert_publishable(results[0])
    assert Path(results[0]["published"]).exists()


def test_single_source_script(media):
    _assert_publishable(jobs.run_script("underestimated", "gaming", {"a": {"video": media["a_1080"]}},
                                        "own_footage", fast=True))


def test_comparison_script_and_hook_text(media):
    result = jobs.run_script("one_vs_hundred", "gaming",
                             {"a": {"video": media["a_1080"]}, "b": {"video": media["b_1080"]}},
                             "own_footage", fast=True)
    _assert_publishable(result)
    plan = read_json(Path(result["plan"]))
    assert plan["text_cards"][0]["text"] == "1% vs 100%"


def test_720p_uses_wide_fill(media):
    result = jobs.run_script("cold_open", "gaming", {"a": {"video": media["a_720"]}}, "own_footage", fast=True)
    _assert_publishable(result)
    assert {s["layout"] for s in read_json(Path(result["plan"]))["segments"]} == {"wide_fill"}


def test_low_res_phone_recording_is_rejected(media):
    with pytest.raises(jobs.SourceRejected, match="too low-res"):
        jobs.run_auto(media["phone"], "gaming", "own_footage", fast=True)


def test_unknown_rights_never_reach_final(media):
    result = jobs.run_script("cold_open", "gaming", {"a": {"video": media["a_1080"]}}, None, fast=True)
    assert result["qc"] == "FAIL" and "final" not in result
    assert any(f.startswith("rights") for f in result["failures"])


def test_source_without_audio_is_rejected_by_qc(media):
    result = jobs.run_script("cold_open", "gaming", {"a": {"video": media["no_audio"]}}, "own_footage", fast=True)
    assert result["qc"] == "FAIL"
    assert any(f.startswith(("loudness", "audio_level")) for f in result["failures"])


def _framemd5(path: Path) -> list[str]:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v", "-f", "framemd5", "-"],
                         capture_output=True, text=True, check=True).stdout
    return [line.split(",")[-1].strip() for line in out.splitlines() if not line.startswith("#")]


def test_same_plan_renders_identical_frames(media, tmp_path):
    result = jobs.run_script("detail", "gaming", {"a": {"video": media["a_1080"]}}, "own_footage", fast=True)
    plan = EditPlan(**json.loads(Path(result["plan"]).read_text(encoding="utf-8")))
    one = render_plan(plan, tmp_path / "one.mp4", tmp_path / "w1", preset="veryfast")
    two = render_plan(plan, tmp_path / "two.mp4", tmp_path / "w2", preset="veryfast")
    assert _framemd5(one) == _framemd5(two)
