"""Acceptance tests on real speech: two synthetic voices, Whisper, the full edit, and checks on the output file.

Needs Windows text-to-speech (David and Zira), ffmpeg and a GPU for Whisper large-v3. Skipped otherwise.
"""
from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

from quillcut import conversation_job, media, reelplan

pytestmark = pytest.mark.slow

TURNS = [
    ("Zira", "What do you do, and can you tell us about yourself?"),
    ("David", "My name is Alex. I am a software engineer. I started learning programming three years ago. "
              "Initially, I struggled with Python, and I almost quit twice. Then I built my first small machine "
              "learning project, which changed how I approached technology. I realized that building real things "
              "teaches faster than reading tutorials. Today, I work on applications that use artificial "
              "intelligence to solve everyday problems for small businesses. That is what I do, and that is why "
              "I love this work."),
    ("Zira", "How do you stay productive when working from home?"),
    ("David", "I protect my mornings. The first two hours of every day are completely quiet. No messages, no "
              "meetings, just the hardest task on my list. After that, I take a proper walk outside, because "
              "screens drain my attention faster than I expected. In the afternoon I answer messages in batches "
              "rather than constantly. Honestly, the biggest mistake I made early on was working all day without "
              "any structure. Once I added boundaries, my output doubled and I stopped feeling exhausted. So my "
              "advice is simple: guard your attention first, and everything else gets easier."),
]

PS = r"""
Add-Type -AssemblyName System.Speech
$turns = Get-Content -Raw -Path '{turns}' | ConvertFrom-Json
$i = 0
foreach ($t in $turns) {{
  $i++
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $s.SelectVoice('Microsoft ' + $t.voice + ' Desktop')
  $s.SetOutputToWaveFile('{out}\u' + ('{{0:00}}' -f $i) + '.wav')
  $s.Speak($t.text)
  $s.Dispose()
}}
"""


def _have_voices() -> bool:
    if sys.platform != "win32":
        return False
    try:
        out = subprocess.run(  # nosec B603 B607
            ["powershell", "-NoProfile", "-Command",
             "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer)"
             ".GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"],
            capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "David" in out and "Zira" in out


def _have_gpu() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def interview(tmp_path_factory) -> Path:
    if not _have_voices() or not _have_gpu():
        pytest.skip("needs Windows David/Zira voices and a CUDA GPU")
    import json
    d = tmp_path_factory.mktemp("speech")
    (d / "turns.json").write_text(json.dumps([{"voice": v, "text": t} for v, t in TURNS]), encoding="utf-8")
    (d / "say.ps1").write_text(PS.format(turns=str(d / "turns.json"), out=str(d)), encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(d / "say.ps1")],  # nosec B603 B607
                   check=True, timeout=300)
    parts = sorted(d.glob("u*.wav"))
    lines = []
    for k, p in enumerate(parts):
        lines.append(f"file '{p.as_posix()}'")
        gap = d / f"gap{k}.wav"
        media.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono", "-t",
                   "0.9" if k % 2 == 0 else "3.0", str(gap)])
        lines.append(f"file '{gap.as_posix()}'")
    (d / "list.txt").write_text("\n".join(lines), encoding="utf-8")
    media.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(d / "list.txt"),
               "-ar", "44100", "-ac", "1", str(d / "audio.wav")])
    video = d / "interview.mp4"
    media.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=25", "-i",
               str(d / "audio.wav"), "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
               "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)])
    return video


@pytest.fixture(scope="module")
def made(interview, tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("reels")
    return conversation_job.run(
        interview, rights="own_footage", targets=[30.0, 60.0, 45.0], fast=True, use_llm=False, verify=True,
        publish_dir=out, options=reelplan.PlanOptions(intensity="dynamic"))


def test_complete_answers_survive_a_short_target(made):
    reels = [r for r in made["results"] if r.get("final")]
    assert len(reels) == 2                          # two Q&As in the video, so two reels, not three
    first = reels[0]
    assert first["target"] == 30.0
    assert first["actual"] > 30.0 and first["extended"]            # the answer is longer than 30 s
    for r in reels:
        by_name = {v["name"]: v for v in r["validation"]}
        assert by_name["ends_on_finished_sentence"]["status"] == "PASS", by_name
        assert by_name["starts_on_planned_words"]["status"] == "PASS", by_name
        assert by_name["speech_matches_plan"]["status"] == "PASS", by_name


def test_requested_count_beyond_content_is_explained(made):
    missing = [r for r in made["results"] if r.get("rejected")]
    assert len(missing) == 1
    assert "filler" in missing[0]["rejected"] or "distinct" in missing[0]["rejected"]
    assert made["capacity"] == 2


def test_voices_are_separated(made):
    assert made["speakers"] == ["A", "B"]


def test_every_reel_has_a_truthful_hook_and_real_editing(made):
    for r in made["results"]:
        if not r.get("final"):
            continue
        assert r["hook"]["text"] and r["hook"]["kind"] in ("question", "cold_open", "opening")
        assert r["qc"] == "PASS" and not r.get("needs_review")
        info = media.video_info(Path(r["final"]))
        assert (info["width"], info["height"]) == (1080, 1920)


# --- optional rules, enforced on real speech -------------------------------------------------------------------

def _run_with_rules(interview, tmp_path_factory, text: str, targets=(45.0,)):
    from quillcut import rules as rules_mod
    out = tmp_path_factory.mktemp("ruled")
    parsed = rules_mod.parse(text)
    settings = {"intensity": "dynamic", "aspect": "9:16", "cta": None, "music": None, "include_question": "auto",
                "hook_style": None, "strict": False, "targets": list(targets)}
    rules_mod.apply_to_options(parsed, settings)
    return conversation_job.run(
        interview, rights="own_footage", targets=settings["targets"], fast=True, use_llm=False, verify=True,
        publish_dir=out, rules=parsed,
        options=reelplan.PlanOptions(intensity=settings["intensity"], aspect=settings["aspect"],
                                     hook_style=settings["hook_style"]))


def test_a_topic_rule_picks_only_the_matching_answer(interview, tmp_path_factory):
    res = _run_with_rules(interview, tmp_path_factory, "Only clips about mornings.\nNo profanity.", targets=(45.0, 45.0))
    made = [r for r in res["results"] if r.get("final")]
    skipped = [r for r in res["results"] if r.get("rejected")]
    assert len(made) == 1 and len(skipped) == 1
    assert "rules" in skipped[0]["rejected"].lower() or "distinct" in skipped[0]["rejected"].lower()
    checks = {c["rule"]: c for c in made[0]["rules_check"]}
    assert all(c["ok"] for c in checks.values()), checks
    assert any("mornings" in k for k in checks)
    assert not made[0].get("needs_review")


def test_a_rule_nothing_satisfies_makes_no_reel_and_says_why(interview, tmp_path_factory):
    res = _run_with_rules(interview, tmp_path_factory, "Only clips about quantum physics")
    assert not [r for r in res["results"] if r.get("final")]
    assert "rules" in res["results"][0]["rejected"].lower()


def test_rules_can_change_the_edit_and_the_hook(interview, tmp_path_factory):
    res = _run_with_rules(interview, tmp_path_factory, "Square video\nHigh energy\nStart with a question\nNo cold open")
    reel = next(r for r in res["results"] if r.get("final"))
    assert reel["aspect"] == "1:1" and reel["intensity"] == "high_energy"
    assert reel["hook"]["kind"] in ("question", "opening")
    info = media.video_info(Path(reel["final"]))
    assert (info["width"], info["height"]) == (1080, 1080)