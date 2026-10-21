"""The brief's look: cuts on the beat between words, cutaways from mixed footage, grades, catalog captions."""
from __future__ import annotations

import pytest

from quillcut import beats, looks, reelplan
from quillcut import conversation as cv
from quillcut import rules as rules_mod
from quillcut.config import load_config
from quillcut.formats import load_style
from quillcut.render import segment_command
from quillcut.types import SourceRef
from quillcut.web import reels_job

from test_conversation import ML, interview, talk  # noqa: E402  (shared synthetic transcripts)
from test_conversation_edit import ref  # noqa: E402

BPM = 96.0


@pytest.fixture(scope="module")
def cfg():
    return load_config("gaming")


@pytest.fixture(scope="module")
def style():
    return load_style("gaming", "catalog")


def track(seconds: float = 300.0) -> beats.Track:
    p = 60.0 / BPM
    return beats.Track(bpm=BPM, beats=[round(1.0 + k * p, 4) for k in range(int((seconds - 1) / p))],
                       duration=seconds, start=1.0)


def still(path="ad.png") -> SourceRef:
    return SourceRef(path=path, sha1="1", width=1200, height=1500, fps="30/1", duration=3600.0, has_audio=False,
                     content_box=(1200, 1500, 0, 0), still=True)


def clip(path="cartoon.mp4", dur=40.0) -> SourceRef:
    return SourceRef(path=path, sha1="2", width=640, height=480, fps="30/1", duration=dur, has_audio=True,
                     content_box=(640, 480, 0, 0))


def build(cfg, style, tmp_path, **opts):
    words, end = talk(interview("What do you do, and can you tell us about yourself?", ML, 44.0))
    out = cv.analyse(words, end + 3, targets=[60.0])
    c = out["selections"][0].candidate
    song = tmp_path / "song.wav"
    song.write_bytes(b"RIFF")                      # only has to exist: nothing is decoded while planning
    options = reelplan.PlanOptions(music={"file": str(song), "licence": "test", "beats": track().as_dict()},
                                   **opts)
    built = reelplan.build(c, out["sentences"], ref(dur=end + 3), cfg, style, clip_id="r_01", project_id="p",
                           rights={"status": "own_footage"}, options=options)
    body = [w for s in out["sentences"][c.first:c.last + 1] for w in s.words]
    return built, body


def joins(plan) -> list[float]:
    """Source times where one shot of the speaker hands over to the next without a trimmed pause."""
    segs = plan.segments
    return [a.src_out for a, b in zip(segs, segs[1:]) if abs(a.src_out - b.src_in) < 1e-6]


def test_beat_cuts_land_between_words_and_near_a_beat(cfg, style, tmp_path):
    built, words = build(cfg, style, tmp_path, beat_cuts=True, intensity="clean")
    plan = built.plan
    for t in joins(plan):                           # never inside a word (half a frame of rounding allowed)
        assert not any(w["start"] + 0.021 < t < w["end"] - 0.021 for w in words), t
    offset = plan.audio.music.start_offset
    clip_beats = [b - offset for b in track().beats]
    clip_t, beat_cuts = 0.0, 0
    for seg, nxt in zip(plan.segments, plan.segments[1:]):
        clip_t += seg.out_duration
        if abs(seg.src_out - nxt.src_in) < 1e-6 and min(abs(clip_t - b) for b in clip_beats) <= 0.06 + 0.021:
            beat_cuts += 1
    assert beat_cuts >= 5
    assert any("Cut on the beat" in n for n in built.notes)


def test_every_shot_is_whole_frames(cfg, style, tmp_path):
    built, _ = build(cfg, style, tmp_path, beat_cuts=True)
    fps = beats.parse_fps(built.plan.sources["a"].fps)
    for seg in built.plan.segments:
        for t in (seg.src_in, seg.src_out):
            assert abs(t * fps - round(t * fps)) < 1e-3, t


def test_beat_cuts_without_music_fall_back_to_speech_rhythm(cfg, style, tmp_path):
    words, end = talk(interview("What do you do, and can you tell us about yourself?", ML, 44.0))
    out = cv.analyse(words, end + 3, targets=[60.0])
    built = reelplan.build(out["selections"][0].candidate, out["sentences"], ref(dur=end + 3), cfg, style,
                           clip_id="r", project_id="p", rights={}, options=reelplan.PlanOptions(beat_cuts=True))
    assert len(built.plan.segments) > 3
    assert any("needs a music track" in n for n in built.notes)


def test_cutaways_follow_the_rules(cfg, style, tmp_path):
    built, _ = build(cfg, style, tmp_path, beat_cuts=True, mix=[("m0", still()), ("m1", clip())],
                     mix_amount="light")
    plan = built.plan
    segs = plan.segments
    cut = [i for i, s in enumerate(segs) if s.video_source]
    assert cut, "some shots should become cutaways"
    total = sum(s.out_duration for s in segs)
    assert sum(segs[i].out_duration for i in cut) <= reelplan.MIX_SHARE["light"] * total + 1e-6
    assert all(b - a > 1 for a, b in zip(cut, cut[1:])), "never two cutaways in a row"
    assert cut[-1] != len(segs) - 1, "the reel ends on the speaker"
    starts = [sum(s.out_duration for s in segs[:i]) for i in cut]
    assert min(starts) >= reelplan.HOOK_SECONDS
    for i in cut:
        s = segs[i]
        assert s.source == "a"                      # the speaker's audio continues
        assert reelplan.CUTAWAY_SPAN[0] <= s.out_duration <= reelplan.CUTAWAY_SPAN[1]
        assert s.video_source in plan.sources
        assert (s.push_in > 0) == plan.sources[s.video_source].still
    assert {segs[i].video_source for i in cut} == {"m0", "m1"}      # the files take turns


def test_photo_cutaway_loops_the_image_for_the_shot(cfg, style, tmp_path):
    built, _ = build(cfg, style, tmp_path, mix=[("m0", still())], mix_amount="medium")
    seg = next(s for s in built.plan.segments if s.video_source == "m0")
    cmd = segment_command(seg, built.plan, tmp_path / "o.mkv")
    assert cmd[cmd.index("-loop") + 1] == "1" and "ad.png" in cmd
    # the speaker's audio is the second input, starting where the shot starts
    assert cmd.count("-i") == 2 and f"{seg.src_in:.3f}" in cmd


def test_grade_reaches_the_plan_and_every_grade_builds(cfg, style, tmp_path):
    built, _ = build(cfg, style, tmp_path, grade="hazy")
    assert built.plan.grade == "hazy"
    assert looks.grade_graph(None, 1080, 1920) == "" and looks.grade_graph("none", 1080, 1920) == ""
    for name in looks.GRADES:
        if name != "none":
            assert looks.grade_graph(name, 1080, 1920)


def test_catalog_style_keeps_its_own_accent_when_page_colours_are_dropped(cfg, style, tmp_path):
    built, _ = build(cfg, style, tmp_path, accent=None)
    assert built.plan.caption_style["colors"]["accent"] == style["colors"]["accent"]


BRIEF = """The look: old-money polish colliding with early Chicago and Atlanta rap. Elegant, dark, a little hazy.
✕  Loud meme edits. Impact-font captions, vine booms, zoom punches, Subway Surfers split screens.
3. Cut on the beat.  Pick a track that fits the mood of the moment.
Text on screen is optional. If you use it, keep it tasteful. Think catalog typography, not meme font.
✕  No hook, off-beat cuts, or lazy editing"""


def test_the_brief_sets_the_look():
    r = rules_mod.parse(BRIEF)
    assert (r.beat_cuts, r.grade, r.caption_style, r.intensity) == (True, "hazy", "catalog", "clean")
    opts = {"targets": [45.0], "preset": "gaming_bold", "beat_cuts": False, "grade": "none"}
    notes = rules_mod.apply_to_options(r, opts)
    assert opts["preset"] == "catalog" and opts["style_from_rules"] and opts["grade"] == "hazy"
    assert any("preset" in n for n in notes)


def test_a_negative_line_never_turns_a_grade_on():
    assert rules_mod.parse("✕ Hazy, washed-out footage.").grade is None


def test_reel_options_validate_the_new_fields():
    base = {"reels": [{"target": 45}], "preset": "catalog"}
    good = reels_job.validate({**base, "grade": "vhs", "beat_cuts": True, "mix_amount": "medium",
                               "music_upload": "0123456789ab", "mix_uploads": ["0123456789ab", "0123456789ab"]})
    assert good["grade"] == "vhs" and good["beat_cuts"] and good["mix_uploads"] == ["0123456789ab"]
    for bad in ({"grade": "sepia"}, {"mix_amount": "heavy"}, {"music_upload": "../x"},
                {"mix_uploads": ["nope"]}, {"mix_uploads": ["0123456789ab"] * 13}):
        with pytest.raises(reels_job.OptionError):
            reels_job.validate({**base, **bad})
