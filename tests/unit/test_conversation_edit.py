"""Edit planning on top of the conversation engine: speakers, hooks, cuts, framing, graphics."""
from __future__ import annotations

import numpy as np
import pytest

from quillcut import conversation as cv
from quillcut import reelplan, reframe, speakers
from quillcut.captions import build_ass
from quillcut.config import load_config
from quillcut.formats import load_style
from quillcut.render import segment_command
from quillcut.types import SourceRef

from test_conversation import ML, FOOD, interview, say, talk, topic  # noqa: E402  (shared synthetic transcripts)

SR = 16000


def voice(f0: float, formant: float, seconds: float, seed: int = 0) -> np.ndarray:
    """A crude synthetic voice: harmonics of f0 shaped by one formant, with syllable-rate amplitude."""
    t = np.arange(int(seconds * SR)) / SR
    sig = np.zeros_like(t)
    for h in range(1, int(4000 / f0)):
        freq = f0 * h
        sig += np.exp(-((freq - formant) / 400.0) ** 2) * np.sin(2 * np.pi * freq * t + h)
    sig *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t) ** 2
    return (0.3 * sig / (np.abs(sig).max() + 1e-9)).astype(np.float32)


def test_two_different_voices_are_separated():
    low, high = lambda s: voice(105, 700, s), lambda s: voice(190, 1400, s)
    parts, spans, t = [], [], 0.0
    for who in "ABBABBABAB":
        seg = low(2.4) if who == "B" else high(2.4)
        parts += [seg, np.zeros(int(0.3 * SR), dtype=np.float32)]
        spans.append((t, t + 2.4))
        t += 2.7
    labels = speakers.assign(np.concatenate(parts), spans, protect=set(range(len(spans))))
    assert len(set(labels)) == 2
    # the same voice always gets the same label
    assert len({labels[i] for i, c in enumerate("ABBABBABAB") if c == "A"}) == 1
    assert len({labels[i] for i, c in enumerate("ABBABBABAB") if c == "B"}) == 1


def test_one_voice_stays_one_speaker():
    parts, spans, t = [], [], 0.0
    for _ in range(8):
        parts += [voice(120, 800, 2.0), np.zeros(int(0.3 * SR), dtype=np.float32)]
        spans.append((t, t + 2.0))
        t += 2.3
    assert set(speakers.assign(np.concatenate(parts), spans)) == {"A"}


def test_questions_are_not_smoothed_into_the_answerer():
    low, high = voice(105, 700, 2.4), voice(190, 1400, 2.4)
    order = "ABBBABBBA"
    parts, spans, t = [], [], 0.0
    for c in order:
        parts += [high if c == "A" else low, np.zeros(int(0.3 * SR), dtype=np.float32)]
        spans.append((t, t + 2.4))
        t += 2.7
    labels = speakers.assign(np.concatenate(parts), spans, protect={4})
    assert labels[4] == labels[0] != labels[1]


def test_sentence_ending_in_for_or_you_is_finished():
    for text in ("Most founders build a product that nobody asked for.", "The polished version will not save you."):
        w, _ = say(0.0, text)
        assert cv.split_sentences(w)[0].complete


# --- crop geometry ----------------------------------------------------------------------------------------

def test_crop_stays_inside_the_frame_and_follows_the_speaker():
    cw, ch, x = reframe.crop_x_pixels(1920, 1080, 9 / 16, 0.8)
    assert (cw, ch) == (606, 1080) and 0 <= x <= 1920 - cw
    assert x + cw / 2 == pytest.approx(0.8 * 1920, abs=3)
    _, _, edge = reframe.crop_x_pixels(1920, 1080, 9 / 16, 0.99)
    assert edge + cw == 1920 or edge + cw == 1920 - 1 or edge + cw <= 1920
    assert reframe.crop_x_pixels(1920, 1080, 9 / 16, None)[2] == (1920 - cw) // 2 // 2 * 2


def test_active_speaker_is_the_face_whose_mouth_moves():
    frames = np.zeros((6, 100, 200, 3), dtype=np.uint8)
    rng = np.random.default_rng(1)
    for i in range(6):                              # the right-hand face's mouth region changes every frame
        frames[i, 62:80, 130:170] = rng.integers(0, 255, size=(18, 40, 3), dtype=np.uint8)
    left, right = (0.1, 0.2, 0.3, 0.8), (0.6, 0.2, 0.9, 0.8)
    center = reframe.pick_active_center(frames, [[left, right]] * 6)
    assert center == pytest.approx(0.75, abs=0.02)


def test_single_face_is_followed_and_no_face_returns_none():
    frames = np.zeros((4, 50, 50, 3), dtype=np.uint8)
    assert reframe.pick_active_center(frames, [[(0.2, 0.1, 0.4, 0.5)]] * 4) == pytest.approx(0.3)
    assert reframe.pick_active_center(frames, [[], [], [], []]) is None


# --- edit planning ----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return load_config("gaming")


@pytest.fixture(scope="module")
def style():
    return load_style("gaming", "gaming_bold")


def ref(w=1920, h=1080, dur=200.0) -> SourceRef:
    return SourceRef(path="x.mp4", sha1="0", width=w, height=h, fps="25/1", duration=dur, has_audio=True,
                     content_box=(w, h, 0, 0))


def plan_for(cfg, style, answer_seconds=44.0, intensity="dynamic", aspect="9:16", centers=None, question=None,
             cold_open=True, nouns=ML):
    words, end = talk(interview(question or "What do you do, and can you tell us about yourself?", nouns,
                                answer_seconds))
    out = cv.analyse(words, end + 3, targets=[60.0])
    sel = out["selections"][0]
    assert sel.candidate is not None
    opts = reelplan.PlanOptions(intensity=intensity, aspect=aspect, cold_open=cold_open)
    built = reelplan.build(sel.candidate, out["sentences"], ref(dur=end + 3), cfg, style, clip_id="r_01",
                           project_id="p", rights={"status": "own_footage"}, options=opts,
                           crop_centers=centers)
    return built, out["sentences"], sel.candidate


def test_pauses_are_tightened_but_words_are_never_cut():
    words = [{"word": "a", "start": 0.0, "end": 0.3}, {"word": "b", "start": 0.4, "end": 0.7},
             {"word": "c", "start": 2.7, "end": 3.0}]
    shots = reelplan.trim_pauses(words, 0.0, 3.2)
    assert len(shots) == 2
    assert shots[0][1] >= 0.7 and shots[1][0] <= 2.7          # both edges stay outside every word
    assert shots[1][0] - shots[0][1] == pytest.approx(2.0 - reelplan.PAUSE_KEEP, abs=0.01)


def test_plan_keeps_every_word_of_the_answer_in_order(cfg, style):
    built, sentences, c = plan_for(cfg, style, intensity="clean")
    mapped = reelplan.map_words(built.plan.segments, [w for s in sentences[c.first:c.last + 1] for w in s.words])
    want = [w["word"] for s in sentences[c.first:c.last + 1] for w in s.words]
    assert [w["word"] for w in mapped] == want
    starts = [w["start"] for w in mapped]
    assert starts == sorted(starts)


def test_hook_is_a_verbatim_line_from_the_conversation(cfg, style):
    built, sentences, c = plan_for(cfg, style, cold_open=False)
    text = " ".join(s.text for s in sentences)
    hook = built.hook["text"].rstrip("…").strip()
    assert hook in text                                   # nothing invented or reworded
    assert built.plan.text_cards[0].kind == "hook" and built.plan.text_cards[0].start < 0.5


def test_cold_open_replays_a_real_line_and_returns_to_the_start(cfg, style):
    first = [(0.0, "Why did the team change the training pipeline so often?")]
    flat = [(0.25, f"The {a} work moved along at a steady pace every week.") for a in
            ("dataset", "feature", "model", "pipeline", "latency", "accuracy")]
    punch = [(0.25, "Honestly, the biggest mistake we made was shipping 3 untested models to customers.")]
    words, end = talk(first + [(0.7, flat[0][1])] + flat[1:] + punch + flat[:2])
    out = cv.analyse(words, end + 3, targets=[90.0])
    c = out["selections"][0].candidate
    built = reelplan.build(c, out["sentences"], ref(dur=end + 3), cfg, style, clip_id="r", project_id="p",
                           rights={"status": "own_footage"}, options=reelplan.PlanOptions(intensity="dynamic"))
    if built.hook["kind"] == "cold_open":
        segs = built.plan.segments
        assert segs[0].src_in > segs[1].src_in            # plays the later line first, then rewinds
        assert built.hook["text"].rstrip("…").lower() in " ".join(s.text for s in out["sentences"]).lower()
    else:  # a non-cold-open hook must still be verbatim
        assert built.hook["text"].rstrip("…") in " ".join(s.text for s in out["sentences"])


def test_intensity_changes_the_edit_not_just_the_captions(cfg, style):
    clean, *_ = plan_for(cfg, style, intensity="clean")
    dyn, *_ = plan_for(cfg, style, intensity="dynamic")
    high, *_ = plan_for(cfg, style, intensity="high_energy")
    assert max(s.zoom for s in clean.plan.segments) == 1.0
    assert max(s.zoom for s in dyn.plan.segments) > 1.0
    assert len(high.plan.segments) >= len(dyn.plan.segments) > len(clean.plan.segments)
    assert high.plan.progress_bar and not clean.plan.progress_bar
    kinds = lambda b: {c.kind for c in b.plan.text_cards}                     # noqa: E731
    assert "lower_third" not in kinds(clean)
    # the label is for a question that is not already the title
    assert ("lower_third" in kinds(dyn)) == (dyn.hook["kind"] != "question")


def test_every_card_is_inside_the_safe_area(cfg, style):
    built, *_ = plan_for(cfg, style, intensity="high_energy")
    safe = cfg["safe_area"]
    for card in built.plan.text_cards:
        x0, y0, x1, y1 = card.bbox_norm
        assert x0 >= safe["left"] - 1e-3 and x1 <= 1 - safe["right"] + 1e-3, card.text
        assert y0 >= safe["top"] - 1e-3 and y1 <= 1 - safe["bottom"] + 1e-3, card.text


def test_framing_follows_the_speaker_and_reaches_the_ffmpeg_command(cfg, style, tmp_path):
    seen = []

    def centers(shots):
        seen.append(len(shots))
        return [0.2 if i % 2 == 0 else 0.8 for i in range(len(shots))]

    built, *_ = plan_for(cfg, style, intensity="dynamic", centers=centers)
    xs = [s.crop_x for s in built.plan.segments]
    assert seen and set(xs) == {0.2, 0.8}
    cmd = segment_command(built.plan.segments[0], built.plan, tmp_path / "o.mkv")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "crop=606:1080:" in graph and "scale=1080:1920" in graph
    zoomed = next(s for s in built.plan.segments if s.zoom > 1.0)
    graph = segment_command(zoomed, built.plan, tmp_path / "o.mkv")
    assert "[zm]" in " ".join(graph)


def test_other_aspect_ratios_render_at_their_own_size(cfg, style, tmp_path):
    for aspect, size in (("1:1", (1080, 1080)), ("16:9", (1920, 1080))):
        built, *_ = plan_for(cfg, style, aspect=aspect, intensity="clean")
        assert (built.plan.output.width, built.plan.output.height) == size
        graph = segment_command(built.plan.segments[0], built.plan, tmp_path / "o.mkv")
        assert f"scale={size[0]}:{size[1]}" in " ".join(graph)


def test_question_label_is_a_lower_third_when_the_title_is_something_else(cfg, style):
    first = [(0.0, "Why did the team change the training pipeline so often?")]
    flat = [(0.25, f"The {a} work moved along at a steady pace every week.") for a in
            ("dataset", "feature", "model", "pipeline", "latency", "accuracy")]
    punch = [(0.25, "Honestly, the biggest mistake we made was shipping 3 untested models to customers.")]
    words, end = talk(first + [(0.7, flat[0][1])] + flat[1:] + punch + flat[:2])
    out = cv.analyse(words, end + 3, targets=[90.0])
    built = reelplan.build(out["selections"][0].candidate, out["sentences"], ref(dur=end + 3), cfg, style,
                           clip_id="r", project_id="p", rights={"status": "own_footage"},
                           options=reelplan.PlanOptions(intensity="high_energy"))
    if built.hook["kind"] == "cold_open":
        label = [c for c in built.plan.text_cards if c.kind == "lower_third"]
        assert label and label[0].text == "Question"
        title = next(c for c in built.plan.text_cards if c.kind == "hook")
        assert label[0].start >= title.end - 0.01          # never on screen together
        x0, y0, x1, y1 = label[0].bbox_norm
        assert y0 >= cfg["safe_area"]["top"] - 1e-3


def test_motion_graphics_are_drawn_into_the_subtitle_track(cfg, style):
    built, *_ = plan_for(cfg, style, intensity="high_energy", nouns=FOOD,
                         question="How do you cook for a crowd of twenty people?")
    ass = build_ass(built.plan.text_cards, built.plan.caption_style)
    assert "\\p1" in ass                                   # vector shapes: title bar, lower third
    assert "\\fscx0" in ass and "\\t(" in ass              # animated reveal
    assert "\\fad(" in ass
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass


def test_music_is_ducked_under_speech_when_a_track_exists(cfg, style, tmp_path):
    track = tmp_path / "bed.wav"
    track.write_bytes(b"RIFF")
    words, end = talk(interview("What do you do for a living?", ML, 40.0))
    out = cv.analyse(words, end + 3, targets=[45.0])
    c = out["selections"][0].candidate
    opts = reelplan.PlanOptions(music={"file": str(track), "licence": "CC0"})
    built = reelplan.build(c, out["sentences"], ref(dur=end + 3), cfg, style, clip_id="r", project_id="p",
                           rights={"status": "own_footage"}, options=opts)
    assert built.plan.audio.music is not None and built.plan.audio.music.duck
    missing = reelplan.build(c, out["sentences"], ref(dur=end + 3), cfg, style, clip_id="r", project_id="p",
                             rights={"status": "own_footage"},
                             options=reelplan.PlanOptions(music={"file": str(tmp_path / "nope.wav")}))
    assert missing.plan.audio.music is None and missing.notes
