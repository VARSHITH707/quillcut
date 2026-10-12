import pytest

from quillcut import reels
from quillcut.config import load_config
from quillcut.formats import load_style
from quillcut.scripts import PlanRejected, SourceCtx
from quillcut.types import SourceRef


def _src(words=(), shots=None, extra=()):
    ref = SourceRef(path="x.mp4", sha1="0" * 40, width=3840, height=2160, fps="60/1", duration=60.0,
                    has_audio=True, content_box=(3840, 2160, 0, 0))
    shots = shots or [(0, 4), (4, 9.5), (9.5, 15), (15, 22), (22, 31), (31, 40), (40, 52), (52, 60)]
    return reels.ShotSource(SourceCtx(key="a", ref=ref, t_peak=0, lo=0, hi=60, words=list(words)), shots, extra)


def _w(text, start, end):
    return {"word": text, "start": start, "end": end, "probability": 0.9}


def test_part_snaps_to_shot_cuts():
    s, e, _ = reels.snap_part(_src(), 4.6, 20.0)
    assert (s, e) == (4.0, 22.0)          # nearest cut for the start, end of the containing shot


def test_end_never_cuts_a_sentence():
    words = [_w("this", 19.0, 19.4), _w("line", 19.6, 20.2), _w("keeps", 20.4, 21.0), _w("going.", 21.2, 23.0)]
    _, e, notes = reels.snap_part(_src(words), 4.0, 20.0)
    assert e == 31.0 and "finish the line" in notes[0]


def test_start_avoids_mid_sentence_cut():
    words = [_w("hello", 8.8, 9.6), _w("there.", 9.7, 10.4)]
    s, _, _ = reels.snap_part(_src(words), 9.6, 30.0)
    assert s in (4.0, 15.0) and s != 9.5


def test_extra_cuts_are_used():
    s, _, _ = reels.snap_part(_src(extra=(25.0,)), 25.2, 40.0)
    assert s == 25.0


def test_apply_fixes_replaces_and_deletes():
    words = [_w("Dynamic", 6.1, 6.8), _w("officer", 6.8, 7.2), _w("squad", 7.2, 7.5), _w("loose", 52.7, 52.9)]
    fixed = reels.apply_fixes(words, [{"start": 6.05, "end": 7.25, "text": "Dynamic Ops are"},
                                      {"start": 52.0, "end": 53.0, "text": ""}])
    assert [w["word"] for w in fixed] == ["Dynamic", "Ops", "are", "squad"]
    assert fixed[0]["start"] == 6.05 and abs(fixed[2]["end"] - 7.25) < 1e-6


def test_reel_duration_limits_and_end_card(monkeypatch):
    cfg, style = load_config("gaming"), load_style("gaming", "gaming_bold")
    src = _src()
    book = {"key": "t", "end_card": {"source": "a", "from": 52.0, "to": 54.8}}
    plan = reels.build_reel({"id": "ok", "hook": "test hook", "parts": [["a", 4.0, 39.0]], "music": {"track": "none"}},
                            book, {"a": src}, cfg, style, {}, "p")
    assert plan.segments[-1].layout == "fit_blur" and 30 <= plan.duration <= 90
    assert plan.text_cards[0].kind == "hook"
    with pytest.raises(PlanRejected):
        reels.build_reel({"id": "short", "parts": [["a", 4.0, 9.5]], "music": {"track": "none"}},
                         book, {"a": src}, cfg, style, {}, "p")
