import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from quillcut import captions, scripts
from quillcut.config import load_config
from quillcut.formats import list_formats, load_style
from quillcut.gpu import GpuLock, pid_alive
from quillcut.models.asr import keep_segment
from quillcut.storage import resolve_rights
from quillcut.types import EditPlan, Segment, SourceRef

SRC = Path(__file__).resolve().parents[2] / "src" / "quillcut"
MODEL_LIBS = {"faster_whisper", "ollama", "open_clip", "ultralytics", "mediapipe", "rapidocr", "torch"}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_shared_engine_never_imports_formats_and_only_models_import_model_libs():
    for path in SRC.rglob("*.py"):
        imports = _imports(path)
        assert "formats" not in imports, path
        if "models" not in path.parts:
            # cli.doctor may probe torch for CUDA; that is a check, not model use
            allowed = {"torch"} if path.name == "cli.py" else set()
            assert not (imports & MODEL_LIBS) - allowed, path


def test_renderer_makes_no_creative_decisions():
    tree = ast.parse((SRC / "render.py").read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not modules & {"scripts", "analysis", "jobs", "models"}


def test_four_formats_exist_and_every_template_validates():
    assert {"gaming", "podcast", "gaming_podcast", "longform"} <= set(list_formats())
    templates = scripts.load_templates("gaming")
    assert len(templates) == 9
    floor = load_config("gaming")["duration"]["hard_floor_seconds"]
    assert all(t["duration"] >= floor for t in templates.values())


def test_template_validation_rejects_gaps_and_mid_holds():
    with pytest.raises(ValueError):
        scripts.validate_template({"duration": 10, "sections": [{"start": 0, "end": 4}, {"start": 5, "end": 10}]})
    with pytest.raises(ValueError):
        scripts.validate_template({"duration": 10, "sections": [{"start": 0, "end": 5, "hold": 1},
                                                                 {"start": 5, "end": 10}]})


def test_accelerating_shots_get_shorter_and_sum_to_section():
    lengths = scripts.shot_lengths(10, 5, accelerate=True)
    assert abs(sum(lengths) - 10) < 1e-9
    assert all(a > b for a, b in zip(lengths, lengths[1:]))
    assert lengths[-1] <= lengths[0] * 0.8 ** 4 + 1e-9


def _ref(w=1920, h=1080, dur=120.0):
    return SourceRef(path="x.mp4", sha1="0" * 40, width=w, height=h, fps="30/1", duration=dur,
                     has_audio=True, content_box=(w, h, 0, 0))


def _ctx(key="a", peak=60.0, dur=120.0, w=1920, h=1080):
    return scripts.SourceCtx(key=key, ref=_ref(w, h, dur), t_peak=peak, lo=0.0, hi=dur)


def _plan(name, sources, text=None):
    cfg = load_config("gaming")
    style = load_style("gaming", "gaming_bold")
    tpl = scripts.load_templates("gaming")[name]
    return scripts.build_plan(tpl, sources, cfg, style, clip_id="t", project_id="p", mode="cinematic",
                              rights={"status": "own_footage"}, text=text)


@pytest.mark.parametrize("name", ["cold_open", "build_broken", "underestimated", "detail", "emotional"])
def test_single_source_scripts_reach_the_floor_with_real_footage(name):
    plan = _plan(name, {"a": _ctx()}, text="test line")
    assert plan.duration >= 30
    assert sum(s.hold_after for s in plan.segments) <= 2.0
    for seg in plan.segments:
        assert 0 <= seg.src_in < seg.src_out <= 120


@pytest.mark.parametrize("name", ["movie_vs_game", "old_vs_new", "one_vs_hundred", "hours_vs_hours"])
def test_comparison_scripts_use_both_sources(name):
    plan = _plan(name, {"a": _ctx("a"), "b": _ctx("b", peak=40.0, dur=90.0)})
    used = {s.source for s in plan.segments} | {s.second_source for s in plan.segments if s.second_source}
    assert used == {"a", "b"}
    assert abs(plan.duration - 30) < 0.01


def test_comparison_script_without_b_is_rejected():
    with pytest.raises(scripts.PlanRejected):
        _plan("movie_vs_game", {"a": _ctx()})


def test_floor_rejects_short_template():
    tpl = {"key": "short", "duration": 10, "sections": [{"start": 0, "end": 10, "kind": "side", "source": "a"}]}
    with pytest.raises(scripts.PlanRejected, match="floor"):
        scripts.build_plan(tpl, {"a": _ctx()}, load_config("gaming"), load_style("gaming", "gaming_bold"),
                           clip_id="t", project_id="p", mode="cinematic", rights={})


def test_layout_rule_center_for_1080p_wide_fill_for_720p():
    assert scripts.choose_layout(_ref(1920, 1080), 2.0)[0] == "center_crop"
    assert scripts.choose_layout(_ref(1280, 720), 2.0)[0] == "wide_fill"


def test_editplan_rejects_mid_clip_hold_and_out_of_range():
    plan = _plan("cold_open", {"a": _ctx()})
    segs = [s.model_copy(update={"hold_after": 1.0}) if i == 0 else s for i, s in enumerate(plan.segments)]
    with pytest.raises(ValueError):
        EditPlan(**{**plan.model_dump(), "segments": [s.model_dump() for s in segs]})
    with pytest.raises(ValueError):
        Segment(id="x", source="a", src_in=5, src_out=4)


def test_hook_text_fits_inside_safe_area_and_wraps():
    style, safe = load_style("gaming", "gaming_bold"), load_config("gaming")["safe_area"]
    card = captions.hook_card("this build shouldn't exist and nobody can stop it", 0, 2, style, safe)
    x0, y0, x1, y1 = card.bbox_norm
    assert len(card.lines) <= 2
    assert x0 >= 0.06 and x1 <= 0.94 and y0 >= 0.14 and y1 <= 0.78
    assert card.text == card.text.lower()


def test_devanagari_uses_the_fallback_font():
    style, safe = load_style("gaming", "gaming_bold"), load_config("gaming")["safe_area"]
    card = captions.hook_card("चलो शुरू करते हैं", 0, 2, style, safe)
    assert card.font == "Nirmala UI"


def test_speech_cards_highlight_one_word_at_a_time():
    style, safe = load_style("gaming", "gaming_bold"), load_config("gaming")["safe_area"]
    words = [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.35, "probability": 0.9}
             for i, w in enumerate("he really did that. no way".split())]
    cards = captions.speech_cards(words, 5.0, style, safe)
    assert [c.text for c in cards] == ["he really did that.", "no way"]
    ass = captions.build_ass(cards, style)
    assert ass.count("Dialogue:") == 6
    assert ass.count("\\t(0,120") == 2      # scale-in only on each card's first event
    for card in cards:
        assert card.bbox_norm[2] <= 0.84 + 1e-3  # right rail in the lower half


def test_hallucination_guard():
    seg = lambda text, nsp=0.1, lp=-0.3, cr=1.2: SimpleNamespace(text=text, no_speech_prob=nsp,
                                                                 avg_logprob=lp, compression_ratio=cr)
    assert keep_segment(seg("that was a clean headshot"))
    assert not keep_segment(seg("Thanks for watching!"))
    assert not keep_segment(seg("go go go go go go"))
    assert not keep_segment(seg("hmm", nsp=0.9, lp=-1.5))
    assert not keep_segment(seg("la la", cr=3.0))


def test_gpu_lock_clears_stale_holder(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text("999999", encoding="utf-8")
    assert not pid_alive(999999)
    with GpuLock(lock, poll=0.01, timeout=2):
        assert lock.exists()
    assert not lock.exists()


def test_rights_resolution_order(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    assert resolve_rights(video, None)["status"] == "unknown"
    (tmp_path / "RIGHTS.yaml").write_text("status: client_owned\nclient: Acme\n", encoding="utf-8")
    assert resolve_rights(video, None)["status"] == "client_owned"
    (tmp_path / "v.mp4.yaml").write_text("rights: own_footage\n", encoding="utf-8")
    assert resolve_rights(video, None)["resolved_by"] == "sidecar"
    assert resolve_rights(video, "cc_verified")["resolved_by"] == "flag"
    with pytest.raises(ValueError):
        resolve_rights(video, "mine")
