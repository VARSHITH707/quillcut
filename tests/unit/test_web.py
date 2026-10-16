"""Web app: API validation, local-only protections, job flow (pipeline stubbed), downloads."""
from __future__ import annotations

import http.client
import io
import json
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import pytest

from quillcut import jobs as pipeline_jobs
from quillcut.web import copywriter, server


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(server, "JOBS", tmp_path / "jobs")
    monkeypatch.setattr(server, "STORE", None)
    monkeypatch.setattr(server, "ollama_state", lambda: {"running": False, "model": "m", "has_model": False})
    monkeypatch.setattr(copywriter, "_ask_ollama", lambda *a, **k: None)
    monkeypatch.setattr("quillcut.media.video_info",
                        lambda p: {"duration": 90.0, "width": 1920, "height": 1080, "fps": 30, "has_audio": True})
    httpd = server.make_server(0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], tmp_path
    httpd.shutdown()
    httpd.server_close()


def call(port: int, method: str, path: str, body: bytes | dict | None = None, headers: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    hdrs = dict(headers or {})
    if isinstance(body, dict):
        body = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    conn.request(method, path, body=body, headers=hdrs)
    res = conn.getresponse()
    data = res.read()
    conn.close()
    return res.status, dict(res.getheaders()), data


def as_json(data: bytes) -> dict:
    return json.loads(data.decode())


def upload(port: int, name: str = "match.mp4", payload: bytes = b"0" * 2048):
    return call(port, "POST", f"/api/upload?name={quote(name, safe='%')}", payload)


OPTIONS = {"preset": "gaming_bold", "colors": {"base": "FFFFFF", "active": "FFD400", "stroke": "000000"},
           "caption_size": 78, "max_clips": 2, "quality": "draft", "rights": "own_footage", "rights_confirmed": True}


def test_status_lists_presets_formats_and_rights(app):
    port, _ = app
    status, _, data = call(port, "GET", "/api/status")
    body = as_json(data)
    assert status == 200
    assert {"gaming_bold", "gaming_headline", "brand_clean"} <= {p["key"] for p in body["presets"]}
    assert "own_footage" in body["rights"]
    assert any(f["key"] == "gaming" and f["available"] for f in body["formats"])


def test_busy_port_gives_a_plain_message(app):
    port, _ = app
    with pytest.raises(SystemExit) as exc:
        server.serve(port, open_browser=False)
    assert "not available" in str(exc.value) and f"--port {port + 1}" in str(exc.value)


def test_index_page_and_security_headers(app):
    port, _ = app
    status, headers, data = call(port, "GET", "/")
    assert status == 200 and b"Quillcut" in data
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_static_path_traversal_is_blocked(app):
    port, _ = app
    for path in ("/../../pyproject.toml", "/..%2f..%2fpyproject.toml", "/%2e%2e/%2e%2e/pyproject.toml"):
        status, _, _ = call(port, "GET", path)
        assert status == 404


def test_foreign_host_header_is_refused(app):
    port, _ = app
    status, _, _ = call(port, "GET", "/api/status", headers={"Host": "evil.example.com"})
    assert status == 403


def test_cross_origin_post_is_refused(app):
    port, _ = app
    status, _, _ = call(port, "POST", "/api/jobs", {}, headers={"Origin": "https://evil.example.com"})
    assert status == 403


def test_upload_rejects_wrong_type_empty_and_too_large(app, monkeypatch):
    port, _ = app
    assert upload(port, "notes.txt")[0] == 415
    assert call(port, "POST", "/api/upload?name=a.mp4", b"")[0] == 400
    monkeypatch.setattr(server, "MAX_UPLOAD", 1000)
    assert upload(port, "a.mp4", b"0" * 2000)[0] == 413


def test_upload_accepts_a_video_and_cleans_the_name(app):
    port, tmp = app
    status, _, data = upload(port, "..%2F..%2Fevil name.mp4")
    body = as_json(data)
    assert status == 201 and body["duration"] == 90.0
    saved = list((tmp / "uploads" / body["upload_id"]).iterdir())
    assert len(saved) == 1 and "/" not in saved[0].name and ".." not in saved[0].name


def test_upload_rejects_short_video(app, monkeypatch):
    port, _ = app
    monkeypatch.setattr("quillcut.media.video_info", lambda p: {"duration": 5.0, "width": 1920, "height": 1080})
    status, _, data = upload(port)
    assert status == 422 and "20 seconds" in as_json(data)["error"]


@pytest.mark.parametrize("patch,message", [
    ({"rights_confirmed": False}, "right to use"),
    ({"rights": "stolen"}, "right to use"),
    ({"colors": {"base": "red"}}, "hex color"),
    ({"caption_size": 5}, "between 40 and 140"),
    ({"max_clips": 99}, "between 1 and 8"),
    ({"preset": "nope"}, "caption style"),
    ({"quality": "ultra"}, "draft or high"),
])
def test_job_options_are_validated(app, patch, message):
    port, _ = app
    up = as_json(upload(port)[2])
    status, _, data = call(port, "POST", "/api/jobs", {**OPTIONS, **patch, "upload_id": up["upload_id"]})
    assert status == 400 and message in as_json(data)["error"]


def test_job_requires_a_real_upload(app):
    port, _ = app
    status, _, _ = call(port, "POST", "/api/jobs", {**OPTIONS, "upload_id": "../../etc/passwd"})
    assert status == 400
    status, _, _ = call(port, "POST", "/api/jobs", {**OPTIONS, "upload_id": "0" * 12})
    assert status == 400


def _fake_run_auto(tmp_path: Path):
    def fake(video, fmt, rights, max_clips, fast=False, *, caption_style=None, style_overrides=None,
             publish_dir=None, on_progress=None):
        assert caption_style == "gaming_bold"
        assert style_overrides["colors"]["active"] == "FFD400"
        on_progress("analysing")
        project = tmp_path / "proj"
        for sub in ("final", "metadata", "analysis"):
            (project / sub).mkdir(parents=True, exist_ok=True)
        publish_dir.mkdir(parents=True, exist_ok=True)
        data = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 40
        (publish_dir / "c_01.mp4").write_bytes(data)
        (project / "final" / "c_01.mp4").write_bytes(data)
        (project / "metadata" / "c_01.json").write_text(json.dumps(
            {"hook_text": "the clutch", "topic": "a big play", "source_ranges": [["a", 0.0, 10.0]]}))
        (project / "analysis" / "transcript.json").write_text(json.dumps(
            {"words": [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, w in enumerate(
                "Then the final round started and everyone expected the usual rush".split())]}))
        on_progress("rendering", 1, 2)
        return [{"clip_id": "c_01", "qc": "PASS", "duration": 32.0, "final": str(project / "final" / "c_01.mp4"),
                 "published": str(publish_dir / "c_01.mp4")},
                {"clip_id": "c_02", "rejected": "no peak #2 found in 0-90s"}]
    return fake


def wait_for(port: int, job_id: str, timeout: float = 15.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = as_json(call(port, "GET", f"/api/jobs/{job_id}")[2])
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_full_job_flow_downloads_and_zip(app, monkeypatch):
    port, tmp = app
    monkeypatch.setattr(pipeline_jobs, "run_auto", _fake_run_auto(tmp))
    up = as_json(upload(port)[2])
    status, _, data = call(port, "POST", "/api/jobs", {**OPTIONS, "upload_id": up["upload_id"]})
    assert status == 201
    job = wait_for(port, as_json(data)["id"])
    assert job["status"] == "done", job
    (clip,) = job["clips"]
    assert clip["qc"] == "PASS" and clip["caption"] and clip["hashtags"]
    assert clip["source"] == "template"          # Ollama stubbed off: labelled, never passed off as AI
    assert all(t.startswith("#") for t in clip["hashtags"])

    status, headers, data = call(port, "GET", clip["url"])
    assert status == 200 and headers["Content-Type"] == "video/mp4" and len(data) > 1000
    status, headers, part = call(port, "GET", clip["url"], headers={"Range": "bytes=0-9"})
    assert status == 206 and len(part) == 10 and headers["Content-Range"].startswith("bytes 0-9/")
    status, headers, _ = call(port, "GET", clip["download"])
    assert "attachment" in headers["Content-Disposition"]
    assert call(port, "GET", clip["url"], headers={"Range": "bytes=999999-"})[0] == 416

    status, _, data = call(port, "GET", f"/api/jobs/{job['id']}/zip")
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert status == 200 and "c_01.mp4" in names and "captions.txt" in names


def test_media_rejects_traversal(app):
    port, _ = app
    for path in ("/media/000000000000/..%2f..%2fjob.json", "/media/zzzz/c_01.mp4", "/media/000000000000/c_01.mp4"):
        assert call(port, "GET", path)[0] == 404


def test_pipeline_rejection_becomes_a_plain_message(app, monkeypatch):
    port, _ = app

    def boom(*a, **k):
        raise pipeline_jobs.SourceRejected("real picture is 640x360; under 480 px is too low-res")
    monkeypatch.setattr(pipeline_jobs, "run_auto", boom)
    up = as_json(upload(port)[2])
    job_id = as_json(call(port, "POST", "/api/jobs", {**OPTIONS, "upload_id": up["upload_id"]})[2])["id"]
    job = wait_for(port, job_id)
    assert job["status"] == "failed" and "too low-res" in job["error"]


def test_unexpected_error_does_not_leak_details(app, monkeypatch):
    port, _ = app
    monkeypatch.setattr(pipeline_jobs, "run_auto", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("C:/secret/path")))
    up = as_json(upload(port)[2])
    job_id = as_json(call(port, "POST", "/api/jobs", {**OPTIONS, "upload_id": up["upload_id"]})[2])["id"]
    job = wait_for(port, job_id)
    assert job["status"] == "failed" and "secret" not in job["error"]


def test_delete_removes_job(app, monkeypatch):
    port, tmp = app
    monkeypatch.setattr(pipeline_jobs, "run_auto", _fake_run_auto(tmp))
    up = as_json(upload(port)[2])
    job_id = as_json(call(port, "POST", "/api/jobs", {**OPTIONS, "upload_id": up["upload_id"]})[2])["id"]
    wait_for(port, job_id)
    assert call(port, "DELETE", f"/api/jobs/{job_id}")[0] == 200
    assert call(port, "GET", f"/api/jobs/{job_id}")[0] == 404


# --- copywriter + style merge ---------------------------------------------------------------------------

def test_hashtags_are_clean_unique_and_capped():
    tags = copywriter.clean_tags(["Final Round!", "#final round", "x", "GAMING", *map(str, range(20))], ["#shorts"])
    assert tags[0] == "#finalround" and tags.count("#finalround") == 1
    assert "#x" not in tags and len(tags) <= copywriter.MAX_TAGS


def test_template_copy_uses_only_the_transcript():
    out = copywriter.template_copy(None, None, "Nobody expected the clutch. It ended fast.")
    assert out["source"] == "template" and "clutch" in out["caption"].lower()
    assert "#clutch" in out["hashtags"] and len(out["caption"]) <= copywriter.MAX_CAPTION


def test_describe_uses_model_answer_when_valid(monkeypatch):
    monkeypatch.setattr(copywriter, "_ask_ollama",
                        lambda *a, **k: {"caption": "A big play", "hashtags": ["#a1", "#b2"], "source": "ollama"})
    out = copywriter.describe("hook", "topic", "text", url="http://x", model="m")
    assert out["source"] == "ollama"


def test_describe_retries_once_then_falls_back(monkeypatch):
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        return {"caption": "Second try", "hashtags": ["#ok1"], "source": "ollama"} if len(calls) == 2 else None
    monkeypatch.setattr(copywriter, "_ask_ollama", flaky)
    assert copywriter.describe("h", "t", "words", url="http://x", model="m")["caption"] == "Second try"
    monkeypatch.setattr(copywriter, "_ask_ollama", lambda *a, **k: None)
    assert copywriter.describe("h", "t", "words", url="http://x", model="m")["source"] == "template"


def test_merge_style_never_mutates_and_clamps_min_size():
    base = {"colors": {"base": "FFFFFF", "active": "FFD400"}, "speech": {"size": 78, "min_size": 56}}
    merged = pipeline_jobs.merge_style(base, {"colors": {"active": "00FF00"}, "speech": {"size": 50}})
    assert base["colors"]["active"] == "FFD400" and base["speech"]["size"] == 78
    assert merged["colors"] == {"base": "FFFFFF", "active": "00FF00"}
    assert merged["speech"]["size"] == 50 and merged["speech"]["min_size"] == 50


# --- conversation reels: analysis, per-reel durations, regeneration --------------------------------------------

from quillcut import conversation_job  # noqa: E402

REELS = {"kind": "reels", "preset": "gaming_bold", "colors": {"base": "FFFFFF", "active": "FFD400", "stroke": "000000"},
         "caption_size": 78, "quality": "draft", "rights": "own_footage", "rights_confirmed": True,
         "reels": [{"target": 35}, {"target": 60}], "intensity": "dynamic", "aspect": "9:16"}


def _fake_run(tmp_path: Path, seen: dict):
    def fake(video, *, rights, targets, publish_dir, on_progress=None, exclude=None, prefer=None, profile=None,
             strict=False, include_question="auto", options=None, **kw):
        seen.update(targets=targets, exclude=exclude, prefer=prefer, profile=profile, strict=strict,
                    include_question=include_question, options=options)
        on_progress("rendering", 0, len(targets))
        publish_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for k, t in enumerate(targets, start=1):
            if k == 2:
                results.append({"reel": k, "target": t, "rejected": "No further complete, distinct moment is long enough."})
                continue
            f = publish_dir / f"r_{k:02d}.mp4"
            f.write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 20)
            results.append({"reel": k, "clip_id": f"r_{k:02d}", "target": t, "actual": 42.0, "extended": True,
                            "note": "Extended from 35s to 42s to finish the thought.", "source_start": 10.0,
                            "source_end": 52.0, "question_included": True, "partial": False,
                            "hook": {"kind": "question", "text": "What do you do?"}, "topic": "", "scores": {},
                            "score": 0.9, "reasons": [], "intensity": "dynamic", "aspect": "9:16", "notes": [],
                            "qc": "PASS", "failures": [], "warnings": [], "final": str(tmp_path / "p" / "final" / "r.mp4"),
                            "published": str(f), "duration": 42.0,
                            "validation": [{"name": "speech_matches_plan", "status": "PASS", "message": "100%"}]})
        return {"results": results, "capacity": 1, "sentences": 20, "speakers": ["A", "B"]}
    return fake


def test_analyze_job_reports_what_the_video_supports(app, monkeypatch):
    port, _ = app
    monkeypatch.setattr(conversation_job, "analyse_only", lambda video, **k: {
        "duration": 130.0, "capacity": 3, "moments": [], "questions": 3, "speakers": ["A", "B"], "units": 3,
        "sentences": 27, "min_seconds": 12.0})
    up = as_json(upload(port)[2])
    status, _, data = call(port, "POST", "/api/jobs", {"kind": "analyze", "upload_id": up["upload_id"]})
    assert status == 201
    job = wait_for(port, as_json(data)["id"])
    assert job["status"] == "done" and job["analysis"]["capacity"] == 3 and job["kind"] == "analyze"


@pytest.mark.parametrize("patch,message", [
    ({"reels": []}, "between 1 and 20"),
    ({"reels": [{"target": 5}]}, "between 15 and 180"),
    ({"reels": [{"target": "soon"}]}, "number of seconds"),
    ({"reels": [{"target": 40}] * 21}, "between 1 and 20"),
    ({"intensity": "wild"}, "intensity"),
    ({"aspect": "4:3"}, "Aspect"),
    ({"include_question": "maybe"}, "Question setting"),
    ({"accent": "orange"}, "hex color"),
    ({"caption_y": 0.2}, "centre third"),
    ({"music": "loud"}, "Music"),
    ({"prefer": [5, 2]}, "Preferred range"),
])
def test_reels_job_options_are_validated(app, patch, message):
    port, _ = app
    up = as_json(upload(port)[2])
    status, _, data = call(port, "POST", "/api/jobs", {**REELS, **patch, "upload_id": up["upload_id"]})
    assert status == 400 and message in as_json(data)["error"], data


def test_reels_job_makes_clips_with_target_actual_and_note(app, monkeypatch):
    port, tmp = app
    seen: dict = {}
    monkeypatch.setattr(conversation_job, "run", _fake_run(tmp, seen))
    up = as_json(upload(port)[2])
    status, _, data = call(port, "POST", "/api/jobs", {**REELS, "upload_id": up["upload_id"], "strict": True,
                                                       "content_style": "interview", "include_question": "yes"})
    assert status == 201
    job = wait_for(port, as_json(data)["id"])
    assert job["status"] == "done", job
    assert seen["targets"] == [35.0, 60.0] and seen["strict"] is True and seen["profile"] == "interview"
    good, missing = job["clips"][0], job["clips"][1]
    assert good["target"] == 35 and good["actual"] == 42.0 and good["extended"] is True
    assert good["hook"]["kind"] == "question" and good["caption"] and good["url"].endswith("r_01.mp4")
    assert good["validation"][0]["status"] == "PASS"
    assert "rejected" in missing and "file" not in missing
    assert "Made 1 of 2" in job["note"]
    status, headers, body = call(port, "GET", good["url"])
    assert status == 200 and headers["Content-Type"] == "video/mp4"


def test_regeneration_passes_exclude_and_prefer_ranges(app, monkeypatch):
    port, tmp = app
    seen: dict = {}
    monkeypatch.setattr(conversation_job, "run", _fake_run(tmp, seen))
    up = as_json(upload(port)[2])
    body = {**REELS, "reels": [{"target": 45}], "upload_id": up["upload_id"], "exclude": [[10, 52]],
            "prefer": [60, 100]}
    job = wait_for(port, as_json(call(port, "POST", "/api/jobs", body)[2])["id"])
    assert job["status"] == "done"
    assert seen["exclude"] == [(10.0, 52.0)] and seen["prefer"] == (60.0, 100.0)


def test_logo_upload_accepts_images_only(app):
    port, _ = app
    from PIL import Image
    import io as _io
    buf = _io.BytesIO()
    Image.new("RGBA", (40, 40), (248, 88, 48, 255)).save(buf, "PNG")
    status, _, data = call(port, "POST", "/api/upload?kind=logo&name=brand.png", buf.getvalue())
    assert status == 201 and as_json(data)["kind"] == "logo"
    assert call(port, "POST", "/api/upload?kind=logo&name=brand.exe", b"MZ" * 50)[0] == 415
    assert call(port, "POST", "/api/upload?kind=logo&name=fake.png", b"not an image at all" * 20)[0] == 422



# --- social post: re-hook, bullets, call to action ------------------------------------------------------------

TRANSCRIPT = ("My name is Alex. I am a software engineer. I started learning programming three years ago. "
              "Initially I struggled with Python and almost quit twice. Then I built my first machine learning "
              "project which changed how I approached technology. Today I work on applications that solve real "
              "problems for small businesses.")


def test_social_post_has_a_short_rehook_bullets_and_a_cta(monkeypatch):
    monkeypatch.setattr(copywriter, "_ask_post", lambda *a, **k: None)
    post = copywriter.social_post("What do you do?", "Learning to code", TRANSCRIPT, content_style="educational",
                                  url="http://x", model="m")
    assert post["source"] == "template"
    assert len(post["rehook"].split()) < 10
    assert 3 <= len(post["bullets"]) <= 4
    assert all(len(b.split()) <= 19 for b in post["bullets"])
    assert post["caption"].startswith(post["rehook"]) and post["caption"].rstrip().endswith("Save this guide for later.")
    assert post["caption"].count("\u2022") == len(post["bullets"])
    # every bullet is the speaker's own sentence (possibly shortened), never new wording
    plain = TRANSCRIPT.replace(".", "").lower()
    assert all(b.rstrip("\u2026").replace(".", "").lower() in plain for b in post["bullets"])


@pytest.mark.parametrize("style,cta", [("educational", "Save this guide for later."),
                                      ("debate", "Comment your thoughts below."),
                                      ("business", "Send this to a teammate.")])
def test_call_to_action_follows_the_clip_type(style, cta):
    assert copywriter.cta_for(copywriter.clip_type(style, "")) == cta


def test_clip_type_is_read_from_the_words_when_no_style_is_chosen():
    biz = "Our customers drove revenue growth because the startup pricing and sales strategy worked for clients."
    opinion = "Honestly I think this is wrong, and I believe you should disagree, because it is a myth."
    assert copywriter.clip_type(None, biz) == "business"
    assert copywriter.clip_type(None, opinion) == "opinion"
    assert copywriter.clip_type("auto", "We baked bread.") == "educational"


def test_model_bullets_are_refused_when_they_are_not_from_the_transcript():
    assert not copywriter._grounded("Sharks migrate across the arctic every winter season", TRANSCRIPT)
    assert not copywriter._grounded("Quantum computers solve everything instantly for investors", TRANSCRIPT)
    assert copywriter._grounded("I built my first machine learning project", TRANSCRIPT)


def test_unusable_model_post_falls_back_to_the_template(monkeypatch):
    monkeypatch.setattr(copywriter, "_ask_post", lambda *a, **k: None)
    post = copywriter.social_post(None, None, TRANSCRIPT, content_style=None, url="http://x", model="m")
    assert post["source"] == "template" and post["bullets"] and post["hashtags"]


# --- two inputs: a video (required) and rules (optional) ---------------------------------------------------------

def test_nothing_runs_without_a_video_even_with_rules(app):
    port, _ = app
    for body in ({**REELS}, {**REELS, "rules": "Only clips about pricing"}, {"kind": "analyze"},
                 {**REELS, "rules": "x", "upload_id": "0" * 12}, {**REELS, "upload_id": "../../etc"}):
        status, _, data = call(port, "POST", "/api/jobs", body)
        assert status == 400, body
        assert "Upload a video first" in as_json(data)["error"]
    status, _, data = call(port, "POST", "/api/jobs", {**REELS, "rules": "Only clips about pricing"})
    assert "Rules on their own can't make reels" in as_json(data)["error"]


def test_a_video_alone_is_enough(app, monkeypatch):
    port, tmp = app
    seen: dict = {}
    monkeypatch.setattr(conversation_job, "run", _fake_run(tmp, seen))
    up = as_json(upload(port)[2])
    job = wait_for(port, as_json(call(port, "POST", "/api/jobs", {**REELS, "upload_id": up["upload_id"]})[2])["id"])
    assert job["status"] == "done"
    assert job["rules_report"] == {"applied": [], "advisory": [], "filters": False}


def test_rules_are_parsed_applied_and_reported(app, monkeypatch):
    port, tmp = app
    seen: dict = {}

    def capture(video, *, rules=None, **kw):
        seen["rules"] = rules
        return _fake_run(tmp, seen)(video, **kw)
    monkeypatch.setattr(conversation_job, "run", capture)
    up = as_json(upload(port)[2])
    text = "Only clips about pricing\nNo profanity\nHigh energy\nSquare video\nKeep the tone warm"
    job = wait_for(port, as_json(call(port, "POST", "/api/jobs",
                                      {**REELS, "upload_id": up["upload_id"], "rules": text})[2])["id"])
    assert job["status"] == "done"
    r = seen["rules"]
    assert r.include_any == ["pricing"] and r.profanity and r.filters
    opts = job["options"]
    assert opts["intensity"] == "high_energy" and opts["aspect"] == "1:1"          # rules beat the page
    assert any("intensity" in n for n in job["rule_notes"])
    assert job["rules_report"]["advisory"] == ["Keep the tone warm"]
    assert any("Not enforced automatically: Keep the tone warm" in m for m in job["messages"])


def test_rules_are_validated(app):
    port, _ = app
    up = as_json(upload(port)[2])
    status, _, data = call(port, "POST", "/api/jobs", {**REELS, "upload_id": up["upload_id"], "rules": "x" * 4001})
    assert status == 400 and "at most 4000" in as_json(data)["error"]
    status, _, data = call(port, "POST", "/api/jobs", {**REELS, "upload_id": up["upload_id"], "rules": ["a"]})
    assert status == 400 and "must be text" in as_json(data)["error"]


def test_rules_preview_needs_no_video_and_counts_surviving_moments(app, monkeypatch):
    port, _ = app
    status, _, data = call(port, "POST", "/api/rules/preview", {"rules": "Only clips about pricing\nBe funny"})
    body = as_json(data)
    assert status == 200 and body["capacity"] is None
    assert body["report"]["applied"][0]["kind"] == "filter" and body["report"]["advisory"] == ["Be funny"]

    cands = [{"start": 0.0, "end": 40.0, "duration": 40.0, "score": 0.9, "text": "We changed our pricing in March"},
             {"start": 50.0, "end": 90.0, "duration": 40.0, "score": 0.8, "text": "We baked sourdough bread daily"},
             {"start": 100.0, "end": 140.0, "duration": 40.0, "score": 0.7, "text": "Pricing experiments taught us a lot"}]
    monkeypatch.setattr(conversation_job, "analyse_only", lambda video, **k: {
        "duration": 150.0, "capacity": 3, "moments": [], "questions": 3, "speakers": ["A"], "units": 3,
        "sentences": 9, "min_seconds": 12.0, "candidates": cands})
    up = as_json(upload(port)[2])
    job_id = as_json(call(port, "POST", "/api/jobs", {"kind": "analyze", "upload_id": up["upload_id"]})[2])["id"]
    job = wait_for(port, job_id)
    assert "candidates" not in job["analysis"]                         # the page never receives the raw texts
    status, _, data = call(port, "POST", "/api/rules/preview",
                           {"rules": "only about pricing", "analysis_job": job_id})
    assert as_json(data)["capacity"]["capacity"] == 2 and as_json(data)["capacity"]["passing"] == 2
    status, _, data = call(port, "POST", "/api/rules/preview", {"rules": "", "analysis_job": job_id})
    assert as_json(data)["capacity"]["capacity"] == 3


def test_rules_preview_rejects_oversized_input(app):
    port, _ = app
    assert call(port, "POST", "/api/rules/preview", {"rules": "x" * 4001})[0] == 400
    assert call(port, "POST", "/api/rules/preview", {"rules": 5})[0] == 400