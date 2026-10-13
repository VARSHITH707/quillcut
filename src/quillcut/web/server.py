"""Local web app for Quillcut: drag in a video, pick a style, download captioned reels.

Standard library only. It binds to 127.0.0.1, so it is reachable from this computer only.
"""
from __future__ import annotations

import io
import json
import logging
import mimetypes
import os
import queue
import re
import shutil
import threading
import time
import uuid
import webbrowser
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from .. import jobs as pipeline_jobs
from ..config import load_config, load_yaml
from ..formats import list_formats
from ..paths import CLIPPING, FORMATS
from . import copywriter

log = logging.getLogger("quillcut.web")

STATIC = Path(__file__).parent / "static"
WEB_ROOT = CLIPPING / "web"
UPLOADS = WEB_ROOT / "uploads"
JOBS = WEB_ROOT / "jobs"

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
RIGHTS = {"own_footage": "I own this footage", "client_owned": "My client owns it and I have permission",
          "client_licensed": "It is licensed to me for this use", "cc_verified": "It is Creative Commons and I verified the licence"}
MAX_UPLOAD = int(float(os.environ.get("QUILLCUT_MAX_UPLOAD_GB", "10")) * 1024 ** 3)
ID_RE = re.compile(r"^[0-9a-f]{12}$")
HEX_RE = re.compile(r"^[0-9A-Fa-f]{6}$")
FORMAT = "gaming"  # the native automatic format; delegated formats are listed but disabled
PRESET_NAMES = {"gaming_bold": "Bold gaming", "gaming_headline": "Headline", "brand_clean": "Clean brand"}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(name).name).strip(" .") or "video"
    return stem[:120]


# --- job store ---------------------------------------------------------------------------------------------

class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self.queue: queue.Queue[str] = queue.Queue()
        JOBS.mkdir(parents=True, exist_ok=True)
        for path in JOBS.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.get("status") in ("queued", "running"):  # the server stopped mid-run
                job.update(status="failed", error="The app was closed before this job finished. Please start it again.")
            self._jobs[job["id"]] = job

    def add(self, job: dict) -> None:
        with self._lock:
            self._jobs[job["id"]] = job
            self._save(job)

    def update(self, job_id: str, **fields) -> dict:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields)
            self._save(job)
            return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def all(self) -> list[dict]:
        with self._lock:
            return sorted((dict(j) for j in self._jobs.values()), key=lambda j: j["created"], reverse=True)

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    @staticmethod
    def _save(job: dict) -> None:
        folder = JOBS / job["id"]
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "job.json.tmp"
        tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        tmp.replace(folder / "job.json")


STORE: JobStore | None = None


def store() -> JobStore:
    global STORE
    if STORE is None:
        STORE = JobStore()
    return STORE


# --- reel post-processing ----------------------------------------------------------------------------------

def _clip_text(project_dir: Path, meta: dict) -> str:
    transcript = project_dir / "analysis" / "transcript.json"
    try:
        words = json.loads(transcript.read_text(encoding="utf-8")).get("words", [])
    except (OSError, ValueError):
        return ""
    ranges = [(r[1], r[2]) for r in meta.get("source_ranges", [])]
    return " ".join(w["word"] for w in words if any(lo <= w["start"] < hi for lo, hi in ranges))


def build_clips(job_id: str, results: list[dict], cfg: dict) -> list[dict]:
    llm = cfg.get("creative", {}).get("llm", {})
    clips = []
    for res in results:
        if res.get("rejected"):
            continue
        entry = {"clip_id": res["clip_id"], "duration": round(res.get("duration", 0), 1), "qc": res.get("qc"),
                 "failures": res.get("failures", []), "warnings": res.get("warnings", [])}
        published = res.get("published")
        if res.get("qc") == "PASS" and published and Path(published).exists():
            final = Path(res["final"])
            try:
                meta = json.loads((final.parent.parent / "metadata" / f"{res['clip_id']}.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            text = _clip_text(final.parent.parent, meta)
            entry.update(file=Path(published).name, hook=meta.get("hook_text"), topic=meta.get("topic"),
                         transcript=text[:600],
                         **copywriter.describe(meta.get("hook_text"), meta.get("topic"), text,
                                               url=llm.get("url", "http://localhost:11434"),
                                               model=llm.get("text_model", "qwen2.5:7b-instruct-q4_K_M")))
        clips.append(entry)
    return clips


def run_job(job_id: str) -> None:
    s = store()
    job = s.update(job_id, status="running", stage="starting", started=time.time())
    opts = job["options"]
    overrides = {"colors": opts["colors"], "speech": {"size": opts["caption_size"]},
                 "hook": {"size": opts["hook_size"]}}

    def on_progress(stage: str, done: int = 0, total: int = 0) -> None:
        s.update(job_id, stage=stage, done=done, total=total)

    try:
        results = pipeline_jobs.run_auto(
            Path(job["source"]), FORMAT, opts["rights"], opts["max_clips"], fast=opts["quality"] == "draft",
            caption_style=opts["preset"], style_overrides=overrides, publish_dir=JOBS / job_id / "reels",
            on_progress=on_progress)
        s.update(job_id, stage="writing_captions")
        clips = build_clips(job_id, results, load_config(FORMAT))
        passed = [c for c in clips if c.get("file")]
        note = None if passed else (next((r["rejected"] for r in results if r.get("rejected")), None)
                                   or "No reel passed the quality checks.")
        s.update(job_id, status="done", stage="finished", clips=clips, note=note, finished=time.time())
    except pipeline_jobs.SourceRejected as exc:
        s.update(job_id, status="failed", error=str(exc), finished=time.time())
    except Exception:  # noqa: BLE001 - the page needs a message, the log keeps the trace
        log.exception("job %s failed", job_id)
        s.update(job_id, status="failed", finished=time.time(),
                 error="Something went wrong while making the reels. Check the terminal for details.")


def worker() -> None:
    s = store()
    while True:
        job_id = s.queue.get()
        try:
            run_job(job_id)
        finally:
            s.queue.task_done()


# --- API ---------------------------------------------------------------------------------------------------

def presets() -> list[dict]:
    """Caption styles for the gaming format, the format's own default first."""
    default = load_config(FORMAT).get("editing", {}).get("caption_style")
    out = []
    for path in sorted((FORMATS / FORMAT / "styles").glob("*.yaml"), key=lambda p: (p.stem != default, p.stem)):
        style = load_yaml(path)
        key = path.stem
        out.append({"key": key, "name": PRESET_NAMES.get(key, key.replace("_", " ").title()),
                    "case": style.get("case", "natural"), "colors": style.get("colors", {}),
                    "speech_size": style.get("speech", {}).get("size", 78),
                    "hook_size": style.get("hook", {}).get("size", 84)})
    return out


def format_list() -> list[dict]:
    out = []
    for key in list_formats():
        conf = load_yaml(FORMATS / key / "config.yaml").get("format", {})
        native = conf.get("status") == "native"
        out.append({"key": key, "name": conf.get("name", key), "description": conf.get("description", ""),
                    "available": native and key == FORMAT,
                    "note": "" if native else "Needs OpenShorts running locally"})
    return out


def ollama_state() -> dict:
    cfg = load_config(FORMAT).get("creative", {}).get("llm", {})
    url = cfg.get("url", "http://localhost:11434")
    try:
        names = [m["name"] for m in httpx.get(f"{url}/api/tags", timeout=1.5).json().get("models", [])]
    except (httpx.HTTPError, ValueError, KeyError):
        return {"running": False, "model": cfg.get("text_model"), "has_model": False}
    model = cfg.get("text_model")
    return {"running": True, "model": model, "has_model": model in names}


def job_view(job: dict) -> dict:
    view = {k: job.get(k) for k in ("id", "filename", "status", "stage", "done", "total", "error", "note",
                                      "created", "started", "finished", "options")}
    clips = []
    for c in job.get("clips", []):
        c = dict(c)
        if c.get("file"):
            c["url"] = f"/media/{job['id']}/{c['file']}"
            c["download"] = f"/media/{job['id']}/{c['file']}?download=1"
        clips.append(c)
    view["clips"] = clips
    return view


def validate_options(body: dict, available: dict[str, dict]) -> dict:
    preset = body.get("preset")
    if preset not in available:
        raise ApiError(400, "Pick a caption style.")
    colors = body.get("colors") or {}
    base = {k: available[preset]["colors"].get(k, "FFFFFF") for k in ("base", "active", "stroke")}
    for key in base:
        value = str(colors.get(key, base[key])).lstrip("#")
        if not HEX_RE.match(value):
            raise ApiError(400, f"The {key} color must be a hex color like FFD400.")
        base[key] = value.upper()
    try:
        size = int(body.get("caption_size", available[preset]["speech_size"]))
        max_clips = int(body.get("max_clips", 3))
    except (TypeError, ValueError):
        raise ApiError(400, "Caption size and reel count must be numbers.") from None
    if not 40 <= size <= 140:
        raise ApiError(400, "Caption size must be between 40 and 140.")
    if not 1 <= max_clips <= 8:
        raise ApiError(400, "Choose between 1 and 8 reels.")
    quality = body.get("quality", "draft")
    if quality not in ("draft", "high"):
        raise ApiError(400, "Quality must be draft or high.")
    rights = body.get("rights")
    if body.get("rights_confirmed") is not True or rights not in RIGHTS:
        raise ApiError(400, "Please confirm you have the right to use this video.")
    return {"preset": preset, "colors": base, "caption_size": size, "hook_size": round(size * 1.08),
            "max_clips": max_clips, "quality": quality, "rights": rights}


class Handler(BaseHTTPRequestHandler):
    server_version = "Quillcut"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quiet
        log.debug(fmt, *args)

    # -- plumbing
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        origin = self.headers.get("Origin")
        if host not in ("127.0.0.1", "localhost", "::1"):
            return False  # blocks DNS-rebinding
        return origin is None or urlparse(origin).hostname in ("127.0.0.1", "localhost", "::1")

    def _json(self, data, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _drain(self) -> None:
        """Read a small unread request body so an error reply isn't lost to a connection reset."""
        if self._body_done:
            return
        self._body_done = True
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        if 0 < length <= 1 << 20:
            self.rfile.read(length)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 65536:
            raise ApiError(413, "Request too large.")
        self._body_done = True
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            raise ApiError(400, "That request was not valid JSON.") from None
        if not isinstance(data, dict):
            raise ApiError(400, "That request was not valid JSON.")
        return data

    def _dispatch(self, method: str) -> None:
        self._body_done = False
        if not self._host_ok():
            self.close_connection = True
            self._drain()
            return self._json({"error": "Forbidden"}, 403)
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if method == "GET" and not path.startswith(("/api/", "/media/")):
                return self._static(path)
            self._route(method, path, parse_qs(url.query))
        except ApiError as exc:
            self.close_connection = True  # an unread request body must not leak into the next request
            self._drain()
            self._json({"error": exc.message}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:  # noqa: BLE001
            log.exception("request failed: %s %s", method, path)
            self.close_connection = True
            self._json({"error": "Unexpected server error."}, 500)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    # -- routes
    def _route(self, method: str, path: str, query: dict) -> None:
        s = store()
        if method == "GET" and path == "/api/status":
            return self._json({"ollama": ollama_state(), "formats": format_list(), "presets": presets(),
                               "rights": RIGHTS, "max_upload_gb": round(MAX_UPLOAD / 1024 ** 3, 1),
                               "extensions": sorted(VIDEO_EXT)})
        if method == "POST" and path == "/api/upload":
            return self._upload(query)
        if method == "POST" and path == "/api/jobs":
            return self._create_job()
        if method == "GET" and path == "/api/jobs":
            return self._json({"jobs": [job_view(j) for j in s.all()]})
        m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})(/zip)?", path)
        if m:
            job = s.get(m.group(1))
            if not job:
                raise ApiError(404, "That job was not found.")
            if method == "DELETE" and not m.group(2):
                if job["status"] == "running":
                    raise ApiError(409, "That job is still running.")
                s.remove(job["id"])
                shutil.rmtree(JOBS / job["id"], ignore_errors=True)
                return self._json({"ok": True})
            if method == "GET" and m.group(2):
                return self._zip(job)
            if method == "GET":
                return self._json(job_view(job))
        m = re.fullmatch(r"/media/([0-9a-f]{12})/(c_\d{2}\.mp4)", path)
        if m and method == "GET":
            return self._media(m.group(1), m.group(2), "download" in query)
        raise ApiError(404, "Not found.")

    def _upload(self, query: dict) -> None:
        name = safe_name((query.get("name") or [""])[0])
        if Path(name).suffix.lower() not in VIDEO_EXT:
            raise ApiError(415, f"Use one of: {', '.join(sorted(VIDEO_EXT))}")
        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError:
            raise ApiError(411, "The upload size is missing.") from None
        if length <= 0:
            raise ApiError(400, "That file is empty.")
        if length > MAX_UPLOAD:
            raise ApiError(413, f"That file is larger than {MAX_UPLOAD / 1024 ** 3:.0f} GB.")
        upload_id = uuid.uuid4().hex[:12]
        folder = UPLOADS / upload_id
        folder.mkdir(parents=True)
        target = folder / name
        remaining = length
        self._body_done = True  # the stream is consumed below; never drain it again
        try:
            with open(target, "wb") as fh:
                while remaining:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        raise ApiError(400, "The upload was interrupted.")
                    fh.write(chunk)
                    remaining -= len(chunk)
            from .. import media
            try:
                info = media.video_info(target)
            except Exception:  # noqa: BLE001
                raise ApiError(422, "That file doesn't look like a video we can read.") from None
            if info["duration"] < 20:
                raise ApiError(422, f"That video is only {info['duration']:.0f} seconds. Use one of at least 20 seconds.")
        except ApiError:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        short = min(info["width"], info["height"])
        warning = ("Low resolution: reels will look soft. 720p or higher works best." if short < 720 else None)
        self._json({"upload_id": upload_id, "name": name, "size": length, "duration": round(info["duration"], 1),
                    "width": info["width"], "height": info["height"], "warning": warning}, 201)

    def _create_job(self) -> None:
        body = self._read_json()
        upload_id = str(body.get("upload_id", ""))
        if not ID_RE.match(upload_id) or not (UPLOADS / upload_id).is_dir():
            raise ApiError(400, "Upload a video first.")
        files = [p for p in (UPLOADS / upload_id).iterdir() if p.suffix.lower() in VIDEO_EXT]
        if not files:
            raise ApiError(400, "Upload a video first.")
        options = validate_options(body, {p["key"]: p for p in presets()})
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "filename": files[0].name, "source": str(files[0]), "status": "queued",
               "stage": "queued", "done": 0, "total": options["max_clips"], "options": options, "clips": [],
               "created": time.time()}
        store().add(job)
        store().queue.put(job_id)
        self._json(job_view(job), 201)

    def _zip(self, job: dict) -> None:
        clips = [c for c in job.get("clips", []) if c.get("file")]
        if not clips:
            raise ApiError(404, "This job has no reels to download.")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            lines = []
            for c in clips:
                zf.write(JOBS / job["id"] / "reels" / c["file"], c["file"])
                lines.append(f"{c['file']}\n{c.get('caption', '')}\n{' '.join(c.get('hashtags', []))}\n")
            zf.writestr("captions.txt", "\n".join(lines))
        data = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f'attachment; filename="quillcut-reels-{job["id"]}.zip"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _media(self, job_id: str, name: str, download: bool) -> None:
        path = JOBS / job_id / "reels" / name
        if not path.is_file():
            raise ApiError(404, "That reel is no longer available.")
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if not m or not (m.group(1) or m.group(2)):
                raise ApiError(416, "Bad range.")
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{job_id}_{name}"')
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left:
                block = fh.read(min(1 << 20, left))
                if not block:
                    break
                self.wfile.write(block)
                left -= len(block)

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (STATIC / rel).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self._json({"error": "Not found"}, 404)
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self'; "
                         "script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)


class LocalServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second process bind a port that is already in use, so two copies of
    # the app would both "start". Turn it off: a busy port must fail loudly.
    allow_reuse_address = False
    daemon_threads = True


def make_server(port: int = 8765) -> ThreadingHTTPServer:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    httpd = LocalServer(("127.0.0.1", port), Handler)  # bind first so a busy port fails before any thread starts
    threading.Thread(target=worker, daemon=True, name="quillcut-worker").start()
    return httpd


def serve(port: int = 8765, open_browser: bool = True) -> None:
    try:
        server = make_server(port)
    except OSError as exc:
        raise SystemExit(f"Port {port} is not available ({exc.strerror}). Quillcut may already be running: "
                         f"open http://127.0.0.1:{port}/ or start with --port {port + 1}.") from None
    url = f"http://127.0.0.1:{port}/"
    print(f"Quillcut is running at {url}  (press Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
