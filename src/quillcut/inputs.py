"""Inputs from anywhere: uploaded files, Google Drive / Docs links, video sites and direct links.

Two kinds of input:
  * media  - videos (and photos) to clip from: a file, a Google Drive file or folder link, a YouTube / TikTok /
             Instagram / Vimeo / X link (anything yt-dlp supports), or a direct link to a file.
  * rules  - text to follow: typed, a .txt / .md / .docx / .pdf / .html file, a Google Docs link, or a link to
             one of those files.

Only links that work without signing in are supported: "anyone with the link" on Google Drive, public videos
elsewhere. A private file gives a plain message telling the person to share it or download it themselves.
The server never fetches private network addresses, so a link cannot be used to reach other machines.
"""
from __future__ import annotations

import html as html_lib
import ipaddress
import re
import socket
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import httpx

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
DOC_EXT = {".txt", ".md", ".docx", ".pdf", ".html", ".htm"}
MAX_DOC_BYTES = 20 * 1024 ** 2
MAX_RULES_CHARS = 20000
MAX_FOLDER_FILES = 25
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Quillcut/1.0"}


class InputError(Exception):
    """A message that can be shown to the person as-is."""


# --------------------------------------------------------------------------------------------- classifying

def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def drive_id(url: str) -> str | None:
    """The file or folder id in any common Google Drive / Docs link shape."""
    m = re.search(r"/(?:file|document|presentation|spreadsheets)/d/([\w-]{10,})", url) or \
        re.search(r"/folders/([\w-]{10,})", url)
    if m:
        return m.group(1)
    q = parse_qs(urlparse(url).query)
    return q.get("id", [None])[0]


def classify(url: str) -> str:
    """gdoc | gdrive_folder | gdrive_file | direct | video_site"""
    host = _host(url)
    path = urlparse(url).path.lower()
    if host == "docs.google.com" and "/document/" in path:
        return "gdoc"
    if host in ("drive.google.com", "drive.usercontent.google.com", "docs.google.com"):
        return "gdrive_folder" if "/folders/" in path else "gdrive_file"
    if Path(unquote(path)).suffix.lower() in VIDEO_EXT | IMAGE_EXT | DOC_EXT:
        return "direct"
    return "video_site"


def check_url(url: str) -> str:
    """Only http(s) to public internet addresses."""
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise InputError("Paste a full link starting with https://")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise InputError(f"Can't reach {parsed.hostname}. Check the link and your internet connection.") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise InputError("Links to private or local network addresses are not allowed.")
    return url


# --------------------------------------------------------------------------------------------- text

def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n", raw)
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", raw))
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines() if line.strip())


def docx_to_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", "replace")
    except (zipfile.BadZipFile, KeyError):
        raise InputError("That Word file could not be read. Save it as .docx and try again.") from None
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab/>", " ", xml)
    return "\n".join(line.strip() for line in html_lib.unescape(re.sub(r"<[^>]+>", "", xml)).splitlines()
                     if line.strip())


def pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:200])
    except Exception:  # noqa: BLE001 - any parser failure becomes one plain message
        raise InputError("That PDF could not be read.") from None
    if not text.strip():
        raise InputError("That PDF has no readable text (it may be a scanned image).")
    return text


def clean_rules_text(text: str) -> str:
    """Strip document formatting (markdown, table pipes, escapes) so each rule reads as plain words."""
    text = text.replace("﻿", "")
    text = re.sub(r"\\([\\#*\[\]_|>-])", r"\1", text)
    text = re.sub(r"[*_`]{1,3}", "", text)
    text = re.sub(r"^\s*\|?[\s:|-]*\|\s*$", "", text, flags=re.M)        # table rule lines
    text = text.replace("|", "\n")
    text = re.sub(r"&#10;", "\n", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    out = "\n".join(line for line in lines if line)
    return out[:MAX_RULES_CHARS]


def file_to_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in DOC_EXT:
        raise InputError(f"Rules can be read from {', '.join(sorted(DOC_EXT))} files.")
    if path.stat().st_size > MAX_DOC_BYTES:
        raise InputError("That document is larger than 20 MB.")
    if ext == ".docx":
        text = docx_to_text(path)
    elif ext == ".pdf":
        text = pdf_to_text(path)
    elif ext in (".html", ".htm"):
        text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return clean_rules_text(text)


# --------------------------------------------------------------------------------------------- downloads

def _filename(resp: httpx.Response, fallback: str) -> str:
    cd = resp.headers.get("content-disposition", "")
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="([^"]+)"', cd)
    name = unquote(m.group(1)) if m else fallback
    name = re.sub(r"[^\w.\- ()\[\]&]", "_", Path(name).name).strip(" .") or fallback
    return name[:120]


def _explain_html(text: str) -> str:
    low = text.lower()
    if "quota exceeded" in low or "too many users have viewed or downloaded" in low:
        return ("Google is blocking downloads of this file right now (download quota exceeded). "
                "Try again later, or open the link and download it yourself, then upload it.")
    if "sign in" in low or "accounts.google.com" in low or "request access" in low:
        return ("This file isn't shared publicly. Set sharing to “Anyone with the link”, "
                "or download it yourself and upload it.")
    return "That link returned a web page, not a file. Use a direct share link or upload the file."


def download(url: str, dest: Path, max_bytes: int, fallback_name: str = "download",
             client: httpx.Client | None = None) -> Path:
    """Stream one file to `dest`, refusing web pages and anything over `max_bytes`."""
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=60, headers=UA)
    try:
        with client.stream("GET", url) as resp:
            if resp.status_code == 404:
                raise InputError("That file was not found (404). Check the link.")
            if resp.status_code in (401, 403):
                raise InputError("Access denied. The file isn't shared publicly.")
            if resp.status_code >= 400:
                raise InputError(f"The server refused the download ({resp.status_code}).")
            if "text/html" in resp.headers.get("content-type", ""):
                raise InputError(_explain_html(resp.read().decode("utf-8", "replace")[:20000]))
            length = int(resp.headers.get("content-length") or 0)
            if length and length > max_bytes:
                raise InputError(f"That file is larger than {max_bytes / 1024 ** 3:.1f} GB.")
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / _filename(resp, fallback_name)
            got = 0
            with open(target, "wb") as fh:
                for chunk in resp.iter_bytes(1 << 20):
                    got += len(chunk)
                    if got > max_bytes:
                        fh.close()
                        target.unlink(missing_ok=True)
                        raise InputError(f"That file is larger than {max_bytes / 1024 ** 3:.1f} GB.")
                    fh.write(chunk)
            return target
    except httpx.HTTPError as exc:
        raise InputError(f"The download failed: {exc.__class__.__name__}.") from None
    finally:
        if own:
            client.close()


def drive_folder_entries(folder_id: str, client: httpx.Client | None = None) -> list[dict]:
    """Files in a publicly shared Drive folder: [{"id", "name"}] (subfolders are skipped)."""
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=30, headers=UA)
    try:
        resp = client.get(f"https://drive.google.com/embeddedfolderview?id={folder_id}")
    except httpx.HTTPError as exc:
        raise InputError(f"Couldn't open the folder: {exc.__class__.__name__}.") from None
    finally:
        if own:
            client.close()
    if resp.status_code != 200:
        raise InputError("That folder isn't shared publicly. Set it to “Anyone with the link”.")
    entries = []
    for fid, title in re.findall(r'id="entry-([\w-]+)".*?class="flip-entry-title">([^<]*)<', resp.text, flags=re.S):
        if f"/drive/folders/{fid}" in resp.text:
            continue
        entries.append({"id": fid, "name": html_lib.unescape(title).strip() or fid})
    if not entries:
        raise InputError("That folder is empty or isn't shared publicly.")
    return entries[:MAX_FOLDER_FILES]


def drive_file_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def video_site_download(url: str, dest: Path, max_bytes: int) -> Path:
    """YouTube, TikTok, Instagram, Vimeo, X and the other sites yt-dlp supports. One video, never a playlist."""
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError:
        raise InputError("Video-site links need yt-dlp. Run: pip install yt-dlp") from None
    dest.mkdir(parents=True, exist_ok=True)
    opts = {"outtmpl": str(dest / "%(title).80B [%(id)s].%(ext)s"), "noplaylist": True, "quiet": True,
            "no_warnings": True, "max_filesize": max_bytes, "merge_output_format": "mp4", "restrictfilenames": False,
            "format": "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b"}
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    except DownloadError as exc:
        msg = str(exc)
        if "Private video" in msg or "Sign in" in msg or "login" in msg.lower():
            raise InputError("That video is private or needs a login. Download it yourself and upload it.") from None
        if "Unsupported URL" in msg:
            raise InputError("That link isn't a video page Quillcut can read. Upload the file instead.") from None
        raise InputError("The video couldn't be downloaded from that link.") from None
    if not path.exists():
        found = sorted(dest.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise InputError("The video couldn't be downloaded from that link.")
        path = found[-1]
    return path


# --------------------------------------------------------------------------------------------- entry points

def fetch_media(url: str, dest: Path, max_bytes: int, on_progress=None) -> list[Path]:
    """Download the video(s) or photo(s) behind a link into `dest`. Returns the saved files."""
    url = check_url(url)
    kind = classify(url)
    if kind == "gdoc":
        raise InputError("That's a Google Doc. Paste it under Rules, not as the video.")
    if kind == "gdrive_folder":
        entries = drive_folder_entries(drive_id(url) or "")
        out, problems = [], []
        for i, e in enumerate(entries, start=1):
            if on_progress:
                on_progress(f"Downloading {i} of {len(entries)}: {e['name']}")
            try:
                out.append(download(drive_file_url(e["id"]), dest / e["id"], max_bytes, e["name"]))
            except InputError as exc:
                problems.append(f"{e['name']}: {exc}")
        media = [p for p in out if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT]
        if not media:
            raise InputError("No video or photo could be downloaded from that folder. " + " ".join(problems[:2]))
        return media
    if kind == "gdrive_file":
        fid = drive_id(url)
        if not fid:
            raise InputError("That Google Drive link doesn't contain a file id.")
        return [download(drive_file_url(fid), dest, max_bytes, "drive-file")]
    if kind == "direct":
        return [download(url, dest, max_bytes, Path(unquote(urlparse(url).path)).name or "download")]
    if on_progress:
        on_progress("Downloading the video")
    return [video_site_download(url, dest, max_bytes)]


def fetch_rules_text(url: str, work: Path) -> tuple[str, str]:
    """Rules text from a link. Returns (text, short description of the source)."""
    url = check_url(url)
    kind = classify(url)
    if kind == "gdoc":
        fid = drive_id(url)
        try:
            resp = httpx.get(f"https://docs.google.com/document/d/{fid}/export?format=txt", follow_redirects=True,
                             timeout=30, headers=UA)
        except httpx.HTTPError as exc:
            raise InputError(f"Couldn't open the document: {exc.__class__.__name__}.") from None
        if resp.status_code != 200 or "text/plain" not in resp.headers.get("content-type", ""):
            raise InputError("That Google Doc isn't shared publicly. Set it to “Anyone with the link”, "
                             "or download it as .docx or .pdf and load the file.")
        return clean_rules_text(resp.text), "Google Doc"
    if kind == "gdrive_file":
        path = download(drive_file_url(drive_id(url) or ""), work, MAX_DOC_BYTES, "rules")
        return file_to_text(path), f"Google Drive file {path.name}"
    if kind == "direct":
        path = download(url, work, MAX_DOC_BYTES, Path(unquote(urlparse(url).path)).name or "rules")
        return file_to_text(path), path.name
    try:                                                     # any other page: read its visible text
        resp = httpx.get(url, follow_redirects=True, timeout=30, headers=UA)
    except httpx.HTTPError as exc:
        raise InputError(f"Couldn't open that page: {exc.__class__.__name__}.") from None
    if resp.status_code >= 400 or "text/html" not in resp.headers.get("content-type", ""):
        raise InputError("That link doesn't lead to a readable page or document.")
    return clean_rules_text(html_to_text(resp.text)), _host(url)
