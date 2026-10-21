"""Inputs from anywhere: link types, safety, document text, Drive folders, and honest failures."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest

from quillcut import inputs


@pytest.mark.parametrize("url,kind", [
    ("https://docs.google.com/document/d/1ZTRWvdFZhlf_l4whiYL1HOwWJw7Ec0rooguQFi-x1wg/edit?tab=t.0", "gdoc"),
    ("https://drive.google.com/drive/folders/1SKpOJVxksUGY7jHDzxdW6fjwfpKrq8bt", "gdrive_folder"),
    ("https://drive.google.com/file/d/1HQfqu6Xwsh_Q8zRmjN3AnQLbeweIJLWb/view?usp=sharing", "gdrive_file"),
    ("https://drive.google.com/open?id=1HQfqu6Xwsh_Q8zRmjN3AnQLbeweIJLWb", "gdrive_file"),
    ("https://example.com/media/interview.mp4", "direct"),
    ("https://example.com/brief.pdf", "direct"),
    ("https://www.youtube.com/watch?v=yJ4g4X-RlUQ", "video_site"),
    ("https://www.tiktok.com/t/ZPL8Qd84F/", "video_site"),
])
def test_links_are_classified(url, kind):
    assert inputs.classify(url) == kind


def test_drive_ids_come_out_of_every_link_shape():
    assert inputs.drive_id("https://drive.google.com/file/d/1HQfqu6Xwsh_Q8zRmjN3AnQLbeweIJLWb/view") == \
        "1HQfqu6Xwsh_Q8zRmjN3AnQLbeweIJLWb"
    assert inputs.drive_id("https://drive.google.com/drive/folders/1SKpOJVxksUGY7jHDzxdW6fjwfpKrq8bt?usp=x") == \
        "1SKpOJVxksUGY7jHDzxdW6fjwfpKrq8bt"
    assert inputs.drive_id("https://drive.google.com/open?id=abcdefghijk123") == "abcdefghijk123"


@pytest.mark.parametrize("url", ["ftp://example.com/x.mp4", "file:///C:/secret.txt", "not a link", "",
                                 "http://127.0.0.1:8765/api/status", "http://localhost/x.mp4",
                                 "http://192.168.1.10/cam.mp4", "http://10.0.0.5/x", "http://169.254.169.254/latest"])
def test_unsafe_or_local_links_are_refused(url):
    with pytest.raises(inputs.InputError):
        inputs.check_url(url)


# --- documents -----------------------------------------------------------------------------------------------

def _docx(paragraphs: list[str]) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = ('<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
           f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", xml)
    return buf.getvalue()


def _pdf(text: str) -> bytes:
    """A minimal single-page PDF with one line of text."""
    stream = f"BT /F1 18 Tf 40 700 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>stream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


def test_rules_come_out_of_word_pdf_html_and_text(tmp_path):
    (tmp_path / "r.docx").write_bytes(_docx(["Only clips about pricing", "No profanity"]))
    assert inputs.file_to_text(tmp_path / "r.docx").splitlines() == ["Only clips about pricing", "No profanity"]
    (tmp_path / "r.pdf").write_bytes(_pdf("No longer than 45 seconds"))
    assert "No longer than 45 seconds" in inputs.file_to_text(tmp_path / "r.pdf")
    (tmp_path / "r.html").write_text("<html><style>x{}</style><p>High energy</p><p>Square video</p></html>")
    assert inputs.file_to_text(tmp_path / "r.html").splitlines() == ["High energy", "Square video"]
    (tmp_path / "r.md").write_text("**Vertical 9:16**\n| keep it short |\n", encoding="utf-8")
    assert inputs.file_to_text(tmp_path / "r.md").splitlines() == ["Vertical 9:16", "keep it short"]


def test_unreadable_documents_say_so(tmp_path):
    (tmp_path / "bad.docx").write_bytes(b"not a zip")
    with pytest.raises(inputs.InputError, match="Word"):
        inputs.file_to_text(tmp_path / "bad.docx")
    (tmp_path / "x.exe").write_bytes(b"MZ")
    with pytest.raises(inputs.InputError, match="Rules can be read"):
        inputs.file_to_text(tmp_path / "x.exe")


def test_document_formatting_is_cleaned():
    raw = "\ufeff**THE MISSION**\n| :-: |\n\\*\\*Vertical 9:16\\*\\*&#10;  - Keep it under 60 seconds |"
    assert inputs.clean_rules_text(raw).splitlines() == ["THE MISSION", "Vertical 9:16", "- Keep it under 60 seconds"]


# --- downloads, with the network replaced by a fake -----------------------------------------------------------

def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_download_streams_a_file_and_uses_its_name(tmp_path):
    client = _client(lambda req: httpx.Response(200, content=b"\x00" * 5000, headers={
        "content-type": "video/mp4", "content-disposition": 'attachment; filename="Episode 1.mp4"'}))
    path = inputs.download("https://example.com/x", tmp_path, 10_000, client=client)
    assert path.name == "Episode 1.mp4" and path.stat().st_size == 5000


@pytest.mark.parametrize("page,message", [
    ("<title>Google Drive - Quota exceeded</title> Too many users have viewed or downloaded this file", "quota"),
    ("<a href='https://accounts.google.com/ServiceLogin'>Sign in</a>", "shared publicly"),
    ("<html>some other page</html>", "web page"),
])
def test_web_pages_instead_of_files_get_a_clear_reason(tmp_path, page, message):
    client = _client(lambda req: httpx.Response(200, text=page, headers={"content-type": "text/html"}))
    with pytest.raises(inputs.InputError, match=message):
        inputs.download("https://drive.usercontent.google.com/download?id=x", tmp_path, 10_000, client=client)


def test_oversized_and_missing_files_are_refused(tmp_path):
    big = _client(lambda req: httpx.Response(200, content=b"\x00" * 3000, headers={"content-type": "video/mp4"}))
    with pytest.raises(inputs.InputError, match="larger"):
        inputs.download("https://example.com/x.mp4", tmp_path, 1000, client=big)
    assert not list(tmp_path.glob("*"))                       # nothing half-written is left behind
    gone = _client(lambda req: httpx.Response(404))
    with pytest.raises(inputs.InputError, match="404"):
        inputs.download("https://example.com/x.mp4", tmp_path, 1000, client=gone)


FOLDER_HTML = """
<div class="flip-entry" id="entry-1AAAAAAAAAAAAAA"><a href="https://drive.google.com/file/d/1AAAAAAAAAAAAAA/view">
<div class="flip-entry-title">Trailer &amp; Teaser.mp4</div></a></div>
<div class="flip-entry" id="entry-1BBBBBBBBBBBBBB"><a href="https://drive.google.com/drive/folders/1BBBBBBBBBBBBBB">
<div class="flip-entry-title">Subfolder</div></a></div>
<div class="flip-entry" id="entry-1CCCCCCCCCCCCCC"><a href="https://drive.google.com/file/d/1CCCCCCCCCCCCCC/view">
<div class="flip-entry-title">Cover.jpg</div></a></div>
"""


def test_public_drive_folder_lists_files_and_skips_subfolders():
    client = _client(lambda req: httpx.Response(200, text=FOLDER_HTML, headers={"content-type": "text/html"}))
    entries = inputs.drive_folder_entries("1SKpOJVxksUGY7jHDzxdW6fjwfpKrq8bt", client=client)
    assert [e["name"] for e in entries] == ["Trailer & Teaser.mp4", "Cover.jpg"]


def test_private_drive_folder_is_explained():
    client = _client(lambda req: httpx.Response(404))
    with pytest.raises(inputs.InputError, match="shared publicly"):
        inputs.drive_folder_entries("x", client=client)


def test_a_google_doc_is_not_accepted_as_the_video(tmp_path):
    with pytest.raises(inputs.InputError, match="Rules"):
        inputs.fetch_media("https://docs.google.com/document/d/1ZTRWvdFZhlf_l4whiYL1HOwWJw7Ec0rooguQFi-x1wg/edit",
                           tmp_path, 1000)


def test_google_doc_rules_are_read_from_the_text_export(monkeypatch, tmp_path):
    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        return httpx.Response(200, text="\ufeffOnly clips about pricing\nNo profanity",
                              headers={"content-type": "text/plain; charset=utf-8"}, request=httpx.Request("GET", url))
    monkeypatch.setattr(inputs.httpx, "get", fake_get)
    monkeypatch.setattr(inputs, "check_url", lambda u: u)
    text, source = inputs.fetch_rules_text("https://docs.google.com/document/d/ABCDEFGHIJKLMNOP/edit", tmp_path)
    assert seen["url"].endswith("/document/d/ABCDEFGHIJKLMNOP/export?format=txt")
    assert text.splitlines() == ["Only clips about pricing", "No profanity"] and source == "Google Doc"


def test_private_google_doc_is_explained(monkeypatch, tmp_path):
    monkeypatch.setattr(inputs.httpx, "get", lambda url, **kw: httpx.Response(
        200, text="<html>Sign in</html>", headers={"content-type": "text/html"}, request=httpx.Request("GET", url)))
    monkeypatch.setattr(inputs, "check_url", lambda u: u)
    with pytest.raises(inputs.InputError, match="Anyone with the link"):
        inputs.fetch_rules_text("https://docs.google.com/document/d/ABCDEFGHIJKLMNOP/edit", tmp_path)
