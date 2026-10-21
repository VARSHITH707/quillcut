"""Fetch a video (or a folder of videos and photos) from a link, then register each file as an upload."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from .. import inputs, media

MIN_SECONDS = 20
MIN_MIX_SECONDS = 1        # mixed footage is a cutaway, so a short clip is fine


def run_fetch(job: dict, update, uploads_dir: Path, max_bytes: int) -> None:
    """Download into a private scratch folder, then move each usable file into its own upload folder.
    job["role"] == "mix" accepts short clips (cutaways); otherwise a video must be a full source."""
    min_seconds = MIN_MIX_SECONDS if job.get("role") == "mix" else MIN_SECONDS
    scratch = uploads_dir / f"fetch-{job['id']}"
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        files = inputs.fetch_media(job["url"], scratch, max_bytes,
                                   on_progress=lambda text: update(stage="downloading", progress_text=text))
        update(stage="checking", progress_text="Checking the downloaded files")
        videos, photos, problems = [], [], []
        for path in files:
            ext = path.suffix.lower()
            upload_id = uuid.uuid4().hex[:12]
            folder = uploads_dir / upload_id
            folder.mkdir(parents=True)
            target = folder / path.name
            shutil.move(str(path), target)
            if ext in inputs.IMAGE_EXT:
                photos.append({"upload_id": upload_id, "name": target.name, "size": target.stat().st_size})
                continue
            try:
                info = media.video_info(target)
            except Exception:  # noqa: BLE001 - one bad file must not sink the rest
                problems.append(f"{target.name}: not a readable video")
                shutil.rmtree(folder, ignore_errors=True)
                continue
            if info["duration"] < min_seconds:
                problems.append(f"{target.name}: only {info['duration']:.0f} s long (needs {min_seconds} s)")
                shutil.rmtree(folder, ignore_errors=True)
                continue
            short = min(info["width"], info["height"])
            videos.append({"upload_id": upload_id, "name": target.name, "size": target.stat().st_size,
                           "duration": round(info["duration"], 1), "width": info["width"], "height": info["height"],
                           "warning": "Low resolution: reels will look soft. 720p or higher works best."
                           if short < 720 else None})
        if not videos and not photos:
            raise inputs.InputError("Nothing usable came from that link. " + " ".join(problems[:2]))
        update(status="done", stage="finished", uploads=videos, photos=photos, problems=problems,
               progress_text=f"{len(videos)} video{'s' if len(videos) != 1 else ''} ready")
    except inputs.InputError as exc:
        update(status="failed", error=str(exc))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
