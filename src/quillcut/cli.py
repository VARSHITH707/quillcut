"""quillcut command line."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from .paths import CACHE, FORMATS, apply_cache_env


def _range(text: str | None):
    if not text:
        return None
    lo, hi = text.split("-")
    return float(lo), float(hi)


def cmd_doctor(_args) -> int:
    import httpx
    from . import media
    from .config import load_config
    from .formats import list_formats, load_style
    from .storage import disk_free_gb

    rows, ok = [], True

    def row(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        rows.append(f"{'OK  ' if good else 'FAIL'} {name:<22} {detail}")

    row("ffmpeg", shutil.which("ffmpeg"), shutil.which("ffmpeg") or "not on PATH")
    enc = media.run(["ffmpeg", "-hide_banner", "-encoders"], check=False).stdout
    row("libx264", "libx264" in enc, "final encoder")
    rows.append(f"{'OK  ' if 'h264_nvenc' in enc else 'WARN'} {'h264_nvenc':<22} previews only")
    try:
        import torch
        cuda = torch.cuda.is_available()
        free, total = torch.cuda.mem_get_info() if cuda else (0, 0)
        row("CUDA", cuda, f"{torch.cuda.get_device_name(0)} {free / 1e9:.1f}/{total / 1e9:.1f} GB free" if cuda else "none")
    except Exception as exc:
        row("CUDA", False, str(exc))
    free = disk_free_gb(FORMATS)
    row("disk I:", free >= 5, f"{free:.1f} GB free")
    try:
        tags = httpx.get("http://localhost:11434/api/tags", timeout=5).json()
        names = [m["name"] for m in tags.get("models", [])]
        rows.append(f"{'OK  ' if names else 'WARN'} {'ollama':<22} {', '.join(names) or 'no models'}")
    except Exception:
        rows.append(f"WARN {'ollama':<22} not reachable (needed from P1)")
    from .creative.llm import Ollama
    st = Ollama(load_config("gaming")["creative"]["llm"]).status()
    rows.append(f"{'OK  ' if st['text'] else 'WARN'} {'creative text model':<22} {st['text'] or st['reason']}")
    rows.append(f"{'OK  ' if st['vision'] else 'WARN'} {'creative vision model':<22} "
                f"{st['vision'] or 'none installed (visual judgement unavailable)'}")
    try:
        httpx.get("http://127.0.0.1:8000/api/status/doctor", timeout=5)
        rows.append(f"OK   {'openshorts api':<22} reachable (delegated formats)")
    except Exception:
        rows.append(f"WARN {'openshorts api':<22} not running (delegated formats unavailable)")
    for key in list_formats():
        cfg = load_config(key)
        if cfg["format"].get("status") == "native":
            style = load_style(key, cfg["editing"]["caption_style"])
            for script, font in style["fonts"].items():
                row(f"font {script}", Path(font["file"]).exists(), font["file"])
    rows.append(f"OK   {'formats':<22} {', '.join(list_formats())}")
    rows.append(f"OK   {'cache':<22} {CACHE}")
    print("\n".join(rows))
    return 0 if ok else 1


def cmd_formats(_args) -> int:
    from .config import load_config
    from .formats import list_formats
    for key in list_formats():
        cfg = load_config(key)
        f = cfg["format"]
        print(f"{key:<16} {f.get('status', '?'):<10} {f.get('name', '')}")
    return 0


def cmd_scripts(args) -> int:
    from .scripts import load_templates
    for key, tpl in load_templates(args.format).items():
        srcs = "+".join(tpl.get("sources", ["a"]))
        print(f"{key:<20} {tpl['duration']:>4}s  sources {srcs:<4} {tpl['name']}")
        if args.verbose:
            for role, desc in (tpl.get("roles") or {}).items():
                print(f"{'':<22}{role}: {desc}")
    return 0


def _print(result) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_run(args) -> int:
    from .jobs import run_auto
    results = run_auto(Path(args.video), args.format, args.rights, args.max_clips, fast=args.fast)
    _print(results)
    return 0 if any(r.get("qc") == "PASS" or r.get("final") for r in results) else 1


def cmd_script(args) -> int:
    from .jobs import run_script
    sources = {"a": {"video": args.a, "range": _range(args.a_range), "peak": args.a_peak}}
    if args.b:
        sources["b"] = {"video": args.b, "range": _range(args.b_range), "peak": args.b_peak}
    result = run_script(args.name, args.format, sources, args.rights, text=args.text,
                        layout=args.layout, fast=args.fast)
    _print(result)
    return 0 if result.get("qc") == "PASS" else 1


def cmd_reels(args) -> int:
    from .jobs import run_book
    results = run_book(Path(args.book), set(args.only.split(",")) if args.only else None, fast=args.fast)
    for r in results:
        print(f"{r['clip_id']:<24} {r.get('qc', 'REJECTED'):<6} {r.get('duration', '')!s:<7} "
              f"{r.get('published') or r.get('rejected') or '; '.join(r.get('failures', []))}")
    return 0 if all(r.get("qc") == "PASS" for r in results) else 1


def cmd_creative(args) -> int:
    from .config import load_config, load_yaml
    from .creative import loop
    from .creative.llm import Models
    book = Path(args.book)
    only = set(args.only.split(",")) if args.only else None
    if args.show:
        report = loop.stored_report(book, args.show)
        if report is None:
            print(f"no stored creative report for {args.show}; run with --review or without flags first")
            return 1
        print(loop.render_text(report))
        return 0
    if args.review:
        models = Models.from_config(load_config(load_yaml(book).get("format", "gaming")))
        for r in loop.evaluate_existing(book, only, models):
            print(f"{r['clip_id']:<28} {r.get('decision', 'SKIPPED'):<14} {r.get('overall', '')!s:<6} "
                  f"{'; '.join(r.get('reasons', [])) or r.get('skipped', '')}")
        return 0
    from .jobs import run_book
    results = run_book(book, only, fast=args.fast, creative=True, repair=not args.no_repair)
    for r in results:
        print(f"{r['clip_id']:<28} {r.get('decision', 'REJECTED'):<14} {r.get('before', '')!s:<5}->{r.get('overall', '')!s:<5} "
              f"retries {r.get('retries', '-')}  {r.get('published') or '; '.join(r.get('reasons', [])) or r.get('rejected', '')}")
    return 0 if all(r.get("decision") == "APPROVED" for r in results) else 1


def cmd_explain(args) -> int:
    from .storage import read_json
    project = Path(args.project)
    if not project.is_absolute():
        project = next(FORMATS.glob(f"*/projects/{args.project}"), project)
    plans = sorted((project / "edit_specs").glob(f"{args.clip}.v*.editplan.json"))
    if not plans:
        print(f"no plan for {args.clip} in {project}")
        return 1
    plan = read_json(plans[-1])
    qc = read_json(project / "qc" / f"{args.clip}.json", [])
    print(f"{plan['clip_id']}  script={plan['script']}  mode={plan['mode']}  "
          f"rights={plan['rights'].get('status')}")
    m = plan["moment"]
    print(f"moment: {m.get('type')} peak {m.get('t_peak')}s intensity {m.get('peak_intensity')} - {m.get('rationale')}")
    t = 0.0
    for seg in plan["segments"]:
        dur = (seg["src_out"] - seg["src_in"]) / seg["speed"] + seg.get("hold_after", 0)
        extra = f" speed {seg['speed']}" if seg["speed"] != 1 else ""
        extra += f" hold {seg['hold_after']}s" if seg.get("hold_after") else ""
        print(f"  {t:5.1f}-{t + dur:5.1f}s  {seg['source']} {seg['src_in']:.2f}-{seg['src_out']:.2f} "
              f"{seg['layout']}{extra}  | {seg['rationale']}")
        t += dur
    for card in plan["text_cards"]:
        print(f"  text {card['start']:.1f}-{card['end']:.1f}s [{card['kind']}] {card['text']!r}")
    for line in plan["rationale"]:
        print(f"  note: {line}")
    if plan["placeholders_used"]:
        print(f"  WARNING placeholder values (not measured): {', '.join(plan['placeholders_used'])}")
    for c in qc:
        if c["status"] != "PASS":
            print(f"  QC {c['status']}: {c['name']} - {c['message']}")
    return 0


def cmd_batch(args) -> int:
    from .jobs import run_auto
    folder = Path(args.folder)
    code = 0
    for video in sorted(p for p in folder.iterdir() if p.suffix.lower() in (".mp4", ".mkv", ".mov", ".webm")):
        try:
            _print({"video": video.name, "clips": run_auto(video, None, None, fast=args.fast)})
        except Exception as exc:              # one bad video never stops the batch
            logging.exception("batch: %s failed", video.name)
            _print({"video": video.name, "error": str(exc)})
            code = 1
    return code


def cmd_clip(args) -> int:
    """Conversation-aware clipping: complete answers, per-reel durations, full edit."""
    from . import reelplan
    from .conversation_job import run
    try:
        targets = [float(t) for t in args.targets.split(",") if t.strip()]
    except ValueError:
        print("--targets must be seconds separated by commas, for example 35,60,90")
        return 2
    if not targets:
        print("--targets needs at least one duration")
        return 2
    options = reelplan.PlanOptions(intensity=args.intensity, aspect=args.aspect, cta=args.cta)
    out = run(Path(args.video), rights=args.rights, targets=targets, caption_style=args.style, options=options,
              strict=args.strict, include_question=args.include_question, fast=args.fast,
              use_llm=not args.no_llm, verify=not args.no_verify,
              publish_dir=Path(args.out) if args.out else None, profile=args.content_style)
    _print(out)
    return 0 if all(r.get("qc") == "PASS" or r.get("rejected") for r in out["results"]) else 1


def cmd_web(args) -> int:
    from .web.server import serve
    serve(args.port, open_browser=not args.no_browser)
    return 0


def main(argv=None) -> int:
    apply_cache_env()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="quillcut")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("formats").set_defaults(fn=cmd_formats)

    s = sub.add_parser("scripts", help="list script templates")
    s.add_argument("--format", default="gaming")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_scripts)

    r = sub.add_parser("run", help="automatic clips from one video")
    r.add_argument("video")
    r.add_argument("--format")
    r.add_argument("--rights", help="own_footage | client_owned | client_licensed | cc_verified")
    r.add_argument("--max-clips", type=int)
    r.add_argument("--fast", action="store_true", help="veryfast encode (tests, drafts)")
    r.set_defaults(fn=cmd_run)

    sc = sub.add_parser("script", help="one clip from a named script template")
    sc.add_argument("name")
    sc.add_argument("--format", default="gaming")
    sc.add_argument("--a", required=True, help="source A video")
    sc.add_argument("--a-range", help="seconds, e.g. 120-300")
    sc.add_argument("--a-peak", type=float, help="force the payoff instant in A (seconds)")
    sc.add_argument("--b", help="source B video (comparison scripts)")
    sc.add_argument("--b-range")
    sc.add_argument("--b-peak", type=float)
    sc.add_argument("--text", help="override the on-screen hook text")
    sc.add_argument("--layout", choices=["center_crop", "wide_fill"])
    sc.add_argument("--rights")
    sc.add_argument("--fast", action="store_true")
    sc.set_defaults(fn=cmd_script)

    rb = sub.add_parser("reels", help="render a reel book (hand-directed reels, YAML)")
    rb.add_argument("book")
    rb.add_argument("--only", help="comma-separated reel ids")
    rb.add_argument("--fast", action="store_true")
    rb.set_defaults(fn=cmd_reels)

    cr = sub.add_parser("creative", help="creative-quality pass on a reel book: evaluate, repair, re-render, decide")
    cr.add_argument("book")
    cr.add_argument("--only", help="comma-separated reel ids")
    cr.add_argument("--review", action="store_true", help="score existing finals, read-only, no rendering")
    cr.add_argument("--show", metavar="REEL", help="print the stored report for one reel (no model, no render)")
    cr.add_argument("--no-repair", action="store_true", help="evaluate the first render only")
    cr.add_argument("--fast", action="store_true")
    cr.set_defaults(fn=cmd_creative)

    e = sub.add_parser("explain", help="why a clip looks the way it does")
    e.add_argument("project")
    e.add_argument("clip")
    e.set_defaults(fn=cmd_explain)

    b = sub.add_parser("batch", help="run every video in a format input folder")
    b.add_argument("folder")
    b.add_argument("--fast", action="store_true")
    b.set_defaults(fn=cmd_batch)

    cl = sub.add_parser("clip", help="conversation-aware reels: complete answers, per-reel durations, full edit")
    cl.add_argument("video")
    cl.add_argument("--targets", default="45", help="target seconds per reel, comma separated, e.g. 35,60,90")
    cl.add_argument("--rights", help="own_footage | client_owned | client_licensed | cc_verified")
    cl.add_argument("--intensity", choices=["clean", "dynamic", "high_energy"], default="dynamic")
    cl.add_argument("--aspect", choices=["9:16", "1:1", "16:9"], default="9:16")
    cl.add_argument("--style", default=None, help="caption style, e.g. gaming_bold, brand_clean")
    cl.add_argument("--content-style", dest="content_style", default=None,
                    choices=["educational", "storytelling", "motivational", "podcast", "interview", "debate",
                             "business"])
    cl.add_argument("--include-question", dest="include_question", choices=["auto", "yes", "no"], default="auto")
    cl.add_argument("--strict", action="store_true",
                    help="never run past the target; report the conflict instead of keeping the full answer")
    cl.add_argument("--cta", default=None, help="closing line shown over the last seconds")
    cl.add_argument("--out", default=None, help="folder for the finished reels")
    cl.add_argument("--fast", action="store_true", help="veryfast encode (drafts)")
    cl.add_argument("--no-llm", dest="no_llm", action="store_true", help="skip the local-model second opinion")
    cl.add_argument("--no-verify", dest="no_verify", action="store_true", help="skip re-listening to each reel")
    cl.set_defaults(fn=cmd_clip)

    w = sub.add_parser("web", help="open the drag-and-drop app in your browser (this computer only)")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    w.set_defaults(fn=cmd_web)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
