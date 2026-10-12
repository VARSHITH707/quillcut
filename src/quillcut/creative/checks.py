"""Deterministic evaluators. Each returns a CategoryResult from measurable evidence only.

Subjective judgement (is the hook curious? is the b-roll relevant?) is NOT guessed here: those
components are listed in `unavailable` until a model fills them in (see score.evaluate).
Every finding carries evidence, timestamps, a confidence and, where a safe edit exists, an action.
"""
from __future__ import annotations

import re

import numpy as np

from .facts import (DANGLING, HOP_S, LEADING_CONNECTIVE, PUNCT_RELIABLE, STOP_GAP, TERMINAL, ClipFacts,
                    content_tokens, end_candidates, gap_at, has_capitals, normalize, punctuation_share)
from .schema import CategoryResult, Finding, Severity

MODEL_COMPONENTS = {   # category -> the evidence only a model can supply (a missing model blocks approval)
    "pacing": "unnecessary_footage", "story": "standalone_context", "visual": "subject_and_broll",
    "ending": "thought_completeness", "fidelity": "claim_support"}


GRAB_WORDS = {"never", "first", "only", "biggest", "largest", "fastest", "cheapest", "secret", "why", "how",
              "nobody", "million", "billion", "thousand", "imagine", "everything", "worst", "best"}
FIGURE_WORDS = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
                "million", "billion"}


def _finding(f: ClipFacts, category: str, severity: Severity, issue: str, evidence: str, fix: str, *,
             action: str = "none", conf: float = 0.9, score: float | None = None, start: float | None = None,
             end: float | None = None, **params) -> Finding:
    pen = f.cc["penalty"][severity]
    return Finding(category=category, severity=severity, score=max(1.0, score if score is not None else 5 - pen),
                   issue=issue, evidence=evidence, recommended_fix=fix, edit_action=action, confidence=conf,
                   start_time=None if start is None else round(start, 2),
                   end_time=None if end is None else round(end, 2), params=params)


def category_score(f: ClipFacts, findings: list[Finding]) -> float:
    """5 minus the penalties of scoring findings, but never above the worst single finding's score."""
    counted = [x for x in findings if x.confidence >= f.cc["count_min_confidence"]]
    base = 5 - sum(f.cc["penalty"][x.severity] for x in counted)
    return round(max(1.0, min([base] + [x.score for x in counted])), 2)


def _done(f: ClipFacts, category: str, findings: list[Finding], note: str = "") -> CategoryResult:
    return CategoryResult(category=category, status="scored", score=category_score(f, findings),
                          findings=findings, note=note)


def _no_media(category: str) -> CategoryResult:
    return CategoryResult(category=category, status="unavailable",
                          unavailable=["rendered file missing: nothing to measure"])


def _quote(words: list[dict], n: int = 8) -> str:
    return " ".join(w["word"] for w in words[-n:])


# --------------------------------------------------------------------------------- ending

def ending(f: ClipFacts) -> CategoryResult:
    c, findings = f.cc["ending"], []
    last_span = f.body_spans[-1]
    if not f.words:
        return _done(f, "ending", findings, "no speech: ending judged on shot boundary and audio only")
    last = f.words[-1]
    sp = f.span_at(last["end"] - 0.001)
    seg = sp.seg
    gap = f.body_end - last["end"]
    src_words = f.src_words.get(seg.source, [])
    last_src_end = seg.src_in + (last["end"] - sp.start) * seg.speed
    pause, _, nxt = gap_at(src_words, last_src_end)           # how long the speaker stops after our last word
    terminal = bool(TERMINAL.search(last["word"]))
    dangling = re.sub(r"\W", "", last["word"]).lower() in DANGLING
    trusted = punctuation_share(src_words) >= PUNCT_RELIABLE    # can sentence ends be read from punctuation?
    cuts = [a for a, _ in f.shots.get(seg.source, [])]
    cand = end_candidates(src_words, last_src_end, 0.0, f.plan.sources[seg.source].duration, cuts,
                          trim_max=c["trim_max_s"], extend_max=c["extend_max_s"])
    where = dict(start=max(0.0, last_span.end - 4.0), end=last_span.end)
    common = dict(part=sp.part, src_out=seg.src_out, source=seg.source,
                  **{k: v for k, v in cand.items() if v is not None})
    fix, action = _end_fix(cand, last_src_end)
    runs_on = not terminal and (trusted or pause < STOP_GAP)    # trusted text: no full stop = not finished
    if runs_on:
        said = f"\"{_quote(f.words)}\" and the source continues {pause * 1000:.0f} ms later with \"{nxt['word'] if nxt else ''}\""
        if trusted:
            findings.append(_finding(f, "ending", "critical", "The clip ends mid-sentence.", said, fix,
                                     action=action, conf=0.95, score=1.0, **common, **where))
        else:
            findings.append(_finding(
                f, "ending", "major", "The clip stops while the speaker is still talking.",
                said + f"; the transcript has little punctuation ({punctuation_share(src_words):.1%}), so a sentence end "
                "cannot be confirmed", fix + " Verify by ear.", action=action, conf=0.65, score=2.5,
                **common, **where))
    elif dangling and not terminal:
        findings.append(_finding(f, "ending", "major", "The final sentence looks unfinished.",
                                 f"ends on a dangling word: \"{_quote(f.words, 4)}\" (gap {gap:.2f}s)", fix,
                                 action=action, conf=0.8, score=2.5, **common, **where))
    elif trusted and not terminal:
        findings.append(_finding(f, "ending", "minor", "The final line has no closing punctuation.",
                                 f"\"{_quote(f.words, 4)}\" then a {pause:.2f}s pause in the source",
                                 "Check the last line reads as finished.", conf=0.45, score=4.0, **where))
    if gap > c["dead_air_max_s"]:
        findings.append(_finding(f, "ending", "minor", f"{gap:.1f}s of nothing after the last word.",
                                 f"last word ends {last['end']:.1f}s, body ends {f.body_end:.1f}s",
                                 "Trim the tail to just after the last word.", action="trim_end", conf=0.85,
                                 score=4.0, part=last_span.part, trim=round(seg.src_out - (gap - 0.4), 3),
                                 src_out=seg.src_out, source=seg.source, **where))
    if f.env is not None and not runs_on:
        tail = f.env_slice(f.body_end - 0.12, f.body_end)
        speech = f.env_slice(max(0, last["start"] - 6), last["end"])
        if tail.size and speech.size and tail.mean() > np.median(speech) - c["abrupt_db"] and gap < 0.15:
            # a loud tail after a finished sentence is a nuisance; mid-thought it is a defect
            findings.append(_finding(f, "ending", "minor" if terminal else "major", "Audio stops abruptly.",
                                     f"final 120 ms is {tail.mean():.0f} dBFS vs speech median {np.median(speech):.0f} dBFS",
                                     "End on a pause or add a short fade.", conf=0.7,
                                     score=4.0 if terminal else 3.0, **where))
    return _done(f, "ending", findings)


def _end_fix(cand: dict, src_out: float) -> tuple[str, str]:
    """Recommendation text and the nearer of the two safe actions (repair re-checks the limits)."""
    t, e = cand.get("trim"), cand.get("extend")
    if t is None and e is None:
        return "No complete-sentence boundary nearby: pick another segment or review manually.", "reselect"
    if e is not None and (t is None or e - src_out <= src_out - t):
        return f"Extend the clip to the end of the sentence at {e:.2f}s (source).", "extend_end"
    return f"Trim back to the last complete sentence at {t:.2f}s (source).", "trim_end"


# --------------------------------------------------------------------------------- hook

def hook(f: ClipFacts) -> CategoryResult:
    c, findings = f.cc["hook"], []
    cards = [x for x in f.plan.text_cards if x.kind == "hook"]
    if not cards:
        findings.append(_finding(f, "hook", "major", "No on-screen hook text.",
                                 "the plan has no hook card", "Add a factual headline from the transcript.",
                                 action="manual", conf=0.9, score=3.0, start=0, end=3))
    else:
        h = cards[0]
        if h.start > 0.3:
            findings.append(_finding(f, "hook", "major", "The hook text appears late.",
                                     f"first hook card starts at {h.start:.1f}s", "Start the headline at 0s.",
                                     conf=0.9, start=0, end=h.start))
        if len(h.text.split()) > c["max_words"]:
            findings.append(_finding(f, "hook", "minor", "The hook is wordy.",
                                     f"{len(h.text.split())} words (limit {c['max_words']})",
                                     "Shorten the headline.", conf=0.8, start=h.start, end=h.end))
        want = content_tokens(h.text)
        have = content_tokens(f.transcript) | content_tokens(" ".join(f.policy.get("brand_terms", [])))
        if f.words and want:
            share = len(want & have) / len(want)
            if share < c["min_overlap"]:                    # a headline the narration never backs up is misleading
                major = share < c["major_overlap"]
                findings.append(_finding(
                    f, "hook", "major" if major else "minor", "The headline does not match what is said.",
                    f"only {len(want & have)}/{len(want)} headline terms appear in the narration; never said: "
                    f"{', '.join(sorted(want - have)) or 'none'}", "Headline should reflect the spoken content.",
                    conf=0.7 if major else 0.55, score=3.0 if major else 4.0, start=h.start, end=h.end))
    first = f.words[0]["start"] if f.words else None
    if first is not None and first > c["speech_start_max_s"]:
        motion = f.opening_motion
        if motion is not None and motion < c["min_opening_motion"]:
            findings.append(_finding(f, "hook", "major", "Slow, static opening.",
                                     f"first word at {first:.1f}s and almost no motion (diff {motion:.3f})",
                                     "Start at the first spoken word or a moving shot.", action="trim_start",
                                     conf=0.7, score=3.0, start=0, end=first, part=f.body_spans[0].part,
                                     source=f.body_spans[0].seg.source,
                                     src_start=round(f.clip_to_source(first)[1] - 0.05, 3)))
    if f.words:                                              # does the narration open on a point worth stopping for?
        opening = [w["word"] for w in f.words[:12]]
        text = " ".join(opening)
        if not (any(ch.isdigit() for ch in text) or "?" in text or set(normalize(text)) & GRAB_WORDS):
            findings.append(_finding(f, "hook", "minor", "The narration opens without a hook.",
                                     f'first words: "{text}" contain no figure, question or strong claim',
                                     "Open on the line that states the point, if the book allows it.", action="manual",
                                     conf=0.5, score=4.0, start=0, end=4))
    return _done(f, "hook", findings)


# --------------------------------------------------------------------------------- pacing

def pacing(f: ClipFacts) -> CategoryResult:
    c, findings = f.cc["pacing"], []
    if not f.words:
        return _done(f, "pacing", findings, "no speech")
    changes = f.visual_changes()
    cuts = {s.seg.source: [a for a, _ in f.shots.get(s.seg.source, [])] for s in f.body_spans}
    pauses = [(b["start"] - a["end"], a, b) for a, b in zip(f.words, f.words[1:])
              if b["start"] - a["end"] > c["pause_max_s"]]
    for gap, a, b in pauses:
        if gap < c["pause_major_s"]:
            continue                                         # ordinary sentence gaps are summed up below
        key, s_a = f.clip_to_source(a["end"])
        _, s_b = f.clip_to_source(b["start"])
        on_cuts = (any(abs(s_a + 0.05 - k) < 0.1 for k in cuts.get(key, []))
                   and any(abs(s_b - k) < 0.1 for k in cuts.get(key, [])))
        findings.append(_finding(
            f, "pacing", "major", f"{gap:.1f}s pause in the narration.",
            f"no speech between \"{a['word']}\" ({a['end']:.1f}s) and \"{b['word']}\" ({b['start']:.1f}s)",
            "Cut the dead air" if on_cuts else "Pause sits inside one shot: removing it would be a jump cut; review.",
            action="cut_span" if on_cuts else "manual", conf=0.85, start=a["end"], end=b["start"],
            span_src=[round(s_a, 3), round(s_b, 3)], source=key, jump_cut=not on_cuts))
    body = f.body_end
    minors = [p for p in pauses if p[0] < c["pause_major_s"]]       # the long ones are reported on their own
    minor_total = sum(g for g, _, _ in minors)
    if len(minors) >= 4 or minor_total / body > c["pause_total_frac"]:
        first = ", ".join(f"{a['end']:.1f}-{b['start']:.1f}s" for _, a, b in minors[:3])
        findings.append(_finding(f, "pacing", "minor", "The narration has many long pauses.",
                                 f"{len(minors)} gaps of {c['pause_max_s']}-{c['pause_major_s']}s, {minor_total:.1f}s of "
                                 f"{body:.0f}s ({minor_total / body:.0%}); e.g. {first}", "Tighten the longest pauses.",
                                 action="manual", conf=0.7, score=4.0))
    sents = [s for s in f.sentences if len(s.words) >= 4]
    for j, t in enumerate(sents):                            # each repeated sentence is reported once, against its first use
        for s in sents[:j]:
            a, b = content_tokens(s.text), content_tokens(t.text)
            if a and b and len(a & b) / len(a | b) >= c["repeat_jaccard"]:
                key, s0 = f.clip_to_source(t.start)
                _, s1 = f.clip_to_source(t.end)
                findings.append(_finding(
                    f, "pacing", "major", "Dialogue repeats.",
                    f"\"{s.text}\" ({s.start:.1f}s) is said again at {t.start:.1f}s: \"{t.text}\"",
                    "Remove the second occurrence.", action="cut_span", conf=0.8, start=t.start, end=t.end,
                    span_src=[round(s0 - 0.05, 3), round(s1 + 0.12, 3)], source=key))
                break
    win = c["slow_window_s"]
    t = 0.0
    while t + win <= body:
        n = sum(1 for w in f.words if t <= w["start"] < t + win)
        if n / win < c["slow_wps"] and not any(t < x < t + win for x in changes[1:]) and n > 0:
            findings.append(_finding(f, "pacing", "minor", "A slow stretch.",
                                     f"{n / win:.1f} words/s with no visual change from {t:.0f}s to {t + win:.0f}s",
                                     "Tighten or cover with relevant b-roll.", action="manual", conf=0.6,
                                     score=4.0, start=t, end=t + win))
            t += win
        else:
            t += 2.0
    return _done(f, "pacing", findings)


# --------------------------------------------------------------------------------- rhythm

def rhythm(f: ClipFacts) -> CategoryResult:
    c, findings = f.cc["rhythm"], []
    marks = f.visual_changes() + [f.body_end]
    gaps = [(b - a, a, b) for a, b in zip(marks, marks[1:])]
    worst, a, b = max(gaps)
    if worst >= c["static_minor_s"]:
        sev = "major" if worst >= c["static_major_s"] else "minor"
        findings.append(_finding(f, "rhythm", sev, "Long stretch with no visual change.",
                                 f"{worst:.1f}s between visual changes ({a:.1f}s to {b:.1f}s); "
                                 f"{len(marks) - 2} changes in {f.body_end:.0f}s",
                                 "Add relevant b-roll or an approved cutaway; none is invented automatically.",
                                 action="manual", conf=0.85, start=a, end=b))
    return _done(f, "rhythm", findings)


# --------------------------------------------------------------------------------- captions

def captions(f: ClipFacts) -> CategoryResult:
    cards = [x for x in f.plan.text_cards if x.kind == "speech"]
    spoken = [w for w in f.words if w["word"].strip()]
    if not cards and not spoken:
        return _done(f, "captions", [], "no speech")
    if not cards:
        return _done(f, "captions", [_finding(f, "captions", "major", "Speech without captions.",
                                              f"{len(spoken)} spoken words and no caption cards", "Enable captions.",
                                              action="manual", score=2.0)])
    findings = (_caption_coverage(f, cards, spoken) + _caption_readability(f, cards)
                + _final_caption(f, cards, spoken) + _overlapping_cards(f) + _caption_face_overlap(f, cards))
    res = _done(f, "captions", findings)
    if f.faces is None:                                     # captions-over-faces could not be checked
        res.unavailable.append("face_overlap: " + ("no rendered file" if f.video is None else "no face detector installed"))
    return res


def _caption_coverage(f: ClipFacts, cards, spoken) -> list[Finding]:
    """Every spoken word should sit under some card. Low-confidence words are dropped on purpose by the planner."""
    c = f.cc["captions"]
    shown = [w for x in cards for w in x.words]

    def captioned(w: dict) -> bool:        # joined tokens ("out" + "-of") still cover their fragments
        mid = (w["start"] + w["end"]) / 2
        return any(s.start - 0.05 <= mid <= s.end + 0.05 for s in shown)

    missing = [w for w in spoken if not captioned(w)]
    cover = 1 - len(missing) / max(1, len(spoken))
    if cover >= c["min_coverage"]:
        return []
    major = cover < c["major_coverage"]
    examples = ", ".join(f"\"{w['word']}\" {w['start']:.1f}s (p={w.get('probability', 1):.2f})" for w in missing[:6])
    return [_finding(f, "captions", "major" if major else "minor", "Some spoken words have no caption.",
                     f"{cover:.0%} of {len(spoken)} spoken words captioned; missing e.g. {examples}",
                     "Low-confidence words are dropped on purpose; verify them and add word_fixes.", action="manual",
                     conf=0.85, score=3.0 if major else 4.0)]


def _caption_readability(f: ClipFacts, cards) -> list[Finding]:
    """One finding per kind of problem (flashing, too fast, too small, late), not one per card."""
    c = f.cc["captions"]
    duration = lambda x: x.end - x.start                   # noqa: E731
    groups = (
        ("Captions flash by.", [x for x in cards if duration(x) < c["min_card_s"]],
         lambda x: f"\"{x.text}\" {duration(x):.2f}s", "Merge short cards."),
        ("Captions are hard to read in time.",
         [x for x in cards if duration(x) >= c["min_card_s"] and len(x.text) / duration(x) > c["max_cps"]],
         lambda x: f"\"{x.text}\" {len(x.text) / duration(x):.0f} chars/s (limit {c['max_cps']})", "Split fast cards."),
        ("Small caption text.", [x for x in cards if x.font_size < c["min_font"]],
         lambda x: f"{x.font_size}px on \"{x.text}\"", "Shorten the line."),
        ("Caption timing is off.",
         [x for x in cards if x.words and abs(x.words[0].start - x.start) > c["start_tol_s"]],
         lambda x: f"card {x.start:.2f}s, first word {x.words[0].start:.2f}s", "Re-time the cards."))
    return [_finding(f, "captions", "minor", issue,
                     f"{len(group)} of {len(cards)} cards; e.g. " + "; ".join(detail(x) for x in group[:3]), fix,
                     conf=0.7, score=4.0, start=group[0].start, end=group[0].end)
            for issue, group, detail, fix in groups if group]


def _final_caption(f: ClipFacts, cards, spoken) -> list[Finding]:
    last = spoken[-1]
    final = max(cards, key=lambda x: x.end)
    last_shown = any(abs(w.start - last["start"]) < 0.06 or w.start <= last["start"] <= w.end for w in final.words)
    if final.end >= last["end"] - 0.05 and last_shown:
        return []
    return [_finding(f, "captions", "major", "The final caption is incomplete.",
                     f"last spoken word \"{last['word']}\" ({last['start']:.1f}-{last['end']:.1f}s) vs final card "
                     f"\"{final.text}\" ending {final.end:.1f}s",
                     "Extend or re-cut the last card so the final words are shown.", action="manual", conf=0.85,
                     score=2.5, start=last["start"], end=last["end"])]


def _overlapping_cards(f: ClipFacts) -> list[Finding]:
    """An overlay (question, watermark) must never sit on top of another kind of card at the same time."""
    out = []
    for i, a in enumerate(f.plan.text_cards):
        for b in f.plan.text_cards[i + 1:]:
            both = min(a.end, b.end) - max(a.start, b.start)
            if a.kind == b.kind or both < 0.2:
                continue
            ox = min(a.bbox_norm[2], b.bbox_norm[2]) - max(a.bbox_norm[0], b.bbox_norm[0])
            oy = min(a.bbox_norm[3], b.bbox_norm[3]) - max(a.bbox_norm[1], b.bbox_norm[1])
            if ox > 0 and oy > 0:
                out.append(_finding(f, "captions", "major", "Two text cards overlap.",
                                    f"\"{a.text}\" and \"{b.text}\" share screen area {max(a.start, b.start):.1f}-"
                                    f"{min(a.end, b.end):.1f}s", "Move or shorten one card.", conf=0.9, score=3.0,
                                    start=max(a.start, b.start), end=min(a.end, b.end)))
    return out


def _caption_face_overlap(f: ClipFacts, cards) -> list[Finding]:
    if f.faces is None or not cards:
        return []                    # evaluate() lists 'face_overlap' as unavailable when there is no detector
    c, bad = f.cc["captions"], []
    for i, boxes in enumerate(f.faces):
        t = i / 2
        card = next((x for x in cards if x.start <= t < x.end), None)
        for x0, y0, x1, y1 in boxes:
            if card is None:
                continue
            bx0, by0, bx1, by1 = card.bbox_norm
            ix, iy = min(x1, bx1) - max(x0, bx0), min(y1, by1) - max(y0, by0)
            if ix > 0 and iy > 0 and ix * iy / ((x1 - x0) * (y1 - y0)) > c["face_overlap"]:
                bad.append((t, card.text))
    if not bad:
        return []
    return [_finding(f, "captions", "major", "Captions cover a face.",
                     f"{len(bad)} sampled frames, first at {bad[0][0]:.1f}s under \"{bad[0][1]}\"",
                     "Move the caption lower.", action="move_captions", conf=0.8, score=3.0,
                     start=bad[0][0], end=bad[-1][0] + 0.5)]


# --------------------------------------------------------------------------------- audio

def audio(f: ClipFacts) -> CategoryResult:
    if f.env is None:
        return _no_media("audio")
    c = f.cc["audio"]
    findings = (_silences(f, c) + _joins(f, c) + _speech_quality(f, c) + _clipped_first_word(f)
                + _av_sync(f, c))
    return _done(f, "audio", findings)


def _silences(f: ClipFacts, c: dict) -> list[Finding]:
    """Stretches of near-silence inside the body, found as runs of quiet analysis hops."""
    quiet = np.concatenate(([False], f.env < c["silence_db"], [False]))
    edges = np.flatnonzero(np.diff(quiet.astype(int)))          # alternating run starts and ends
    runs = [(i * HOP_S, j * HOP_S) for i, j in zip(edges[::2], edges[1::2])]
    return [_finding(f, "audio", "major", "Dead silence.",
                     f"{b - a:.1f}s below {c['silence_db']:.0f} dBFS ({a:.1f}-{b:.1f}s)", "Cut or cover it.",
                     action="manual", conf=0.9, score=3.0, start=a, end=b)
            for a, b in runs if b - a >= c["silence_max_s"] and b <= f.body_end + 0.05]


def _joins(f: ClipFacts, c: dict) -> list[Finding]:
    """What happens to the sound where one part ends and the next begins."""
    out = []
    for prev, nxt in zip(f.body_spans, f.body_spans[1:]):
        if prev.seg.source == nxt.seg.source and abs(prev.seg.src_out - nxt.seg.src_in) < 0.02:
            continue                                         # same audio continues through a b-roll swap
        pause, before, after = gap_at(f.src_words.get(prev.seg.source, []), prev.seg.src_out)
        if pause < STOP_GAP and after is not None:
            out.append(_finding(f, "audio", "major", "Audio is cut mid-speech at a join.",
                                f"join at {prev.end:.1f}s: only {pause * 1000:.0f} ms between "
                                f"\"{before['word'] if before else ''}\" and \"{after['word']}\" in the source",
                                "Move the cut to a pause.", action="manual", conf=0.75, score=3.0,
                                start=prev.end - 0.3, end=prev.end + 0.3))
        a, b = f.env_slice(prev.end - 0.1, prev.end), f.env_slice(prev.end, prev.end + 0.1)
        if a.size and b.size and abs(a.mean() - b.mean()) > c["chop_db"] \
                and min(a.mean(), b.mean()) < c["silence_db"] + 15:
            out.append(_finding(f, "audio", "minor", "Hard level jump at a join.",
                                f"{a.mean():.0f} -> {b.mean():.0f} dBFS at {prev.end:.1f}s", "Crossfade the audio.",
                                conf=0.6, score=4.0, start=prev.end - 0.1, end=prev.end + 0.1))
    return out


def _speech_quality(f: ClipFacts, c: dict) -> list[Finding]:
    out = []
    probs = [w.get("probability", 1.0) for w in f.words]
    if probs and float(np.mean(probs)) < c["min_word_prob"]:
        out.append(_finding(f, "audio", "minor", "Speech may be hard to understand.",
                            f"mean transcription confidence {np.mean(probs):.2f} (limit {c['min_word_prob']})",
                            "Check the narration by ear.", conf=0.6, score=4.0))
    levels = [float(np.median(f.env_slice(s.start, s.end))) for s in f.sentences if s.end - s.start >= 1.0]
    if len(levels) >= 4:
        spread = float(np.percentile(levels, 90) - np.percentile(levels, 10))
        if spread > c["runs_spread_db"]:                     # advisory: loudness is already gated by technical QC
            out.append(_finding(f, "audio", "minor", "Uneven speech volume.",
                                f"typical sentence levels differ by {spread:.0f} dB (10th-90th percentile)",
                                "Review the mix.", conf=0.45, score=4.0))
    return out


def _clipped_first_word(f: ClipFacts) -> list[Finding]:
    if not f.words or f.words[0]["start"] >= 0.06:
        return []
    key, t = f.clip_to_source(0.0)
    through = [w for w in f.src_words.get(key, []) if w["end"] > t - 0.03 and w["start"] < t - 0.02]
    if not through:
        return []
    w = through[-1]
    return [_finding(f, "audio", "major", "The first word is clipped.",
                     f"\"{w['word']}\" runs through the in-point at source {t:.2f}s (ends {w['end'] - t:+.2f}s from it)",
                     "Start just before the first full word.", action="trim_start", conf=0.9, score=3.0, start=0,
                     end=0.5, part=f.body_spans[0].part, source=key,
                     src_start=round(f.clip_to_source(f.words[0]["start"])[1] - 0.05, 3))]


def _av_sync(f: ClipFacts, c: dict) -> list[Finding]:
    from .. import media
    streams = media.probe(f.video).get("streams", [])
    dur = {s["codec_type"]: float(s.get("duration") or 0) for s in streams if s.get("codec_type") in ("video", "audio")}
    if len(dur) == 2 and abs(dur["video"] - dur["audio"]) > c["sync_tol_s"]:
        return [_finding(f, "audio", "major", "Audio and video lengths differ.",
                         f"video {dur['video']:.2f}s vs audio {dur['audio']:.2f}s", "Re-render.", conf=0.9, score=3.0)]
    return []


# --------------------------------------------------------------------------------- visual

def visual(f: ClipFacts) -> CategoryResult:
    if f.frames is None or not len(f.frames):
        return _no_media("visual")
    c, findings = f.cc["visual"], []
    frames = f.frames
    luma = frames.mean(axis=(1, 2, 3))
    dark = np.where(luma < c["dark_luma"])[0]
    if len(dark) > 0.25 * len(frames):
        black = len(dark) > 0.6 * len(frames)               # mostly black is a broken render, not a dim grade
        findings.append(_finding(f, "visual", "major" if black else "minor",
                                 "The picture is almost black." if black else "The picture is very dark.",
                                 f"{len(dark)}/{len(frames)} sampled frames under luma {c['dark_luma']}",
                                 "Check the source and the render." if black else "Raise the grade.",
                                 conf=0.9 if black else 0.7, score=2.0 if black else 4.0,
                                 start=dark[0] / 2, end=dark[-1] / 2))
    try:
        import cv2
        lap = float(np.median([cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for fr in frames]))
        if lap < c["blur_laplacian"]:
            findings.append(_finding(f, "visual", "minor", "The picture is soft.",
                                     f"median sharpness {lap:.0f} (limit {c['blur_laplacian']:.0f})",
                                     "Source may be low-resolution.", conf=0.5, score=4.0))
    except ImportError:
        pass
    soft = [s for s in f.body_spans if _upscale(f, s.seg) > c["max_upscale_soft"]]
    if soft:
        findings.append(_finding(f, "visual", "minor", "Footage is enlarged a lot.",
                                 f"{sum(s.end - s.start for s in soft):.0f}s upscaled over {c['max_upscale_soft']}x",
                                 "Prefer a wider or higher-resolution shot.", conf=0.6, score=4.0,
                                 start=soft[0].start, end=soft[-1].end))
    findings += _repeated_shots(f, c)
    findings += _subject_framing(f)
    return _done(f, "visual", findings)


def _upscale(f: ClipFacts, seg) -> float:
    ref = f.plan.sources[seg.video_source or seg.source]
    cw, ch = ref.content_box[0], ref.content_box[1]
    if seg.layout == "fit_blur":                      # whole frame fitted to the width; only the blurred backdrop is enlarged
        return 1080 / cw
    return 1920 / ch if cw / ch >= 9 / 16 else 1080 / cw


def _repeated_shots(f: ClipFacts, c: dict) -> list[Finding]:
    small = f.frames[:, ::16, ::16].astype(np.float32).mean(axis=3)
    n = len(small)
    for i in range(n - 1):
        for j in range(i + 12, n - 1):                     # at least 6 s apart
            if (np.abs(small[i] - small[j]).mean() < c["repeat_diff"]
                    and np.abs(small[i + 1] - small[j + 1]).mean() < c["repeat_diff"]):
                return [_finding(f, "visual", "minor", "The same shot appears twice.",
                                 f"frames at {i / 2:.1f}s and {j / 2:.1f}s are near-identical",
                                 "Use a different cutaway for the second use.", action="manual", conf=0.6,
                                 score=4.0, start=j / 2, end=j / 2 + 1)]
    return []


def _subject_framing(f: ClipFacts) -> list[Finding]:
    if f.faces is None:
        return []
    edge = [any(y0 < 0.01 or x0 < 0.01 or x1 > 0.99 for x0, y0, x1, y1 in boxes) for boxes in f.faces]
    for i in range(len(edge) - 1):
        if edge[i] and edge[i + 1]:                           # one sampled frame is noise; two in a row is framing
            return [_finding(f, "visual", "major", "A face is cut off by the frame.",
                             f"face box touches the frame edge from {i / 2:.1f}s", "Reframe (fit_blur or shift crop).",
                             action="manual", conf=0.6, score=3.0, start=i / 2, end=i / 2 + 1.0)]
    return []


# --------------------------------------------------------------------------------- CTA

def cta(f: ClipFacts) -> CategoryResult:
    mode = f.policy.get("cta", "off")
    if mode == "off":
        return CategoryResult(category="cta", status="not_applicable", note="campaign policy: no CTA/overlay")
    c, findings = f.cc["cta"], []
    cards = [x for x in f.plan.text_cards if x.kind == "label" and x.start >= f.body_end - 6
             and x.text.strip().endswith("?")]
    if not cards:
        qs = f.policy.get("cta_questions", [])
        findings.append(_finding(
            f, "cta", "minor", "No question or call to action.", "no closing question overlay in the plan",
            ("Add a relevant question" + (f": \"{qs[0]}\"" if qs else "")
             + (" (automatic: policy 'auto')" if mode == "auto" else " (policy 'suggest': not applied automatically)")),
            action="add_question_overlay" if mode == "auto" else "none", conf=0.75, score=4.0,
            start=f.body_end - c["seconds"], end=f.body_end, question=qs[0] if qs else None))
    for card in cards:
        if card.end - card.start < c["min_seconds"]:
            findings.append(_finding(f, "cta", "major", "The CTA is on screen too briefly.",
                                     f"\"{card.text}\" shows {card.end - card.start:.1f}s (needs {c['min_seconds']}s)",
                                     "Hold it longer.", action="retime_cta", conf=0.9, score=3.0,
                                     start=card.start, end=card.end))
        if card.font_size < c["min_font"] or len(card.text.split()) > c["max_words"]:
            findings.append(_finding(f, "cta", "minor", "The CTA is hard to read.",
                                     f"{card.font_size}px, {len(card.text.split())} words", "Shorten it.",
                                     conf=0.7, score=4.0, start=card.start, end=card.end))
        topic = content_tokens(f.transcript + " " + (f.plan.moment.get("topic") or "") + " " + f.hook_text)
        if not content_tokens(card.text) & topic:
            findings.append(_finding(f, "cta", "major", "The CTA does not match the content.",
                                     f"\"{card.text}\" shares no terms with the clip", "Use a question about the topic.",
                                     conf=0.7, score=3.0, start=card.start, end=card.end))
    return _done(f, "cta", findings)


# --------------------------------------------------------------------------------- fidelity

NEGATIONS = {"not", "never", "no", "without", "n't", "nothing", "none", "cannot"}


def fidelity(f: ClipFacts) -> CategoryResult:
    findings = []
    support = set(normalize(f.transcript + " " + (f.plan.moment.get("topic") or "") + " "
                            + " ".join(f.policy.get("facts_allow", []) + f.policy.get("brand_terms", []))))
    texts = [(x.kind, x.text) for x in f.plan.text_cards if x.kind in ("hook", "label")]
    for kind, text in texts:
        for tok in normalize(text):
            if tok.isdigit() and tok not in support:
                findings.append(_finding(
                    f, "fidelity", "major", "An overlay states a number that is never said.",
                    f"\"{text}\" contains {tok}, absent from the narration and topic", "Use a figure the speaker states,"
                    " or add it to the reel's creative.facts_allow once verified.", action="manual", conf=0.75,
                    score=3.0, start=0, end=3))
    for k, sp in enumerate(f.body_spans):
        prev = f.body_spans[k - 1].seg if k else None
        if prev and prev.source == sp.seg.source and abs(prev.src_out - sp.seg.src_in) < 0.02:
            continue                                         # continuous audio, not a real cut
        words = f.src_words.get(sp.seg.source, [])
        before = [w for w in words if sp.seg.src_in - 4 <= w["start"] < sp.seg.src_in - 0.05][-6:]
        if any(re.sub(r"\W", "", w["word"]).lower() in NEGATIONS for w in before):
            findings.append(_finding(f, "fidelity", "minor", "A part starts just after a negation.",
                                     f"source before the cut: \"{' '.join(w['word'] for w in before)}\"",
                                     "Check the meaning is not reversed by the cut.", action="manual", conf=0.5,
                                     score=4.0, start=sp.start, end=sp.start + 3))
    return _done(f, "fidelity", findings)


# --------------------------------------------------------------------------------- story

def story(f: ClipFacts) -> CategoryResult:
    findings = []
    if not f.words:
        return _done(f, "story", findings, "no speech")
    first = f.sentences[0]
    token = re.sub(r"\W", "", first.words[0]["word"]).lower()
    lead = f.body_spans[0]
    caps = has_capitals(f.src_words.get(lead.seg.source, []))     # an all-lowercase transcript says nothing
    if token in LEADING_CONNECTIVE or (caps and first.words[0]["word"][:1].islower() and token not in ("i", "iphone")):
        later = next((s for s in f.sentences[1:] if s.start - first.start < 12), None)
        findings.append(_finding(
            f, "story", "major", "The clip starts mid-thought.",
            f"opens with \"{' '.join(w['word'] for w in first.words[:6])}\" ({'connective' if token in LEADING_CONNECTIVE else 'lower-case'} start)",
            "Start at the next sentence." if later else "Start earlier, at the beginning of the sentence.",
            action="trim_start" if later else "manual", conf=0.75 if caps else 0.6, score=3.0, start=0, end=first.end,
            part=lead.part, source=lead.seg.source,
            src_start=round(f.clip_to_source(later.start)[1], 3) if later else None))
    want = content_tokens(f.hook_text)
    if want and len(want & content_tokens(f.transcript)) == 0 and f.words:
        findings.append(_finding(f, "story", "minor", "The headline's promise never appears in the narration.",
                                 f"none of {sorted(want)} is spoken", "Check the footage carries the payoff.",
                                 action="manual", conf=0.55, score=4.0))
    return _done(f, "story", findings)


# --------------------------------------------------------------------------------- loop

def loop(f: ClipFacts) -> CategoryResult:
    """Rewatch/share potential by transparent proxies, not by asking a small model for an opinion:
    a concrete figure to remember, a finished ending, and a closing brand beat."""
    has_fact = any(any(ch.isdigit() for ch in w["word"]) or w["word"].lower() in FIGURE_WORDS for w in f.words)
    clean_end = not any(x.severity in ("critical", "major") and x.confidence >= f.cc["count_min_confidence"]
                        for x in ending(f).findings)
    flags = {"a concrete figure is stated": has_fact, "the ending is finished": clean_end}
    if f.policy.get("closing_beat", True):            # books with no brand end card set creative.closing_beat: false
        flags["a closing brand beat exists"] = f.spans[-1].seg.id == "end"
    score = round(1 + 4 * sum(flags.values()) / len(flags), 2)
    findings = []
    if sum(flags.values()) < 2:
        missing = ", ".join(k for k, v in flags.items() if not v)
        findings.append(_finding(f, "loop", "minor", "Low rewatch/share potential.", f"missing: {missing}",
                                 "A stated figure and a finished ending make a clip easier to share.", conf=0.45,
                                 score=4.0))
    return CategoryResult(category="loop", status="scored", score=score, findings=findings,
                          note="proxy score: " + "; ".join(f"{k}={'yes' if v else 'no'}" for k, v in flags.items()))


EVALUATORS = {"hook": hook, "pacing": pacing, "story": story, "visual": visual, "audio": audio,
              "captions": captions, "ending": ending, "cta": cta, "rhythm": rhythm, "loop": loop,
              "fidelity": fidelity}
