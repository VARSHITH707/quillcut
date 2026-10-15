"""Conversation-aware clipping: completeness, target vs actual duration, distinctness, honesty."""
from __future__ import annotations

import pytest

from quillcut import conversation as cv

WORD_S, GAP_S = 0.33, 0.06


def say(t: float, text: str):
    """Timestamped words for one spoken sentence starting at t. Returns (words, end_time)."""
    words = []
    for token in text.split():
        words.append({"word": token, "start": round(t, 3), "end": round(t + WORD_S, 3), "probability": 0.95})
        t += WORD_S + GAP_S
    return words, t - GAP_S


def talk(blocks, t0: float = 0.0):
    """blocks: [(pause_before, sentence_text)] -> words, end."""
    words, t = [], t0
    for pause, text in blocks:
        t += pause
        w, t = say(t, text)
        words += w
    return words, t


def topic(nouns: list[str], n: int) -> list[tuple[float, str]]:
    """n sentences of ~9 words that all use the same topic vocabulary (high lexical cohesion)."""
    out = []
    for i in range(n):
        a, b, c, d = (nouns[(i + k) % len(nouns)] for k in range(4))
        out.append((0.25, f"The {a} approach to {b} improved {c} and {d} every single week."))
    return out


ML = ["pipeline", "training", "dataset", "model", "latency", "inference", "accuracy", "feature"]
FOOD = ["kitchen", "recipe", "seasoning", "dough", "oven", "butter", "flavor", "pastry"]
MONEY = ["budget", "savings", "invoice", "revenue", "margin", "pricing", "forecast", "payroll"]
TRAVEL = ["airport", "passport", "itinerary", "hostel", "luggage", "visa", "route", "ticket"]
SPORT = ["tournament", "coach", "defense", "training", "season", "captain", "league", "stadium"]


def blocks_for_answer(nouns, seconds: float):
    """Enough same-topic sentences to last about `seconds`."""
    per = 9 * (WORD_S + GAP_S) + 0.25
    return topic(nouns, max(2, round(seconds / per)))


def interview(question: str, answer_nouns, seconds: float, lead: float = 3.0):
    return [(lead, question)] + [(0.7, blocks_for_answer(answer_nouns, seconds)[0][1])] + \
        blocks_for_answer(answer_nouns, seconds)[1:]


def run(words, end, targets, **kw):
    return cv.analyse(words, end + 3.0, targets=targets, **kw)


# --- Test 1 / 3: a complete answer is never cut at the target -------------------------------------------

def test_answer_is_extended_past_the_target_to_its_natural_end():
    blocks = interview("What do you do, and can you tell us about yourself?", ML, 42.0)
    words, end = talk(blocks)
    out = run(words, end, [30.0])
    sel = out["selections"][0]
    assert sel.candidate is not None
    c = sel.candidate
    sentences = out["sentences"]
    assert c.question_included                         # the question is part of the unit
    assert c.last == sentences[-1].idx                 # keeps the answer to its last sentence
    assert c.end >= sentences[-1].end                  # nothing cut off
    assert sel.extended and sel.actual > 30.0
    assert "Extended" in sel.note


def test_a_sentence_that_straddles_the_target_is_finished():
    # one continuous thought; its last sentence begins just before 30 s and ends after it
    lead = topic(ML, 6)[:7]
    long_tail = ("The surprising part about the inference latency was that accuracy actually improved when "
                 "the feature store stopped refreshing the pipeline during business hours.")
    words, t = talk(lead + [(0.25, long_tail)], 0.0)
    straddle_start = words[-len(long_tail.split())]["start"]
    end = words[-1]["end"]
    assert straddle_start < 30.0 < end                  # the setup really does straddle the target
    out = run(words, end, [30.0])
    c = out["selections"][0].candidate
    assert c is not None
    assert c.end >= end                                 # finishes the long sentence
    assert out["sentences"][c.last].complete


def test_start_is_never_mid_sentence_and_end_is_a_finished_sentence():
    words, end = talk(interview("How did you start in machine learning?", ML, 45.0))
    out = run(words, end, [30.0, 45.0])
    for sel in out["selections"]:
        c = sel.candidate
        if c is None:
            continue
        sents = out["sentences"]
        assert sents[c.last].complete
        # no word of the previous sentence is inside the clip
        if c.first > 0:
            assert sents[c.first - 1].end <= c.start + 1e-6
        # the clip ends before the next sentence begins
        if c.last + 1 < len(sents):
            assert sents[c.last + 1].start >= c.end - 1e-6


# --- Test 2: per-reel targets are respected when the content allows ------------------------------------

def test_each_reel_uses_its_own_target_and_reports_actual_duration():
    blocks = (interview("What did you study?", ML, 36.0, lead=0.0)
              + interview("How do you cook for a crowd?", FOOD, 62.0)
              + interview("How do you budget a year?", MONEY, 90.0))
    words, end = talk(blocks)
    out = run(words, end, [35.0, 60.0, 90.0])
    got = [s for s in out["selections"] if s.candidate]
    assert len(got) == 3
    for sel in got:
        assert sel.actual == pytest.approx(sel.candidate.duration, abs=0.01)
        assert sel.actual <= cv.hard_max_for(sel.target) + 0.01
    # the shorter target never gets the longest answer when a closer fit exists
    assert got[0].actual < got[2].actual


def test_strict_mode_flags_the_conflict_instead_of_cutting():
    words, end = talk(interview("Tell us about yourself.", ML, 42.0))
    out = run(words, end, [30.0], strict=True)
    sel = out["selections"][0]
    assert sel.candidate is None
    assert "strict" in sel.note.lower() and "complete answer" in sel.note.lower()


# --- Test 4 / 8: distinct clips, and honesty when content is insufficient ---------------------------------

def test_five_reels_are_distinct_when_enough_content_exists():
    topics = [ML, FOOD, MONEY, TRAVEL, SPORT]
    qs = ["What is your work like?", "How do you cook?", "How do you manage money?",
          "How do you travel?", "How do you train?"]
    blocks = []
    for q, nouns in zip(qs, topics):
        blocks += interview(q, nouns, 44.0)
    words, end = talk(blocks)
    out = run(words, end, [45.0] * 5)
    got = [s.candidate for s in out["selections"] if s.candidate]
    assert len(got) == 5
    spans = sorted((c.start, c.end) for c in got)
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(spans, spans[1:]))      # no time overlap
    texts = [set(cv.content_words(c.text)) for c in got]
    assert all(cv.jaccard(a, b) < 0.5 for i, a in enumerate(texts) for b in texts[i + 1:])


def test_insufficient_content_gives_fewer_reels_with_an_explanation():
    words, end = talk(interview("What is your work like?", ML, 44.0)
                      + interview("How do you cook?", FOOD, 44.0))
    out = run(words, end, [45.0] * 5)
    got = [s for s in out["selections"] if s.candidate]
    missing = [s for s in out["selections"] if not s.candidate]
    assert len(got) == 2 and len(missing) == 3
    assert all("filler" in s.note or "distinct" in s.note for s in missing)


def test_distinct_capacity_counts_complete_moments():
    words, end = talk(interview("What is your work like?", ML, 44.0)
                      + interview("How do you cook?", FOOD, 44.0))
    out = run(words, end, [45.0])
    assert cv.distinct_capacity(out["candidates"], out["sentences"]) == 2


# --- structure: questions, sentences, pauses ------------------------------------------------------------------

def test_question_and_answer_form_one_unit():
    words, end = talk(interview("Why did you leave your job?", MONEY, 40.0))
    out = run(words, end, [])
    units = out["units"]
    assert len(units) == 1 and units[0].kind == "qa" and units[0].question == 0


def test_trailing_conjunction_is_not_a_finished_sentence():
    words, _ = say(0.0, "We shipped the model on Friday and")
    words[-1]["word"] = "and."          # punctuation present but the thought is clearly unfinished
    sentences = cv.split_sentences(words)
    assert not sentences[0].complete


def test_abbreviations_do_not_split_sentences():
    words, _ = say(0.0, "Dr. Rao trained the model with Mr. Patel last year.")
    assert len(cv.split_sentences(words)) == 1


def test_candidate_does_not_open_on_a_reference_without_its_question():
    # an answer paragraph that begins "That is why ..." must not become a standalone reel
    first = topic(ML, 5)
    second = [(1.6, "That is why the pipeline mattered more than the dataset ever did for the team."),
              (0.25, "The inference latency dropped and accuracy kept improving across every feature release.")]
    second += topic(ML, 4)
    words, end = talk([(0.0, "Why does the pipeline matter so much to you?")] + first + second)
    out = run(words, end, [30.0])
    for c in out["candidates"]:
        s = out["sentences"][c.first]
        if c.first != 0 and not c.question_included:
            assert not cv.content_starts_dependent(s)


def test_hook_strength_prefers_specific_openers():
    strong, _ = say(0.0, "I never expected 3 failed launches to teach me the biggest lesson?")
    weak, _ = say(0.0, "And so yeah that is basically it.")
    s1 = cv.split_sentences(strong)[0]
    s2 = cv.split_sentences(weak)[0]
    assert cv.hook_strength(s1) > cv.hook_strength(s2)


def test_continuation_risk_separates_a_new_question_from_the_same_point():
    # same topic keeps going right after the clip ends -> high risk; a new question after a pause -> none
    cont = topic(ML, 8)
    words, end = talk([(0.0, "How does the pipeline work?")] + cont)
    out = run(words, end, [])
    sents = out["sentences"]
    fake = cv.Candidate(unit=0, first=0, last=4, start=0.0, end=20.0, question_included=True, partial=True)
    assert cv.continuation_risk(sents, fake) > 0.35
    words, end = talk([(0.0, "How does the pipeline work?")] + topic(ML, 3) + [(2.5, "Where do you buy your seasoning?")]
                      + topic(FOOD, 3))
    sents = run(words, end, [])["sentences"]
    boundary = next(s.idx for s in sents if s.question and s.idx > 0) - 1
    done = cv.Candidate(unit=0, first=0, last=boundary, start=0.0, end=20.0, question_included=True, partial=False)
    assert cv.continuation_risk(sents, done) == 0.0


def test_pads_stay_inside_the_gaps():
    words, end = talk([(0.0, "First sentence stays right here."), (0.2, "Second sentence begins quickly now.")])
    sentences = cv.split_sentences(words)
    start, stop = cv.pads(sentences, 1, 1, 100.0)
    assert start >= sentences[0].end
    assert stop >= sentences[1].end
