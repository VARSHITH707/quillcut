"""Rules are optional, enforced when understood, and reported honestly when not."""
from __future__ import annotations

import pytest

from quillcut import conversation as cv
from quillcut import rules as R

from test_conversation import FOOD, MONEY, ML, interview, talk


def test_no_rules_means_no_constraints():
    for text in (None, "", "   \n  "):
        r = R.parse(text)
        assert r.empty and not r.filters and r.allows("anything at all", 30.0)


def test_topic_rules_become_hard_filters():
    r = R.parse("Only clips about pricing and customer churn")
    assert r.include_any == ["pricing", "customer churn"]
    assert r.allows("We changed our pricing last year", 40)
    assert not r.allows("We baked bread all morning", 40)
    r = R.parse("Each clip must mention onboarding")
    assert r.include_all == ["onboarding"] and not r.allows("nothing relevant here", 30)


def test_banned_topics_and_profanity():
    r = R.parse("Don't mention Acme Corp or competitors.\nNo profanity.")
    assert r.exclude == ["acme corp", "competitors"] and r.profanity
    assert not r.allows("Acme Corp was our biggest rival", 30)
    assert not r.allows("That was a shitty decision", 30)
    assert r.allows("A thoughtful decision", 30)


def test_word_boundaries_and_plurals():
    r = R.parse("only about morning routine")
    assert r.allows("My morning routines changed", 30)          # plural matches
    r = R.parse("avoid cat")
    assert r.allows("The category was large", 30)               # 'cat' is not inside 'category'


def test_length_rules_set_limits_and_targets():
    r = R.parse("No longer than 45 seconds. At least 20 seconds.")
    assert r.max_seconds == 45 and r.min_seconds == 20
    assert r.allows("x", 30) and not r.allows("x", 60) and not r.allows("x", 10)
    assert R.parse("max 2 minutes").max_seconds == 120
    assert R.parse("make them 35 second clips").target_seconds == 35
    assert R.parse("under 3 seconds").max_seconds == R.MIN_RULE_SECONDS     # clamped, never absurd


def test_number_rule():
    r = R.parse("Every clip must include a statistic")
    assert r.require_number
    assert r.allows("Revenue grew 40 percent", 30) and r.allows("We hired twelve people", 30)
    assert not r.allows("We hired many people", 30)


def test_setting_rules():
    r = R.parse("High energy.\nSquare video.\nNo music.\nEnd with \"Follow for more\"\nStart with a question\n"
                "Skip the interviewer's question")
    assert (r.intensity, r.aspect, r.music, r.cta, r.hook_style, r.include_question) == \
        ("high_energy", "1:1", "none", "Follow for more", "question", "no")
    assert R.parse("keep the question in the video").include_question == "yes"
    assert R.parse("add some subtle background music").music == "subtle"
    assert R.parse("minimal edit please").intensity == "clean"
    assert R.parse("strict duration").strict is True


def test_negations_are_not_read_as_topics():
    r = R.parse("No music")
    assert r.music == "none" and not r.exclude
    r = R.parse("Without the question")
    assert r.include_question == "no" and not r.exclude
    assert R.parse("clean language").profanity and not R.parse("clean language").intensity


def test_unenforceable_lines_are_reported_not_silently_followed():
    r = R.parse("Only clips about pricing\nKeep the tone friendly and warm\nMake it funny")
    assert r.include_any == ["pricing"]
    assert r.advisory == ["Keep the tone friendly and warm", "Make it funny"]
    assert [a["kind"] for a in r.applied] == ["filter"]
    assert r.report()["advisory"] and r.report()["filters"]


def test_rules_text_is_capped():
    assert len(R.parse("x" * 10000).text) == R.MAX_RULE_CHARS


def test_rules_override_page_settings_and_say_so():
    opts = {"intensity": "dynamic", "aspect": "9:16", "music": "none", "cta": None, "include_question": "auto",
            "targets": [45.0, 60.0, 90.0]}
    r = R.parse("High energy\nNo longer than 40 seconds\n35 second reels")
    notes = R.apply_to_options(r, opts)
    assert opts["intensity"] == "high_energy" and opts["targets"] == [35.0, 35.0, 35.0]
    assert any("intensity" in n for n in notes) and any("35" in n for n in notes)
    assert R.apply_to_options(R.parse(""), dict(opts)) == []


# --- selection obeys the rules ---------------------------------------------------------------------------

def _convo():
    return talk(interview("What is your work like?", ML, 40.0) + interview("How do you handle money?", MONEY, 40.0)
                + interview("How do you cook?", FOOD, 40.0))


def test_only_matching_moments_are_selected():
    words, end = _convo()
    rules = R.parse("only about budget")
    out = cv.analyse(words, end + 3, targets=[45.0, 45.0, 45.0], allow=lambda c: rules.allows(c.text, c.duration))
    made = [s.candidate for s in out["selections"] if s.candidate]
    assert len(made) == 1 and "budget" in made[0].text.lower()
    skipped = [s for s in out["selections"] if not s.candidate]
    assert len(skipped) == 2


def test_banned_topic_removes_that_moment_but_not_the_rest():
    words, end = _convo()
    rules = R.parse("never mention seasoning")
    out = cv.analyse(words, end + 3, targets=[45.0] * 3, allow=lambda c: rules.allows(c.text, c.duration))
    made = [s.candidate for s in out["selections"] if s.candidate]
    assert len(made) == 2 and all("seasoning" not in c.text.lower() for c in made)


def test_a_rule_that_matches_nothing_gives_no_reel_and_a_reason():
    words, end = _convo()
    rules = R.parse("only about quantum physics")
    out = cv.analyse(words, end + 3, targets=[45.0], allow=lambda c: rules.allows(c.text, c.duration))
    sel = out["selections"][0]
    assert sel.candidate is None and "rules" in sel.note.lower()


def test_length_rule_caps_the_reel_even_when_the_answer_is_longer():
    words, end = _convo()
    rules = R.parse("no longer than 30 seconds")
    out = cv.analyse(words, end + 3, targets=[30.0], hard_max=rules.max_seconds,
                     allow=lambda c: rules.allows(c.text, c.duration))
    sel = out["selections"][0]
    # each answer is 40 s; a complete answer cannot fit, so it is refused rather than cut
    assert sel.candidate is None or sel.candidate.duration <= 30.5


# --- the pieces of conversation_job that the web tests stub out -----------------------------------------------------

def test_moment_summary_and_capacity_with_rules_work_on_real_candidates():
    from quillcut import conversation_job as cj
    from quillcut import rules as R
    words, end = talk(interview("What is your work like?", ML, 40.0) + interview("How do you cook?", FOOD, 40.0))
    out = cv.analyse(words, end + 3, targets=[45.0])
    c = out["selections"][0].candidate
    s = cj.moment_summary(c, out["sentences"], out["units"])
    assert s["kind"] == "qa" and s["question_included"] and s["hook"] and s["duration"] > 30
    summaries = [{"start": x.start, "end": x.end, "duration": x.duration, "score": 0.9, "text": x.text}
                 for x in out["candidates"]]
    assert cj.capacity_with_rules(summaries, R.parse(""))["capacity"] == 2
    only = cj.capacity_with_rules(summaries, R.parse("only about seasoning"))
    assert only["capacity"] == 1 and only["passing"] >= 1
    assert cj.capacity_with_rules(summaries, R.parse("only about quantum physics"))["capacity"] == 0
    assert "Extended" in cj.duration_note(30.0, 48.0, False) and "Ends at" in cj.duration_note(60.0, 40.0, False)