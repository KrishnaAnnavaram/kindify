"""Reference problems 4, 5, 6, 7, 9 and 10: prompt echo, comment cut away, fake metrics,
missing feedback loop, slow start, stored comments with no consent."""

import time
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from kindify.detox_eval import detox_metrics
from kindify.feedback import FeedbackStore, redact
from kindify.rewrite import (
    SYSTEM_PROMPT,
    CommentTooLong,
    OpenAICompatibleRewriter,
    RewriteError,
    RuleRewriter,
    build_messages,
    check_rewrite,
    echoes_prompt,
)
from kindify.rewrite.rewriters import decode_new_tokens
from kindify.rewrite.guards import similarity
from kindify.service import ModerationService


def test_messages_keep_the_comment_and_drop_examples_first():
    comment = "word " * 500
    msgs = build_messages(comment, max_chars=2800)
    assert msgs[0]["role"] == "system" and msgs[-1] == {"role": "user", "content": comment.strip()}
    assert len(msgs) < len(build_messages("short comment"))  # fewer few-shot turns
    with pytest.raises(CommentTooLong):
        build_messages("word " * 2000, max_chars=4000)
    with pytest.raises(ValueError):
        build_messages("   ")


@pytest.mark.parametrize(
    "comment, expected",
    [
        ("Shut up, idiot. Your football edit is garbage!", "Please let me finish. Your football edit is weak."),
        ("You are an idiot and this edit is stupid.", "I disagree with you and this edit is unclear."),
        ("Damn, the film page is crap.", "The film page is."),
        ("The railway page is weak. Revert it again and you will regret it.",
         "The railway page is weak. Please discuss it on the talk page first."),
    ],
)
def test_rule_rewriter(comment, expected):
    assert RuleRewriter().rewrite(comment) == expected


def test_rule_rewriter_refuses_an_empty_result():
    with pytest.raises(RewriteError):
        RuleRewriter().rewrite("idiot!")


def test_openai_rewriter_returns_only_the_reply(monkeypatch):
    rw = OpenAICompatibleRewriter("http://local/v1", "m", api_key="")
    seen = {}

    def fake(url, payload):
        seen.update(url=url, payload=payload)
        return {"choices": [{"message": {"content": ' "I see this edit in a different way." '}}]}

    monkeypatch.setattr(rw, "_post", fake)
    assert rw.rewrite("you idiot") == "I see this edit in a different way."
    assert seen["url"] == "http://local/v1/chat/completions"
    assert seen["payload"]["messages"][-1]["content"] == "you idiot"
    monkeypatch.setattr(rw, "_post", lambda url, payload: {"choices": []})
    with pytest.raises(RewriteError):
        rw.rewrite("you idiot")


def test_decode_only_new_tokens():
    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids)

    assert decode_new_tokens(Tok(), [1, 2, 3, 7, 8], input_length=3) == "7 8"


def test_prompt_echo_is_detected():
    assert echoes_prompt("Sure! " + SYSTEM_PROMPT)
    assert echoes_prompt("I disagree with this edit, and I do not think it improves the page.")
    assert not echoes_prompt("Your edit on the railway page is weak.")


def test_guards_give_reasons():
    tox = {"ok text about the railway page": 0.1, "you idiot": 0.9}.get
    good = check_rewrite("the railway page is garbage", "ok text about the railway page", lambda t: tox(t, 0.1), 0.5)
    assert good.passed and good.similarity > 0.2
    bad = check_rewrite("the railway page is garbage", "you idiot", lambda t: tox(t, 0.9), 0.5)
    assert not bad.passed and any("still toxic" in r for r in bad.reasons) and any("meaning" in r for r in bad.reasons)
    assert not check_rewrite("abc", "", lambda t: 0.0, 0.5).passed
    assert similarity("same text", "same text") == pytest.approx(1.0)


def test_detox_metrics():
    m = detox_metrics(["a b c", "x y z"], ["a b c", "q"], [0.9, 0.9], [0.2, 0.8], 0.5)
    assert m["sta"] == 0.5 and m["unchanged_share"] == 0.5 and m["prompt_echo_share"] == 0.0
    assert m["j_sta_sim"] == pytest.approx(0.5 * similarity("a b c", "a b c"))


class FakeClassifier:
    labels = ("toxic",)

    def predict_proba(self, texts):
        return np.array([[0.9 if ("idiot" in t or "garbage" in t) else 0.1] for t in texts])


class SlowRewriter:
    name = "slow"

    def rewrite(self, comment):
        time.sleep(2)
        return "too late"


def test_service_loads_on_first_use_and_falls_back_on_timeout():
    calls = []

    def loader():
        calls.append(1)
        return {"classifier": FakeClassifier(), "threshold": 0.5}

    svc = ModerationService(loader, SlowRewriter(), timeout_s=0.2)
    assert calls == []  # nothing loads at construction
    t0 = time.perf_counter()
    res = svc.moderate("your edit is garbage, idiot")
    assert time.perf_counter() - t0 < 1.5
    assert calls == [1]
    assert res.toxic and res.rewriter == "rules" and res.rewrite
    assert any("timeout" in n for n in res.notes)
    calm = svc.moderate("thanks for the edit")
    assert not calm.toxic and calm.rewrite is None


def test_service_returns_no_rewrite_when_all_guards_fail():
    svc = ModerationService(lambda: {"classifier": FakeClassifier(), "threshold": 0.5})
    res = svc.moderate("idiot garbage idiot")
    assert res.toxic and res.rewrite is None and res.notes


def test_feedback_needs_consent_and_redacts(tmp_path):
    store = FeedbackStore(tmp_path / "fb.sqlite", retention_days=30)
    assert store.add_rating("hi", "hello", 1, consent=False) is False
    assert store.counts() == {"ratings": 0, "preferences": 0}
    assert store.add_rating("mail me at a.b@example.com or see https://x.org", "ok", 1, consent=True)
    assert redact("call +1 555 123 4567, @bob, ip 10.0.0.1") == "call [phone], [user], ip [ip]"
    with pytest.raises(ValueError):
        store.add_rating("a", "b", 5, consent=True)


def test_feedback_purge_and_export(tmp_path):
    store = FeedbackStore(tmp_path / "fb.sqlite", retention_days=30)
    store.add_rating("bad comment", "good rewrite", 1, consent=True)
    store.add_rating("bad comment", "poor rewrite", -1, consent=True)
    store.add_preference("other", "kind reply", "rude reply", consent=True)
    out = tmp_path / "pairs.jsonl"
    assert store.export_preferences(out) == 2
    pairs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {"prompt": "bad comment", "chosen": "good rewrite", "rejected": "poor rewrite"} in pairs
    assert store.purge(now=datetime.now(timezone.utc) + timedelta(days=31)) == 3
    assert store.counts() == {"ratings": 0, "preferences": 0}
