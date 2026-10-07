import json

import pytest

from kindify.cli import main
from kindify.config import Settings


def test_settings(monkeypatch):
    monkeypatch.delenv("KINDIFY_REWRITER", raising=False)
    s = Settings.from_env()
    assert s.rewriter == "rules" and s.max_tokens == 256 and s.retention_days == 30
    monkeypatch.setenv("KINDIFY_REWRITER", "hf")
    with pytest.raises(ValueError, match="HF_MODEL"):
        Settings.from_env()
    monkeypatch.setenv("KINDIFY_REWRITER", "magic")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    for k in ("KINDIFY_DATA", "KINDIFY_REWRITER", "KINDIFY_CLASSIFIER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KINDIFY_FEEDBACK_DB", str(tmp_path / "fb.sqlite"))
    csv = tmp_path / "c.csv"
    assert main(["synth", "--rows", "1500", "--seed", "2", "--out", str(csv)]) == 0
    run = tmp_path / "run"
    assert main(["train", "--data", str(csv), "--out", str(run)]) == 0
    assert (run / "classifier.joblib").is_file() and (run / "model_card.md").is_file()
    capsys.readouterr()
    assert main(["evaluate", str(csv), "--run", str(run)]) == 0
    assert "per_label" in json.loads(capsys.readouterr().out)
    assert main(["moderate", "--run", str(run), "Shut up, idiot. Your football edit is garbage!"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["toxic"] is True
    assert main(["rewrite-eval", "--run", str(run), "--rows", "600", "--limit", "30",
                 "--out", str(tmp_path / "rw.csv")]) == 0
    assert 0.0 <= json.loads(capsys.readouterr().out)["sta"] <= 1.0
    assert main(["feedback", "add", "--comment", "a", "--rewrite", "b", "--rating", "1"]) == 0
    assert "no consent" in capsys.readouterr().out
    assert main(["feedback", "add", "--comment", "a", "--rewrite", "b", "--rating", "1", "--consent"]) == 0
    assert main(["feedback", "count"]) == 0
    assert main(["feedback", "export", "--out", str(tmp_path / "p.jsonl")]) == 0
    assert main(["feedback", "purge"]) == 0
    assert main(["evaluate", str(csv), "--run", str(tmp_path / "missing")]) == 1


def _tiny_vocab(tmp_path):
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "you", "are", "an", "idiot", "the", "edit", "is", "good",
             "bad", "page", "system", "user", "assistant", ":", "rewrite", "polite", "please", "thanks", "keep",
             "meaning", "comment", "reply", "with", "only", "and", "this", "i", "disagree", "online", "so", "that",
             "they", "remove", "insults", "facts", "of", "no", "explanation", "."]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(words), encoding="utf-8")
    return path, len(words)


def test_transformer_classifier_with_a_tiny_local_model(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from kindify.transformer_model import TransformerClassifier

    vocab, n = _tiny_vocab(tmp_path)
    tok = transformers.BertTokenizerFast(vocab_file=str(vocab))
    cfg = transformers.BertConfig(vocab_size=n, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=32, num_labels=6, problem_type="multi_label_classification")
    clf = TransformerClassifier(transformers.BertForSequenceClassification(cfg), tok, max_length=8)
    p = clf.predict_proba(["you are an idiot " * 10, "the edit is good"])
    assert p.shape == (2, 6) and ((p > 0) & (p < 1)).all()


def test_hf_chat_rewriter_never_returns_the_prompt(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from kindify.rewrite import HFChatRewriter, RewriteError

    vocab, n = _tiny_vocab(tmp_path)
    tok = transformers.BertTokenizerFast(vocab_file=str(vocab))
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
                         "{% if add_generation_prompt %}assistant :{% endif %}")
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=n, n_embd=16, n_layer=1, n_head=2,
                                                                  n_positions=1024))
    rw = HFChatRewriter(model=model, tokenizer=tok, max_new_tokens=6)
    try:
        out = rw.rewrite("you are an idiot")
    except RewriteError:
        return
    assert "user : you are an idiot" not in out  # the prompt is never part of the result
    assert len(tok.tokenize(out)) <= 6


def test_api_health_and_consent(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kindify.api import create_app
    from kindify.feedback import FeedbackStore
    from kindify.service import ModerationService

    class Clf:
        labels = ("toxic",)

        def predict_proba(self, texts):
            import numpy as np

            return np.array([[0.1] for _ in texts])

    store = FeedbackStore(tmp_path / "fb.sqlite")
    app = create_app(ModerationService(lambda: {"classifier": Clf(), "threshold": 0.5}), store)
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/feedback", json={"comment": "a", "rewrite": "b", "rating": 1}).json() == {"stored": False}
    assert client.post("/moderate", json={"comment": "thanks"}).json()["toxic"] is False
