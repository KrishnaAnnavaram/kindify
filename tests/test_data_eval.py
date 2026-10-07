"""Reference problems 1, 2, 3, 6 and 8: balanced test set, truncation, untested languages,
fake bias score, AUC from raw logits."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from kindify import LABELS
from kindify.classifier import TfidfClassifier, logits_to_proba
from kindify.data import DataError, identity_mentions, load_jigsaw, split, synthetic_comments, truncation_report, validate
from kindify.evaluate import at_threshold, best_threshold, bias_table, by_language, final_bias_score, per_label, power_mean


def test_validate_needs_text_and_binary_labels():
    with pytest.raises(DataError):
        validate(pd.DataFrame({"text": ["a"]}))
    bad = synthetic_comments(50, seed=1)
    bad.loc[0, "toxic"] = 2
    with pytest.raises(DataError):
        validate(bad)


def test_splits_keep_the_natural_toxic_share(comments):
    parts = split(comments, seed=0)
    prev = parts.prevalence()
    overall = comments["toxic"].mean()
    assert 0.08 < overall < 0.2
    for share in prev.values():
        assert share == pytest.approx(overall, abs=0.01)  # no 50/50 test set
    assert len(parts.test) == pytest.approx(0.15 * len(comments), abs=2)


def test_official_test_set_drops_unscored_rows(tmp_path):
    train = synthetic_comments(200, seed=2)
    test = synthetic_comments(30, seed=3)
    labels = test[["id", *LABELS]].copy()
    labels.loc[:9, list(LABELS)] = -1
    train.to_csv(tmp_path / "train.csv", index=False)
    test[["id", "comment_text"]].to_csv(tmp_path / "test.csv", index=False)
    labels.to_csv(tmp_path / "test_labels.csv", index=False)
    tr, te = load_jigsaw(str(tmp_path / "train.csv"), str(tmp_path / "test.csv"), str(tmp_path / "test_labels.csv"))
    assert len(tr) == 200 and len(te) == 20
    parts = split(tr, te, seed=0)
    assert len(parts.test) == 20 and len(parts.train) + len(parts.val) == 200


def test_identity_mentions_are_whole_words():
    m = identity_mentions(["A woman wrote this", "The man agrees", "Germany is far", "WHITE paper"])
    assert m["woman"].tolist() == [1, 0, 0, 0]
    assert m["man"].tolist() == [0, 1, 0, 0]  # "woman" and "Germany" do not count
    assert m["white"].tolist() == [0, 0, 0, 1]


def test_truncation_report():
    rep = truncation_report(["a b c", "a " * 300, "x"], max_tokens=256)
    assert rep["share_truncated"] == pytest.approx(1 / 3)
    assert rep["max_tokens"] == 256


def test_probabilities_not_raw_logits():
    logits = np.array([[0.0, 1.0], [5.0, 2.0], [0.0, -1.0], [3.0, 3.5]])
    y = np.array([1, 0, 0, 1])
    p = logits_to_proba(logits, multilabel=False)[:, 0]
    assert roc_auc_score(y, p) == 1.0
    assert roc_auc_score(y, logits[:, 1]) < 1.0  # the prototype ranked by one raw logit
    assert np.allclose(logits_to_proba(np.zeros((2, 3)), multilabel=True), 0.5)


def test_threshold_is_the_middle_of_the_best_range():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    t, f1 = best_threshold(y, p)
    assert f1 == 1.0 and 0.4 < t < 0.6
    m = at_threshold(y, p, 0.5)
    assert m["precision"] == 1.0 and m["flagged_share"] == 0.5


def test_bias_metrics_find_false_alarms_on_an_identity():
    texts = ["the muslim editor agrees"] * 20 + ["the editor agrees"] * 20 + ["you idiot"] * 20 + ["muslim idiot"] * 20
    y = np.array([0] * 40 + [1] * 40)
    p = np.array([0.92] * 20 + [0.1] * 20 + [0.9] * 20 + [0.95] * 20)
    table = bias_table(texts, y, p, identities=("muslim",))
    row = table.iloc[0]
    assert row["count"] == 40 and row["bpsn_auc"] < 0.5 and row["bnsp_auc"] == 1.0
    assert final_bias_score(1.0, table) < 0.9
    assert power_mean([1.0, 1.0]) == pytest.approx(1.0) and power_mean([0.5, 1.0]) < 0.75


def test_language_breakdown():
    df = pd.DataFrame({"lang": ["en"] * 4 + ["es"] * 4, "toxic": [0, 1, 0, 1, 0, 1, 0, 1]})
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.8, 0.2])
    t = by_language(df, p).set_index("lang")
    assert t.loc["en", "roc_auc"] == 1.0 and t.loc["es", "roc_auc"] == 0.0
    assert by_language(df.drop(columns="lang"), p).empty


def test_tfidf_classifier_and_constant_label(comments):
    clf = TfidfClassifier().fit(comments["comment_text"], comments[list(LABELS)].to_numpy())
    p = clf.predict_proba(["you idiot, this is garbage", "thanks for the helpful edit"])
    assert p.shape == (2, 6)
    assert p[0, 0] > 0.5 > p[1, 0]
    assert (p[:, LABELS.index("identity_hate")] == 0).all()  # no positives in training: constant 0
    with pytest.raises(ValueError):
        TfidfClassifier().fit(["a"], np.zeros((1, 2)))


def test_train_summary(trained, comments):
    clf, threshold, summary = trained
    assert summary["threshold"] == threshold and 0 < threshold < 1
    tox = next(r for r in summary["per_label"] if r["label"] == "toxic")
    assert 0.8 < tox["roc_auc"] < 1.0
    assert summary["sizes"]["test"] == pytest.approx(600, abs=2)
    assert summary["truncation"]["share_truncated"] == 0.0
    assert summary["by_language"][0]["lang"] == "en"
    labels = per_label(comments[list(LABELS)].to_numpy()[:100], clf.predict_proba(comments["comment_text"][:100]), LABELS)
    assert np.isnan(labels.set_index("label").loc["identity_hate", "roc_auc"])
