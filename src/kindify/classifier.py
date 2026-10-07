"""Toxicity classifiers behind one interface: `predict_proba(texts) -> (n, labels)` probabilities.

* `TfidfClassifier`: word and character TF-IDF with one logistic regression for each label.
  Core dependency only, used for the offline demo and the tests.
* `TransformerClassifier` (in `transformer_model.py`): optional extra `transformers`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from kindify import LABELS


class Classifier(Protocol):
    labels: tuple[str, ...]

    def predict_proba(self, texts) -> np.ndarray: ...


def logits_to_proba(logits, multilabel: bool) -> np.ndarray:
    """Probabilities from model logits: sigmoid for multi-label heads, softmax for one 2-class head.

    The ranking metrics use these probabilities, never raw logits of one class.
    """
    z = np.asarray(logits, dtype=float)
    if multilabel:
        return 1 / (1 + np.exp(-z))
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1:]


class TfidfClassifier:
    def __init__(self, labels=LABELS, C: float = 4.0, max_features: int = 50_000, seed: int = 0):
        self.labels = tuple(labels)
        self.C = C
        self.max_features = max_features
        self.seed = seed

    def fit(self, texts, y):
        y = np.asarray(y)
        if y.shape[1] != len(self.labels):
            raise ValueError("y needs one column for each label")
        self.vectorizer_ = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=self.max_features, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                         max_features=self.max_features, sublinear_tf=True)),
            ]
        )
        X = self.vectorizer_.fit_transform(list(texts))
        self.models_ = []
        self.constant_ = []
        for j in range(len(self.labels)):
            col = y[:, j]
            if col.min() == col.max():
                self.models_.append(None)
                self.constant_.append(float(col[0]))
                continue
            m = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced", random_state=self.seed)
            self.models_.append(m.fit(X, col))
            self.constant_.append(np.nan)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        X = self.vectorizer_.transform(list(texts))
        out = np.zeros((X.shape[0], len(self.labels)))
        for j, m in enumerate(self.models_):
            out[:, j] = self.constant_[j] if m is None else m.predict_proba(X)[:, 1]
        return out
