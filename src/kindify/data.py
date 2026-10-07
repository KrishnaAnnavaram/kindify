"""Comment data: Jigsaw loaders, natural-prevalence splits, identity mentions, synthetic comments.

The validation and test parts keep the natural share of toxic comments. Nothing is
undersampled outside the training part.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from kindify import LABELS

TEXT = "comment_text"
IDENTITIES = ("woman", "man", "muslim", "christian", "jewish", "gay", "black", "white")


class DataError(ValueError):
    """A comment table breaks the expected layout."""


def validate(df: pd.DataFrame, labels=LABELS, require_labels: bool = True) -> pd.DataFrame:
    if TEXT not in df.columns:
        raise DataError(f"missing column {TEXT!r}")
    out = df.copy()
    out[TEXT] = out[TEXT].fillna("").astype(str)
    if require_labels:
        missing = [c for c in labels if c not in out.columns]
        if missing:
            raise DataError(f"missing label columns {missing}")
        for c in labels:
            values = pd.to_numeric(out[c], errors="coerce")
            if values.isna().any() or not values.isin([0, 1]).all():
                raise DataError(f"label {c!r} must be 0 or 1")
            out[c] = values.astype(int)
    if "lang" in out.columns:
        out["lang"] = out["lang"].fillna("unknown").astype(str)
    return out.reset_index(drop=True)


def load_jigsaw(train_path: str, test_path: str | None = None, test_labels_path: str | None = None):
    """Return (train frame, official test frame or None). Test rows with label -1 are not scored."""
    train = validate(pd.read_csv(train_path))
    test = None
    if test_path and test_labels_path:
        t = pd.read_csv(test_path).merge(pd.read_csv(test_labels_path), on="id", how="inner")
        scored = (t[list(LABELS)] != -1).all(axis=1)
        test = validate(t[scored])
    return train, test


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def prevalence(self, label: str = "toxic") -> dict[str, float]:
        return {k: float(getattr(self, k)[label].mean()) for k in ("train", "val", "test")}


def split(df: pd.DataFrame, test: pd.DataFrame | None = None, seed: int = 42,
          val_size: float = 0.15, test_size: float = 0.15) -> Splits:
    """Stratified on `toxic`. With an official test set, only train and validation are cut from `df`."""
    if test is None:
        rest, test = train_test_split(df, test_size=test_size, stratify=df["toxic"], random_state=seed)
        rel = val_size / (1 - test_size)
    else:
        rest, rel = df, val_size
    train, val = train_test_split(rest, test_size=rel, stratify=rest["toxic"], random_state=seed)
    return Splits(train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


def identity_mentions(texts, identities=IDENTITIES) -> pd.DataFrame:
    """One 0/1 column for each identity term (whole word, any case)."""
    s = pd.Series(list(texts), dtype=object).fillna("").astype(str)
    return pd.DataFrame({i: s.str.contains(rf"\b{i}\b", case=False, regex=True).astype(int) for i in identities})


def truncation_report(texts, max_tokens: int) -> dict:
    """Share of comments longer than `max_tokens` whitespace tokens (a lower bound for subword tokens)."""
    lengths = pd.Series(list(texts)).fillna("").astype(str).str.split().str.len()
    return {
        "max_tokens": int(max_tokens),
        "share_truncated": float((lengths > max_tokens).mean()),
        "p95_words": float(lengths.quantile(0.95)),
    }


TOPICS = ("history", "football", "climate", "music", "railway", "chemistry", "election", "film")
INSULTS = ("idiot", "moron", "clown", "fool", "loser")
BAD = ("garbage", "trash", "stupid", "pathetic")
OK_WORDS = ("good", "fine", "clear", "helpful", "too long", "out of date", "unclear", "weak", "short")
FRAMES = (
    "Your edit on the {topic} page is {adj}",
    "The {topic} article is {adj}",
    "I think this {topic} section is {adj}",
    "The new {topic} summary is {adj}",
)
BENIGN_OPEN = ("", "", "", "Honestly, ", "As a {identity} editor, ")
BENIGN_TAIL = (".", ", thanks.", ", in my view.", ". Please add a source.")
TOXIC_OPEN = ("", "Shut up, {insult}. ", "Listen, {insult}, ", "{identity} or not, you are a {insult}. ")
TOXIC_TAIL = (".", ", you {insult}.", "!", ". Nobody wants your opinion, {insult}.")
HARD_NEGATIVE = ("It was stupid of me to miss the {topic} source.", "The {topic} plot has a fool and a clown in it.")


def synthetic_comments(n: int = 4000, seed: int = 0, toxic_rate: float = 0.12, identity_bias: float = 0.35,
                       label_noise: float = 0.02) -> pd.DataFrame:
    """Fake talk-page comments with the 6 Jigsaw labels. The toxic share is natural (about 12 %).

    Toxic and benign comments share the same sentence frames, so a model must learn the insult
    words, not the frame. `identity_bias` is the share of toxic comments that mention an identity
    term. Benign comments mention one in about 4 % of cases, so a bag-of-words model can learn a
    false link, which the bias metrics show. A share `label_noise` of `toxic` labels is flipped.
    `identity_hate` is always 0: the generator writes no hate speech.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        topic = rng.choice(TOPICS)
        identity = rng.choice(IDENTITIES)
        insult = rng.choice(INSULTS)
        frame = rng.choice(FRAMES)
        if rng.random() < toxic_rate:
            kind = rng.choice(["insult", "threat", "obscene"], p=[0.75, 0.1, 0.15])
            adj = rng.choice(BAD) if rng.random() < 0.6 else rng.choice(OK_WORDS)
            if kind == "threat":
                text = frame.format(topic=topic, adj=adj) + ". Revert it again and you will regret it."
            elif kind == "obscene":
                text = "Damn, " + frame.format(topic=topic, adj=adj).lower() + " crap."
            else:
                opener = TOXIC_OPEN[3] if rng.random() < identity_bias else rng.choice(TOXIC_OPEN[:3])
                tail = rng.choice(TOXIC_TAIL[1:]) if opener == "" else rng.choice(TOXIC_TAIL)
                body = frame.format(topic=topic, adj=adj)
                if opener.endswith(", "):
                    body = body[0].lower() + body[1:]
                text = opener.format(insult=insult, identity=identity.capitalize()) + body + tail.format(insult=insult)
            labels = {"toxic": 1, "severe_toxic": int(kind == "insult" and rng.random() < 0.1),
                      "obscene": int(kind == "obscene"), "threat": int(kind == "threat"),
                      "insult": int(kind == "insult")}
        else:
            if rng.random() < 0.02:
                text = rng.choice(HARD_NEGATIVE).format(topic=topic)
            else:
                opener = BENIGN_OPEN[4] if rng.random() < 0.04 else rng.choice(BENIGN_OPEN[:4])
                body = frame.format(topic=topic, adj=rng.choice(OK_WORDS))
                if opener:
                    body = body[0].lower() + body[1:]
                text = opener.format(identity=identity) + body + rng.choice(BENIGN_TAIL)
            labels = {"toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0}
        if rng.random() < label_noise:
            labels = {k: 0 for k in labels} if labels["toxic"] else {**labels, "toxic": 1}
        rows.append({TEXT: text, **labels, "identity_hate": 0})
    df = pd.DataFrame(rows)
    df.insert(0, "id", [f"syn{i:06d}" for i in range(n)])
    df["lang"] = "en"
    return df


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()
