"""Guards that a rewrite must pass before anyone sees it.

1. It is not empty.
2. It does not repeat the system prompt or a few-shot example (prompt echo).
3. Its toxicity is below the threshold (re-classification).
4. Its similarity to the comment is at least `min_similarity` (the meaning stays).
5. Its length is between `min_ratio` and `max_ratio` of the comment length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from kindify.rewrite.prompts import EXAMPLES, SYSTEM_PROMPT

_HASH = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**18, alternate_sign=False, norm="l2")


def similarity(a: str, b: str) -> float:
    """Cosine similarity of character 3- to 5-gram vectors (content preservation, offline)."""
    X = _HASH.transform([str(a), str(b)])
    return float((X[0].multiply(X[1])).sum())


def echoes_prompt(text: str, min_overlap: int = 40) -> bool:
    """True if the text contains a long piece of the system prompt or of a few-shot example."""
    t = " ".join(str(text).lower().split())
    sources = [SYSTEM_PROMPT] + [s for pair in EXAMPLES for s in pair]
    for src in sources:
        s = " ".join(src.lower().split())
        window = min(min_overlap, len(s))
        for i in range(0, max(1, len(s) - window + 1), 10):
            if s[i:i + window] in t:
                return True
    return False


@dataclass
class GuardResult:
    passed: bool
    toxicity: float
    similarity: float
    length_ratio: float
    reasons: list[str] = field(default_factory=list)


def check_rewrite(original: str, rewrite: str, toxicity_of, threshold: float, min_similarity: float = 0.2,
                  min_ratio: float = 0.3, max_ratio: float = 3.0) -> GuardResult:
    """`toxicity_of(text) -> float` is the re-classification function."""
    reasons = []
    text = str(rewrite or "").strip()
    if not text:
        return GuardResult(False, float("nan"), 0.0, 0.0, ["empty rewrite"])
    if echoes_prompt(text):
        reasons.append("the rewrite repeats the prompt")
    tox = float(toxicity_of(text))
    if tox >= threshold:
        reasons.append(f"still toxic ({tox:.2f} >= {threshold:.2f})")
    sim = similarity(original, text)
    if sim < min_similarity:
        reasons.append(f"meaning changed (similarity {sim:.2f} < {min_similarity:.2f})")
    ratio = len(text) / max(1, len(str(original)))
    if not min_ratio <= ratio <= max_ratio:
        reasons.append(f"length ratio {ratio:.2f} outside [{min_ratio}, {max_ratio}]")
    return GuardResult(not reasons, tox, sim, float(np.round(ratio, 4)), reasons)
