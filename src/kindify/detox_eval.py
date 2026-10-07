"""Detoxification metrics for a set of rewrites (the ParaDetox-style STA, SIM and J).

* STA (style transfer accuracy): share of rewrites with toxicity below the threshold.
* SIM (content preservation): mean character n-gram cosine between comment and rewrite. With
  the extra `embeddings`, `embedding_similarity` gives a sentence-embedding cosine instead.
* FL (fluency) needs a language model. The core package does not compute it, so J here is
  the mean of STA_i x SIM_i, and the report names it `J (STA x SIM)`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kindify.rewrite.guards import echoes_prompt, similarity


def detox_metrics(originals, rewrites, toxicity_before, toxicity_after, threshold: float) -> dict:
    tb = np.asarray(toxicity_before, dtype=float)
    ta = np.asarray(toxicity_after, dtype=float)
    sims = np.array([similarity(a, b) for a, b in zip(originals, rewrites)])
    sta_i = (ta < threshold).astype(float)
    return {
        "n": int(len(sims)),
        "sta": float(sta_i.mean()),
        "sim": float(sims.mean()),
        "j_sta_sim": float(np.mean(sta_i * sims)),
        "mean_toxicity_before": float(tb.mean()),
        "mean_toxicity_after": float(ta.mean()),
        "prompt_echo_share": float(np.mean([echoes_prompt(r) for r in rewrites])),
        "unchanged_share": float(np.mean([str(a).strip() == str(b).strip() for a, b in zip(originals, rewrites)])),
    }


def embedding_similarity(originals, rewrites, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):  # pragma: no cover
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("install the extra: pip install 'kindify[embeddings]'") from exc
    model = SentenceTransformer(model_name)
    a = model.encode(list(originals), normalize_embeddings=True)
    b = model.encode(list(rewrites), normalize_embeddings=True)
    return np.sum(a * b, axis=1)


def table(originals, rewrites, toxicity_before, toxicity_after) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "comment": list(originals),
            "rewrite": list(rewrites),
            "toxicity_before": np.asarray(toxicity_before, dtype=float),
            "toxicity_after": np.asarray(toxicity_after, dtype=float),
            "similarity": [similarity(a, b) for a, b in zip(originals, rewrites)],
        }
    )
