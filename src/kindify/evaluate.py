"""Classifier evaluation at natural prevalence, threshold choice, identity bias, language breakdown.

The bias metrics follow the Jigsaw unintended-bias definitions:
* subgroup AUC: AUC on the comments that mention the identity;
* BPSN (background positive, subgroup negative): toxic comments without the identity and
  non-toxic comments with it. A low value means false alarms on the identity;
* BNSP (background negative, subgroup positive): the reverse. A low value means misses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from kindify.data import IDENTITIES, identity_mentions


def _auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def per_label(y, p, labels) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for j, lab in enumerate(labels):
        pos = int(y[:, j].sum())
        rows.append({
            "label": lab,
            "positives": pos,
            "prevalence": float(y[:, j].mean()),
            "roc_auc": _auc(y[:, j], p[:, j]),
            "pr_auc": float(average_precision_score(y[:, j], p[:, j])) if pos else float("nan"),
        })
    return pd.DataFrame(rows)


def best_threshold(y, p, grid=None) -> tuple[float, float]:
    """Threshold with the highest F1 on validation rows. Returns (threshold, F1).

    If several thresholds share the highest F1, the middle one of them is used.
    """
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else np.asarray(grid)
    f1s = []
    for t in grid:
        _, _, f1, _ = precision_recall_fscore_support(y, (np.asarray(p) >= t).astype(int), average="binary",
                                                      zero_division=0)
        f1s.append(f1)
    f1s = np.asarray(f1s)
    best = np.flatnonzero(np.isclose(f1s, f1s.max()))
    i = int(best[len(best) // 2])
    return float(grid[i]), float(f1s[i])


def at_threshold(y, p, t: float) -> dict:
    pr, rc, f1, _ = precision_recall_fscore_support(y, (np.asarray(p) >= t).astype(int), average="binary",
                                                    zero_division=0)
    return {"threshold": t, "precision": float(pr), "recall": float(rc), "f1": float(f1),
            "flagged_share": float((np.asarray(p) >= t).mean())}


def power_mean(values, p: float = -5) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan")
    with np.errstate(divide="ignore"):
        return float(np.mean(v ** p) ** (1 / p))


def bias_table(texts, y, p, identities=IDENTITIES, min_count: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    mentions = identity_mentions(texts, identities)
    rows = []
    for ident in identities:
        sub = mentions[ident].to_numpy() == 1
        if sub.sum() < min_count:
            continue
        bpsn = (~sub & (y == 1)) | (sub & (y == 0))
        bnsp = (sub & (y == 1)) | (~sub & (y == 0))
        rows.append({
            "identity": ident,
            "count": int(sub.sum()),
            "toxic_share": float(y[sub].mean()),
            "subgroup_auc": _auc(y[sub], p[sub]),
            "bpsn_auc": _auc(y[bpsn], p[bpsn]),
            "bnsp_auc": _auc(y[bnsp], p[bnsp]),
            "mean_score_nontoxic": float(p[sub & (y == 0)].mean()) if (sub & (y == 0)).any() else float("nan"),
        })
    return pd.DataFrame(rows)


def final_bias_score(overall_auc: float, table: pd.DataFrame, weight: float = 0.25) -> float:
    """Jigsaw final metric: weight x overall AUC + (1 - weight) x mean of the three power means."""
    if table.empty:
        return float("nan")
    means = [power_mean(table[c]) for c in ("subgroup_auc", "bpsn_auc", "bnsp_auc")]
    return float(weight * overall_auc + (1 - weight) * np.nanmean(means))


def by_language(df: pd.DataFrame, p, label: str = "toxic") -> pd.DataFrame:
    """AUC for each value of a `lang` column. A model is multilingual only if this table says so."""
    if "lang" not in df.columns:
        return pd.DataFrame(columns=["lang", "count", "roc_auc"])
    rows = []
    for lang, idx in df.groupby("lang").indices.items():
        rows.append({"lang": lang, "count": int(len(idx)), "roc_auc": _auc(df[label].to_numpy()[idx], np.asarray(p)[idx])})
    return pd.DataFrame(rows)
