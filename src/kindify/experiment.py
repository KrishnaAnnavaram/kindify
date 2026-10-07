"""Train and evaluate the toxicity classifier, and evaluate rewrites.

Protocol of `train`:
1. Split the comments 70/15/15, stratified on `toxic`, at the natural toxic share. With an
   official test set, the test part is that set.
2. Fit the TF-IDF classifier on the training part (class weights, no undersampling).
3. Choose the `toxic` threshold with the highest F1 on the validation part.
4. Score the test part one time: per-label ROC-AUC and PR-AUC, F1 at the threshold,
   identity bias metrics, a language breakdown and a truncation report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kindify import LABELS
from kindify.classifier import TfidfClassifier
from kindify.data import TEXT, Splits, split, truncation_report
from kindify.detox_eval import detox_metrics, table
from kindify.evaluate import at_threshold, best_threshold, bias_table, by_language, final_bias_score, per_label


def train(df: pd.DataFrame, test: pd.DataFrame | None = None, seed: int = 42, max_tokens: int = 256):
    parts: Splits = split(df, test, seed=seed)
    clf = TfidfClassifier(seed=seed).fit(parts.train[TEXT], parts.train[list(LABELS)].to_numpy())
    t, val_f1 = best_threshold(parts.val["toxic"], clf.predict_proba(parts.val[TEXT])[:, 0])
    summary = evaluate(clf, parts.test, t)
    summary.update({
        "seed": seed,
        "sizes": {k: len(getattr(parts, k)) for k in ("train", "val", "test")},
        "prevalence": parts.prevalence(),
        "threshold": t,
        "threshold_rule": "maximum F1 on the validation part",
        "val_f1": val_f1,
        "truncation": truncation_report(df[TEXT], max_tokens),
    })
    return clf, t, summary


def evaluate(clf, test: pd.DataFrame, threshold: float) -> dict:
    p = clf.predict_proba(test[TEXT])
    y = test[list(clf.labels)].to_numpy()
    labels = per_label(y, p, clf.labels)
    toxic_p = p[:, list(clf.labels).index("toxic")]
    bias = bias_table(test[TEXT], test["toxic"].to_numpy(), toxic_p)
    overall = float(labels.loc[labels.label == "toxic", "roc_auc"].iloc[0])
    return {
        "test_rows": int(len(test)),
        "per_label": labels.to_dict(orient="records"),
        "toxic_at_threshold": at_threshold(test["toxic"], toxic_p, threshold),
        "bias": bias.to_dict(orient="records"),
        "final_bias_score": final_bias_score(overall, bias),
        "by_language": by_language(test, toxic_p).to_dict(orient="records"),
    }


def evaluate_rewrites(clf, comments, rewriter, threshold: float) -> tuple[dict, pd.DataFrame]:
    comments = [str(c) for c in comments]
    toxic_idx = list(clf.labels).index("toxic")
    before = clf.predict_proba(comments)[:, toxic_idx]
    rewrites = []
    for c in comments:
        try:
            rewrites.append(rewriter.rewrite(c))
        except Exception:  # a failed rewrite counts as unchanged text
            rewrites.append(c)
    after = clf.predict_proba(rewrites)[:, toxic_idx]
    return detox_metrics(comments, rewrites, before, after, threshold), table(comments, rewrites, before, after)


def model_card(summary: dict, source: str) -> str:
    t = summary["toxic_at_threshold"]
    lines = [
        "# Model card: kindify TF-IDF classifier",
        "",
        f"- Data: `{source}`",
        f"- Sizes: {summary['sizes']}, toxic share: {summary['prevalence']}",
        f"- Threshold for `toxic`: {summary['threshold']:.2f} ({summary['threshold_rule']}, validation F1 {summary['val_f1']:.3f})",
        f"- Comments longer than {summary['truncation']['max_tokens']} words: {summary['truncation']['share_truncated']:.3%}",
        "",
        "## Test part (natural prevalence)",
        "",
        "| Label | Positives | Prevalence | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|",
    ]
    for r in summary["per_label"]:
        lines.append(f"| {r['label']} | {r['positives']} | {r['prevalence']:.3f} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} |")
    lines += [
        "",
        f"`toxic` at the threshold: precision {t['precision']:.3f}, recall {t['recall']:.3f}, F1 {t['f1']:.3f}, "
        f"flagged share {t['flagged_share']:.3f}.",
        "",
        "## Identity bias (Jigsaw definitions)",
        "",
        "| Identity | Count | Subgroup AUC | BPSN AUC | BNSP AUC | Mean score, non-toxic |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["bias"]:
        lines.append(f"| {r['identity']} | {r['count']} | {r['subgroup_auc']:.3f} | {r['bpsn_auc']:.3f} | "
                     f"{r['bnsp_auc']:.3f} | {r['mean_score_nontoxic']:.3f} |")
    lines += [
        "",
        f"Final bias score: {summary['final_bias_score']:.3f}",
        "",
        "## Limits and responsible use",
        "",
        "- This is not an automatic moderation decision. A person must review flagged comments.",
        "- Identity terms can raise the score of a non-toxic comment. Read the BPSN column.",
        "- The model knows only the languages in its training data. Read the language table.",
    ]
    return "\n".join(lines) + "\n"


def scores_frame(clf, texts) -> pd.DataFrame:
    return pd.DataFrame(np.round(clf.predict_proba(list(texts)), 4), columns=list(clf.labels))
