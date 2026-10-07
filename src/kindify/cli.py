"""Command line: synth | train | evaluate | moderate | rewrite-eval | feedback."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from kindify.config import Settings, load_env_file
from kindify.data import TEXT, DataError, load_jigsaw, synthetic_comments, validate
from kindify.experiment import evaluate, evaluate_rewrites, model_card, train
from kindify.feedback import FeedbackStore
from kindify.service import ModerationService, load_classifier, make_rewriter, save_classifier


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="kindify", description=__doc__)
    ap.add_argument("--env-file", default=".env")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="write synthetic labelled comments")
    p.add_argument("--rows", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/synthetic_comments.csv")

    p = sub.add_parser("train", help="train the TF-IDF classifier and write the run folder")
    p.add_argument("--data", help="Jigsaw train.csv (default KINDIFY_DATA, else synthetic rows)")
    p.add_argument("--rows", type=int, default=8000)
    p.add_argument("--seed", type=int)
    p.add_argument("--out", help="run folder (default KINDIFY_RUN_DIR)")

    p = sub.add_parser("evaluate", help="score a saved classifier on a labelled CSV")
    p.add_argument("csv")
    p.add_argument("--run", help="run folder (default KINDIFY_RUN_DIR)")

    p = sub.add_parser("moderate", help="classify one comment and rewrite it when it is toxic")
    p.add_argument("text", nargs="+")
    p.add_argument("--run")

    p = sub.add_parser("rewrite-eval", help="rewrite the toxic comments of a CSV and measure STA, SIM and J")
    p.add_argument("--data", help="labelled CSV (default: synthetic rows)")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--run")
    p.add_argument("--out", help="write the rewrite table to this CSV")

    p = sub.add_parser("feedback", help="opt-in feedback store")
    fsub = p.add_subparsers(dest="action", required=True)
    f = fsub.add_parser("add", help="store a rating (needs --consent)")
    f.add_argument("--comment", required=True)
    f.add_argument("--rewrite", required=True)
    f.add_argument("--rating", type=int, choices=[-1, 1], required=True)
    f.add_argument("--consent", action="store_true")
    fsub.add_parser("purge", help="delete rows older than KINDIFY_RETENTION_DAYS")
    f = fsub.add_parser("export", help="write preference pairs as JSON lines")
    f.add_argument("--out", default="feedback/preferences.jsonl")
    fsub.add_parser("count", help="print the row counts")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(args.env_file)
    try:
        return _run(args, Settings.from_env())
    except (DataError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args, s: Settings) -> int:
    if args.cmd == "synth":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        synthetic_comments(args.rows, seed=args.seed).to_csv(out, index=False)
        print(f"wrote {args.rows} synthetic comments to {out}")
        return 0

    if args.cmd == "train":
        seed = s.seed if args.seed is None else args.seed
        path = args.data or s.data_path
        if path:
            df, test = load_jigsaw(path, s.test_path, s.test_labels_path)
            source = path
        else:
            df, test = validate(synthetic_comments(args.rows, seed=seed)), None
            source = f"synthetic(n={args.rows}, seed={seed})"
        clf, threshold, summary = train(df, test, seed, s.max_tokens)
        out = save_classifier(args.out or s.run_dir, clf, threshold, summary)
        (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        (out / "model_card.md").write_text(model_card(summary, source), encoding="utf-8")
        tox = next(r for r in summary["per_label"] if r["label"] == "toxic")
        t = summary["toxic_at_threshold"]
        print(f"data: {source} | sizes {summary['sizes']}")
        print(f"toxic: roc_auc={tox['roc_auc']:.3f} pr_auc={tox['pr_auc']:.3f} threshold={threshold:.2f} "
              f"precision={t['precision']:.3f} recall={t['recall']:.3f} f1={t['f1']:.3f}")
        print(f"final bias score: {summary['final_bias_score']:.3f}")
        print(f"saved run to {out}")
        return 0

    run = args.run if getattr(args, "run", None) else s.run_dir

    if args.cmd == "evaluate":
        bundle = load_classifier(run)
        df = validate(pd.read_csv(args.csv))
        print(json.dumps(evaluate(bundle["classifier"], df, bundle["threshold"]), indent=2, default=float))
        return 0

    if args.cmd == "moderate":
        service = ModerationService(lambda: load_classifier(run), make_rewriter(s), timeout_s=s.timeout_s)
        print(json.dumps(asdict(service.moderate(" ".join(args.text))), indent=2))
        return 0

    if args.cmd == "rewrite-eval":
        bundle = load_classifier(run)
        df = validate(pd.read_csv(args.data)) if args.data else validate(synthetic_comments(args.rows, seed=s.seed + 1))
        toxic = df.loc[df["toxic"] == 1, TEXT].head(args.limit)
        metrics, tab = evaluate_rewrites(bundle["classifier"], toxic, make_rewriter(s), bundle["threshold"])
        print(json.dumps(metrics, indent=2))
        if args.out:
            tab.to_csv(args.out, index=False)
        return 0

    if args.cmd == "feedback":
        store = FeedbackStore(s.feedback_db, s.retention_days)
        if args.action == "add":
            stored = store.add_rating(args.comment, args.rewrite, args.rating, args.consent)
            print("stored" if stored else "not stored: no consent")
        elif args.action == "purge":
            print(f"removed {store.purge()} rows")
        elif args.action == "export":
            print(f"wrote {store.export_preferences(args.out)} pairs to {args.out}")
        else:
            print(json.dumps(store.counts()))
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
