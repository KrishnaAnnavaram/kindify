"""The moderation service: classify, rewrite with a time limit, check, fall back.

Models load on first use, not at import time. An LLM rewrite that fails, times out or fails a
guard is replaced by the rule rewrite. If that also fails a guard, the service returns no rewrite.
"""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np

from kindify.rewrite.guards import GuardResult, check_rewrite
from kindify.rewrite.prompts import CommentTooLong
from kindify.rewrite.rewriters import HFChatRewriter, OpenAICompatibleRewriter, RewriteError, RuleRewriter

RUN_FILE = "classifier.joblib"


@dataclass
class ModerationResult:
    comment: str
    scores: dict[str, float]
    toxic: bool
    threshold: float
    rewrite: str | None = None
    rewriter: str | None = None
    guard: dict | None = None
    notes: list[str] = field(default_factory=list)


def save_classifier(run_dir: str | Path, classifier, threshold: float, summary: dict) -> Path:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": classifier, "threshold": threshold, "summary": summary}, out / RUN_FILE)
    return out


def load_classifier(run_dir: str | Path) -> dict:
    path = Path(run_dir) / RUN_FILE
    if not path.is_file():
        raise FileNotFoundError(f"no {RUN_FILE} in {run_dir}. Run 'kindify train' first")
    return joblib.load(path)


def make_rewriter(settings):
    if settings.rewriter == "openai":
        return OpenAICompatibleRewriter(settings.llm_base_url, settings.llm_model, timeout=settings.timeout_s)
    if settings.rewriter == "hf":
        return HFChatRewriter(model_name=settings.hf_model)
    return RuleRewriter()


class ModerationService:
    def __init__(self, loader, rewriter=None, threshold: float | None = None, timeout_s: float = 20.0):
        """`loader()` returns a run bundle: {"classifier", "threshold", ...}. It runs on first use."""
        self._loader = loader
        self._bundle = None
        self.rewriter = rewriter or RuleRewriter()
        self.fallback = RuleRewriter()
        self._threshold = threshold
        self.timeout_s = timeout_s

    @property
    def bundle(self) -> dict:
        if self._bundle is None:
            self._bundle = self._loader()
        return self._bundle

    @property
    def threshold(self) -> float:
        return float(self._threshold if self._threshold is not None else self.bundle["threshold"])

    def toxicity(self, text: str) -> float:
        clf = self.bundle["classifier"]
        p = clf.predict_proba([text])[0]
        return float(p[list(clf.labels).index("toxic")])

    def _try(self, rewriter, comment: str) -> tuple[str | None, str | None]:
        """Run one rewriter with a time limit. A slow call is left behind, the request goes on."""
        pool = cf.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(rewriter.rewrite, comment)
            return future.result(timeout=self.timeout_s), None
        except cf.TimeoutError:
            return None, f"{rewriter.name}: timeout after {self.timeout_s:g} s"
        except (RewriteError, CommentTooLong, ValueError) as exc:
            return None, f"{rewriter.name}: {exc}"
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def moderate(self, comment: str) -> ModerationResult:
        clf = self.bundle["classifier"]
        scores = clf.predict_proba([comment])[0]
        score_map = {lab: float(s) for lab, s in zip(clf.labels, np.round(scores, 4))}
        res = ModerationResult(comment, score_map, score_map["toxic"] >= self.threshold, self.threshold)
        if not res.toxic:
            return res
        candidates = [self.rewriter] if self.rewriter.name == "rules" else [self.rewriter, self.fallback]
        for rw in candidates:
            text, note = self._try(rw, comment)
            if note:
                res.notes.append(note)
                continue
            guard: GuardResult = check_rewrite(comment, text, self.toxicity, self.threshold)
            if guard.passed:
                res.rewrite, res.rewriter, res.guard = text, rw.name, asdict(guard)
                return res
            res.notes.append(f"{rw.name}: guard failed: {'; '.join(guard.reasons)}")
        return res
