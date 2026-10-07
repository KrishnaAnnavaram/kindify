"""Settings from environment variables (and an optional local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PREFIX = "KINDIFY_"
CLASSIFIERS = ("tfidf", "transformer")
REWRITERS = ("rules", "openai", "hf")


def load_env_file(path: str | os.PathLike = ".env") -> int:
    """Read KEY=VALUE lines into os.environ. Existing variables win. Returns the count of new keys."""
    p = Path(path)
    if not p.is_file():
        return 0
    added = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        value = value.strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            added += 1
    return added


def _get(name: str, default: str = "") -> str:
    value = os.environ.get(PREFIX + name, "")
    return value.strip() if value.strip() else default


@dataclass(frozen=True)
class Settings:
    data_path: str | None
    test_path: str | None
    test_labels_path: str | None
    run_dir: str
    seed: int
    classifier: str
    transformer_model: str | None
    max_tokens: int
    rewriter: str
    llm_base_url: str
    llm_model: str
    hf_model: str | None
    timeout_s: float
    feedback_db: str
    retention_days: int

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls(
            data_path=_get("DATA") or None,
            test_path=_get("TEST_DATA") or None,
            test_labels_path=_get("TEST_LABELS") or None,
            run_dir=_get("RUN_DIR", "runs/latest"),
            seed=int(_get("SEED", "42")),
            classifier=_get("CLASSIFIER", "tfidf").lower(),
            transformer_model=_get("TRANSFORMER_MODEL") or None,
            max_tokens=int(_get("MAX_TOKENS", "256")),
            rewriter=_get("REWRITER", "rules").lower(),
            llm_base_url=_get("LLM_BASE_URL", "http://localhost:11434/v1"),
            llm_model=_get("LLM_MODEL", "llama3.2"),
            hf_model=_get("HF_MODEL") or None,
            timeout_s=float(_get("TIMEOUT_S", "20")),
            feedback_db=_get("FEEDBACK_DB", "feedback/kindify.sqlite"),
            retention_days=int(_get("RETENTION_DAYS", "30")),
        )
        s.check()
        return s

    def check(self) -> None:
        if self.classifier not in CLASSIFIERS:
            raise ValueError(f"KINDIFY_CLASSIFIER must be one of {CLASSIFIERS}")
        if self.rewriter not in REWRITERS:
            raise ValueError(f"KINDIFY_REWRITER must be one of {REWRITERS}")
        if self.classifier == "transformer" and not self.transformer_model:
            raise ValueError("KINDIFY_CLASSIFIER=transformer needs KINDIFY_TRANSFORMER_MODEL")
        if self.rewriter == "hf" and not self.hf_model:
            raise ValueError("KINDIFY_REWRITER=hf needs KINDIFY_HF_MODEL")
        if self.max_tokens < 16:
            raise ValueError("KINDIFY_MAX_TOKENS must be 16 or more")
        if self.timeout_s <= 0 or self.retention_days < 1:
            raise ValueError("KINDIFY_TIMEOUT_S must be above 0 and KINDIFY_RETENTION_DAYS at least 1")
