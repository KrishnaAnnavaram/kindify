"""Opt-in feedback store (SQLite) with redaction, retention and preference-pair export.

* Nothing is stored without `consent=True`.
* E-mail addresses, URLs, @handles, phone numbers and IP addresses are replaced before storage.
* `purge` deletes rows older than the retention period.
* `export_preferences` writes `prompt, chosen, rejected` JSON lines for a later preference
  fine-tune (for example DPO). This package does not run that fine-tune.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

REDACTIONS = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"https?://\S+|www\.\S+"), "[url]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[ip]"),
    (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "[phone]"),
    (re.compile(r"(?<!\w)@\w+"), "[user]"),
)


def redact(text: str) -> str:
    out = str(text)
    for pattern, repl in REDACTIONS:
        out = pattern.sub(repl, out)
    return out


class FeedbackStore:
    def __init__(self, path: str | Path, retention_days: int = 30):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS ratings (id INTEGER PRIMARY KEY, created TEXT, comment TEXT, "
                "rewrite TEXT, rating INTEGER)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS preferences (id INTEGER PRIMARY KEY, created TEXT, comment TEXT, "
                "chosen TEXT, rejected TEXT)"
            )

    def _conn(self):
        return sqlite3.connect(self.path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_rating(self, comment: str, rewrite: str, rating: int, consent: bool) -> bool:
        if not consent:
            return False
        if rating not in (-1, 1):
            raise ValueError("rating must be -1 or 1")
        with self._conn() as c:
            c.execute("INSERT INTO ratings (created, comment, rewrite, rating) VALUES (?, ?, ?, ?)",
                      (self._now(), redact(comment), redact(rewrite), rating))
        return True

    def add_preference(self, comment: str, chosen: str, rejected: str, consent: bool) -> bool:
        if not consent:
            return False
        if chosen.strip() == rejected.strip():
            raise ValueError("chosen and rejected must differ")
        with self._conn() as c:
            c.execute("INSERT INTO preferences (created, comment, chosen, rejected) VALUES (?, ?, ?, ?)",
                      (self._now(), redact(comment), redact(chosen), redact(rejected)))
        return True

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("ratings", "preferences")}

    def purge(self, now: datetime | None = None) -> int:
        cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=self.retention_days)).isoformat()
        removed = 0
        with self._conn() as c:
            for t in ("ratings", "preferences"):
                removed += c.execute(f"DELETE FROM {t} WHERE created < ?", (cutoff,)).rowcount
        return removed

    def export_preferences(self, out: str | Path) -> int:
        """Preference pairs from `preferences` and from ratings (a liked and a disliked rewrite of one comment)."""
        with self._conn() as c:
            pairs = [{"prompt": p, "chosen": ch, "rejected": rj}
                     for p, ch, rj in c.execute("SELECT comment, chosen, rejected FROM preferences")]
            rows = c.execute("SELECT comment, rewrite, rating FROM ratings").fetchall()
        by_comment: dict[str, dict[int, list[str]]] = {}
        for comment, rewrite, rating in rows:
            by_comment.setdefault(comment, {1: [], -1: []})[rating].append(rewrite)
        for comment, d in by_comment.items():
            for good in d[1]:
                for bad in d[-1]:
                    if good != bad:
                        pairs.append({"prompt": comment, "chosen": good, "rejected": bad})
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        return len(pairs)
