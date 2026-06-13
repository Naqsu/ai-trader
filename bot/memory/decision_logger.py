"""Persist intermediate decisions, filters, and veto reasons."""

from __future__ import annotations

import json
from pathlib import Path

from bot.core.models import DecisionRecord


class DecisionLogger:
    """Persist intermediate decisions, filters, and veto reasons."""

    def __init__(self, path: str = "storage/logs/decisions.jsonl") -> None:
        self.path = Path(path)
        self.records: list[DecisionRecord] = []

    def log(self, record: DecisionRecord) -> None:
        """Append a decision record to memory and disk."""
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict()) + "\n")
