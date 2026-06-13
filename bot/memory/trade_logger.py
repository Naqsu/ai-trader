"""Persist executed trade records for audit and learning loops."""

from __future__ import annotations

import json
from pathlib import Path

from bot.core.models import ExecutionReport


class TradeLogger:
    """Persist executed trade records for audit and learning loops."""

    def __init__(self, path: str = "storage/logs/trades.jsonl") -> None:
        self.path = Path(path)
        self.records: list[ExecutionReport] = []

    def log(self, report: ExecutionReport) -> None:
        """Append a trade record to memory and disk."""
        self.records.append(report)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report.to_dict()) + "\n")
