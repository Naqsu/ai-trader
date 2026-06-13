"""Build the daily oversight report for system behavior and outcomes."""

from __future__ import annotations


class DailyReport:
    """Build the daily oversight report for system behavior and outcomes."""

    def build(self, summary: dict[str, float | int]) -> str:
        """Build a compact daily report text."""
        return (
            f"Trade count: {summary['trade_count']}, "
            f"Win rate: {summary['win_rate']:.2%}, "
            f"Ending equity: {summary['ending_equity']:.2f}"
        )
