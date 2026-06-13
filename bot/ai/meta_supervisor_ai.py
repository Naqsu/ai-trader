"""Create oversight summaries without direct trade authority."""

from __future__ import annotations

from bot.core.models import ExecutionReport


class MetaSupervisorAI:
    """Create oversight summaries without direct trade authority."""

    def summarize(self, trades: list[ExecutionReport], equity_curve: list[float]) -> dict[str, float | int]:
        """Create a compact oversight snapshot."""
        wins = sum(1 for trade in trades if trade.net_pnl > 0)
        total = len(trades)
        return {
            "trade_count": total,
            "win_rate": wins / total if total else 0.0,
            "ending_equity": equity_curve[-1] if equity_curve else 0.0,
        }
