"""Build performance summaries for backtesting and paper trading."""

from __future__ import annotations


class PerformanceReport:
    """Build performance summaries for backtesting and paper trading."""

    def build(self, metrics: dict[str, float | int]) -> str:
        """Build a compact performance report text."""
        return (
            f"Trades={metrics['trade_count']} | "
            f"WinRate={metrics['win_rate']:.2%} | "
            f"NetPnL={metrics['net_pnl']:.2f} | "
            f"MaxDD={metrics['max_drawdown']:.2f}"
        )
