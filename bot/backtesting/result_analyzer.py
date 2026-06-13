"""Review backtest outputs and summarize stability characteristics."""

from __future__ import annotations


class ResultAnalyzer:
    """Review backtest outputs and summarize stability characteristics."""

    def summarize(self, metrics: dict[str, float | int]) -> str:
        """Return a short textual summary."""
        return (
            f"Trades={metrics['trade_count']}, "
            f"WinRate={metrics['win_rate']:.2%}, "
            f"NetPnL={metrics['net_pnl']:.2f}, "
            f"MaxDD={metrics['max_drawdown']:.2f}"
        )
