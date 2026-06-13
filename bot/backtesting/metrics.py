"""Compute metrics for performance, drawdown, and risk quality."""

from __future__ import annotations

from bot.core.models import ExecutionReport


class BacktestMetrics:
    """Compute metrics for performance, drawdown, and risk quality."""

    def calculate(self, trades: list[ExecutionReport], equity_curve: list[float]) -> dict[str, float | int]:
        """Compute conservative headline metrics."""
        total = len(trades)
        wins = sum(1 for trade in trades if trade.net_pnl > 0)
        total_pnl = sum(trade.net_pnl for trade in trades)
        gross_profit = sum(trade.net_pnl for trade in trades if trade.net_pnl > 0)
        gross_loss = abs(sum(trade.net_pnl for trade in trades if trade.net_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
        max_drawdown = self._max_drawdown(equity_curve)
        return {
            "trade_count": total,
            "win_rate": wins / total if total else 0.0,
            "net_pnl": total_pnl,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
        }

    def _max_drawdown(self, equity_curve: list[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_drawdown = 0.0
        for equity in equity_curve:
            peak = max(peak, equity)
            drawdown = peak - equity
            max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown
