"""Evaluate fill quality and execution conditions after trades."""

from __future__ import annotations

from bot.core.models import ExecutionReport


class ExecutionQualityAI:
    """Evaluate fill quality and execution conditions after trades."""

    def evaluate(self, report: ExecutionReport) -> dict[str, float | str]:
        """Return compact execution diagnostics."""
        return {
            "slippage_cost": report.slippage_cost,
            "commission": report.commission,
            "net_pnl": report.net_pnl,
            "exit_reason": report.exit_reason,
        }
