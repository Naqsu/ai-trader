"""Model recurring mistake signatures from historical trading outcomes."""

from __future__ import annotations

from collections import Counter

from bot.core.models import ExecutionReport


class MistakePatternAI:
    """Model recurring mistake signatures from historical trading outcomes."""

    def summarize_losses(self, trades: list[ExecutionReport]) -> dict[str, int]:
        """Count losing trades by strategy."""
        losses = [trade.strategy_name for trade in trades if trade.net_pnl < 0]
        return dict(Counter(losses))
