"""Analyze repeated losses and surface conservative corrective hints."""

from __future__ import annotations

from bot.ai.mistake_pattern_ai import MistakePatternAI
from bot.core.models import ExecutionReport


class MistakeLearner:
    """Analyze repeated losses and surface conservative corrective hints."""

    def __init__(self) -> None:
        self.pattern_ai = MistakePatternAI()

    def review(self, trades: list[ExecutionReport]) -> dict[str, int]:
        """Summarize recurring loss patterns."""
        return self.pattern_ai.summarize_losses(trades)
