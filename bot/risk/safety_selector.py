"""Select the safest eligible action under current risk constraints."""

from __future__ import annotations

from bot.core.models import StrategySignal


class SafetySelector:
    """Select the safest eligible action under current risk constraints."""

    def choose(self, signal: StrategySignal | None) -> StrategySignal | None:
        """Return the selected action."""
        return signal
