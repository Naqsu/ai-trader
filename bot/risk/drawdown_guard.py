"""Stop or reduce trading activity when drawdown limits are breached."""

from __future__ import annotations

from bot.core.config import RiskConfig


class DrawdownGuard:
    """Stop or reduce trading activity when drawdown limits are breached."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def is_blocked(self, equity: float, peak_equity: float, day_start_equity: float) -> tuple[bool, str | None]:
        """Return whether trading should be blocked due to drawdown."""
        total_drawdown = (peak_equity - equity) / peak_equity if peak_equity else 0.0
        daily_drawdown = (day_start_equity - equity) / day_start_equity if day_start_equity else 0.0
        if total_drawdown >= self.config.max_total_drawdown_pct:
            return True, "max_total_drawdown_breached"
        if daily_drawdown >= self.config.max_daily_drawdown_pct:
            return True, "max_daily_drawdown_breached"
        return False, None
