"""Apply position sizing and portfolio-level risk constraints."""

from __future__ import annotations

from bot.core.config import BotConfig
from bot.core.models import RiskDecision, StrategySignal
from bot.risk.position_sizer import PositionSizer


class RiskManager:
    """Apply position sizing and portfolio-level risk constraints."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.position_sizer = PositionSizer()

    def size_trade(self, signal: StrategySignal, equity: float) -> RiskDecision:
        """Apply conservative sizing limits."""
        risk_pct = self.config.risk.target_risk_per_trade
        quantity = self.position_sizer.calculate(
            equity=equity,
            risk_pct=risk_pct,
            risk_per_unit=signal.risk_per_unit,
            point_value=self.config.execution.point_value,
        )
        if quantity <= 0:
            return RiskDecision(approved=False, veto_reason="position_size_zero")
        return RiskDecision(approved=True, position_size=quantity, risk_pct=risk_pct)
