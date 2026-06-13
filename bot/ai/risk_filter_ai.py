"""Flag poor conditions and assist the RiskGuardian with warnings."""

from __future__ import annotations

import pandas as pd

from bot.ai.continuous_learning_ai import ContinuousLearningAI
from bot.core.models import StrategySignal


class RiskFilterAI:
    """Flag poor conditions and assist the RiskGuardian with warnings."""

    def __init__(self, learner: ContinuousLearningAI | None = None) -> None:
        self.learner = learner

    def evaluate(self, signal: StrategySignal, row: pd.Series) -> list[str]:
        """Return warnings for unfavorable conditions."""
        warnings: list[str] = []
        regime = str(row.get("market_regime", "unknown"))
        atr = float(row.get("atr_14", 0.0))

        if regime == "chop":
            warnings.append("market_chop")
        if regime == "high_volatility" and signal.strategy_name != "breakout_atr":
            warnings.append("high_vol_only_breakout_allowed")
        if signal.risk_per_unit <= 0:
            warnings.append("invalid_stop_distance")
        if atr > 0 and signal.risk_per_unit > atr * 2.5:
            warnings.append("wide_stop")
        if self.learner is not None:
            warnings.extend(self.learner.risk_flags(signal))
        return warnings
