"""Score trade setup quality before the risk stack evaluates execution."""

from __future__ import annotations

import pandas as pd

from bot.ai.continuous_learning_ai import ContinuousLearningAI
from bot.core.models import SetupAssessment, StrategySignal
from bot.utils.math_utils import MathUtils


class SetupQualityAI:
    """Score trade setup quality before the risk stack evaluates execution."""

    def __init__(self, learner: ContinuousLearningAI | None = None) -> None:
        self.learner = learner

    def assess(self, signal: StrategySignal, row: pd.Series) -> SetupAssessment:
        """Score the trade proposal."""
        score = 0.45
        reasons: list[str] = []

        if signal.risk_reward >= 2.0:
            score += 0.2
            reasons.append("rr_above_2")
        elif signal.risk_reward >= 1.5:
            score += 0.1
            reasons.append("rr_above_1_5")

        if signal.confidence >= 0.7:
            score += 0.15
            reasons.append("high_confidence")

        if signal.side == "long" and row["close"] > row["ema_fast"] > row["ema_slow"]:
            score += 0.15
            reasons.append("trend_alignment_long")
        if signal.side == "short" and row["close"] < row["ema_fast"] < row["ema_slow"]:
            score += 0.15
            reasons.append("trend_alignment_short")

        if row.get("market_regime") in {"trend_up", "trend_down", "range"}:
            score += 0.05
            reasons.append("tradable_regime")

        if self.learner is not None:
            score += self.learner.score_signal_adjustment(signal)
            score -= self.learner.setup_score_penalty(signal)

        normalized = MathUtils.clamp(score, 0.0, 1.0)
        label = "high" if normalized >= 0.75 else "medium" if normalized >= 0.55 else "low"
        return SetupAssessment(score=normalized, label=label, reasons=reasons)
