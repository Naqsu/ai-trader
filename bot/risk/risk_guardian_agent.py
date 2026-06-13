"""Final hard-veto layer that can block any trade proposal."""

from __future__ import annotations

import pandas as pd

from bot.ai.continuous_learning_ai import ContinuousLearningAI
from bot.ai.risk_filter_ai import RiskFilterAI
from bot.core.config import BotConfig
from bot.core.models import RiskDecision, SetupAssessment, StrategySignal
from bot.risk.drawdown_guard import DrawdownGuard


class RiskGuardianAgent:
    """Final hard-veto layer that can block any trade proposal."""

    def __init__(self, config: BotConfig, learner: ContinuousLearningAI | None = None) -> None:
        self.config = config
        self.risk_filter = RiskFilterAI(learner=learner)
        self.drawdown_guard = DrawdownGuard(config.risk)

    def review(
        self,
        signal: StrategySignal,
        assessment: SetupAssessment,
        row: pd.Series,
        equity: float,
        peak_equity: float,
        day_start_equity: float,
    ) -> RiskDecision:
        """Apply the hard-veto layer."""
        blocked, reason = self.drawdown_guard.is_blocked(equity, peak_equity, day_start_equity)
        if blocked:
            return RiskDecision(approved=False, veto_reason=reason)

        warnings = self.risk_filter.evaluate(signal, row)
        if assessment.score < self.config.risk.min_setup_quality:
            return RiskDecision(approved=False, veto_reason="setup_quality_too_low", warnings=warnings)
        if "market_chop" in warnings:
            return RiskDecision(approved=False, veto_reason="risk_guardian_rejected_chop", warnings=warnings)
        if "continuous_learning_loss_cluster" in warnings and assessment.score < 0.72:
            return RiskDecision(
                approved=False,
                veto_reason="risk_guardian_rejected_loss_cluster",
                warnings=warnings,
            )
        if "continuous_learning_negative_expectancy" in warnings and assessment.score < 0.78:
            return RiskDecision(
                approved=False,
                veto_reason="risk_guardian_rejected_negative_expectancy",
                warnings=warnings,
            )
        if "continuous_learning_bad_hour" in warnings and assessment.score < 0.70:
            return RiskDecision(
                approved=False,
                veto_reason="risk_guardian_rejected_bad_hour",
                warnings=warnings,
            )
        if (
            self.config.risk.allow_high_volatility_breakout_only
            and "high_vol_only_breakout_allowed" in warnings
        ):
            return RiskDecision(approved=False, veto_reason="risk_guardian_rejected_high_vol", warnings=warnings)
        return RiskDecision(approved=True, warnings=warnings)
