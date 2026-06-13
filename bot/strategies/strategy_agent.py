"""Select and coordinate strategy modules based on market context."""

from __future__ import annotations

import pandas as pd

from bot.ai.continuous_learning_ai import ContinuousLearningAI
from bot.ai.strategy_weight_ai import StrategyWeightAI
from bot.core.config import StrategyConfig
from bot.core.models import StrategySignal
from bot.strategies.breakout_atr import BreakoutATRStrategy
from bot.strategies.reversion_rsi import ReversionRSIStrategy
from bot.strategies.trend_vwap import TrendVWAPStrategy


class StrategyAgent:
    """Select and coordinate strategy modules based on market context."""

    def __init__(
        self,
        config: StrategyConfig,
        weight_ai: StrategyWeightAI,
        learner: ContinuousLearningAI | None = None,
    ) -> None:
        self.config = config
        self.weight_ai = weight_ai
        self.learner = learner
        self.strategies = {
            "trend_vwap": TrendVWAPStrategy(),
            "reversion_rsi": ReversionRSIStrategy(),
            "breakout_atr": BreakoutATRStrategy(),
        }

    def generate_candidates(self, row: pd.Series) -> list[StrategySignal]:
        """Collect all valid strategy proposals."""
        candidates: list[StrategySignal] = []
        for name in self.config.enabled_strategies:
            strategy = self.strategies[name]
            signal = strategy.generate_signal(row)
            if signal is not None:
                candidates.append(signal)
        return candidates

    def select_best(self, row: pd.Series) -> StrategySignal | None:
        """Select the strongest weighted candidate."""
        candidates = self.generate_candidates(row)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda signal: (
                signal.confidence
                * self.weight_ai.get_weight(signal.strategy_name)
                * (self.learner.confidence_multiplier(signal) if self.learner is not None else 1.0)
            ),
        )
