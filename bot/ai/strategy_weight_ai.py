"""Adjust strategy weights slowly under conservative governance."""

from __future__ import annotations

from pathlib import Path

from bot.core.config import LearningConfig, StrategyConfig
from bot.core.models import ExecutionReport
from bot.utils.file_utils import FileUtils
from bot.utils.math_utils import MathUtils


class StrategyWeightAI:
    """Adjust strategy weights slowly under conservative governance."""

    def __init__(self, config: StrategyConfig, learning_config: LearningConfig | None = None) -> None:
        self.weights = dict(config.default_weights)
        self.learning_config = learning_config or LearningConfig()

    def get_weight(self, strategy_name: str) -> float:
        """Return current weight for strategy selection."""
        return self.weights.get(strategy_name, 0.5)

    def update_from_trades(self, trades: list[ExecutionReport]) -> None:
        """Adjust weights conservatively from recent results."""
        recent = trades[-20:]
        if not recent:
            return
        by_strategy: dict[str, list[float]] = {}
        for trade in recent:
            by_strategy.setdefault(trade.strategy_name, []).append(trade.net_pnl)
        for strategy_name, pnls in by_strategy.items():
            avg_pnl = sum(pnls) / len(pnls)
            current = self.weights.get(strategy_name, 0.5)
            delta = self.learning_config.strategy_weight_step if avg_pnl > 0 else -self.learning_config.strategy_weight_step
            self.weights[strategy_name] = MathUtils.clamp(current + delta, 0.4, 1.2)

    def export_state(self, path: str = "storage/state/strategy_weights.json") -> None:
        """Persist current weights."""
        FileUtils.write_json(path, self.weights)

    def load_state(self, path: str = "storage/state/strategy_weights.json") -> None:
        """Restore weights if available."""
        resolved = Path(path)
        if not resolved.exists():
            return
        payload = FileUtils.read_json(path)
        for name, value in payload.items():
            self.weights[name] = float(value)
