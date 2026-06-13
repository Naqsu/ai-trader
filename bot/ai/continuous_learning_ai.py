"""Continuous online learning over trade outcomes and decision contexts."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import datetime
from statistics import mean
from pathlib import Path

from bot.core.config import LearningConfig
from bot.core.models import ExecutionReport, SetupAssessment, StrategySignal
from bot.utils.file_utils import FileUtils


class ContinuousLearningAI:
    """Maintain lightweight online knowledge for runtime adaptation."""

    def __init__(self, config: LearningConfig) -> None:
        self.config = config
        self.recent_trades: deque[dict[str, object]] = deque(maxlen=config.online_window_trades)
        self.strategy_regime_pnls: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=config.online_window_trades)
        )
        self.strategy_regime_hour_pnls: dict[tuple[str, str, int], deque[float]] = defaultdict(
            lambda: deque(maxlen=config.online_window_trades)
        )
        self.strategy_regime_losses: Counter[tuple[str, str]] = Counter()
        self.strategy_regime_wins: Counter[tuple[str, str]] = Counter()
        self.setup_bias: dict[tuple[str, str], float] = defaultdict(float)

    def score_signal_adjustment(self, signal: StrategySignal) -> float:
        """Return setup score adjustment learned from recent outcomes."""
        return self.setup_bias[(signal.strategy_name, signal.regime)]

    def setup_score_penalty(self, signal: StrategySignal) -> float:
        """Return an additional soft penalty before hard veto is considered."""
        key = (signal.strategy_name, signal.regime)
        history = list(self.strategy_regime_pnls[key])
        if not history:
            return 0.0

        penalty = 0.0
        recent = history[-self.config.regime_loss_block_threshold :]
        if len(recent) >= self.config.regime_loss_block_threshold and all(pnl < 0 for pnl in recent):
            penalty += 0.03

        sample = history[-min(len(history), 8) :]
        if len(sample) >= 6:
            win_rate = sum(1 for pnl in sample if pnl > 0) / len(sample)
            avg_pnl = mean(sample)
            if win_rate <= 0.35 and avg_pnl < 0:
                penalty += 0.03

        hour_key = (signal.strategy_name, signal.regime, self._extract_hour(signal.timestamp))
        hour_sample = list(self.strategy_regime_hour_pnls[hour_key])[-6:]
        if len(hour_sample) >= 4:
            hour_win_rate = sum(1 for pnl in hour_sample if pnl > 0) / len(hour_sample)
            hour_avg_pnl = mean(hour_sample)
            if hour_win_rate <= 0.25 and hour_avg_pnl < 0:
                penalty += 0.02

        return min(0.08, penalty)

    def confidence_multiplier(self, signal: StrategySignal) -> float:
        """Return a conservative multiplier for strategy ranking."""
        key = (signal.strategy_name, signal.regime)
        history = self.strategy_regime_pnls[key]
        if not history:
            return 1.0
        avg_pnl = mean(history)
        recent = list(history)[-self.config.regime_loss_block_threshold :]
        if len(recent) >= self.config.regime_loss_block_threshold and all(pnl < 0 for pnl in recent):
            return 0.82
        if avg_pnl <= self.config.penalize_threshold:
            return 0.9
        if avg_pnl >= self.config.reinforce_threshold:
            return 1.05
        return 1.0

    def risk_flags(self, signal: StrategySignal) -> list[str]:
        """Return dynamic risk flags from learned bad patterns."""
        key = (signal.strategy_name, signal.regime)
        history = list(self.strategy_regime_pnls[key])
        if not history:
            history = []
        warnings: list[str] = []

        hard_recent = history[-(self.config.regime_loss_block_threshold + 2) :]
        if len(hard_recent) >= self.config.regime_loss_block_threshold + 2 and all(pnl < 0 for pnl in hard_recent):
            warnings.append("continuous_learning_loss_cluster")

        sample = history[-min(len(history), 12) :]
        if sample:
            win_rate = sum(1 for pnl in sample if pnl > 0) / len(sample)
            avg_pnl = mean(sample)
            if len(sample) >= 10 and win_rate <= 0.2 and avg_pnl <= self.config.penalize_threshold / 2:
                warnings.append("continuous_learning_negative_expectancy")

        hour_key = (signal.strategy_name, signal.regime, self._extract_hour(signal.timestamp))
        hour_history = list(self.strategy_regime_hour_pnls[hour_key])
        if hour_history:
            hour_sample = hour_history[-min(len(hour_history), 6) :]
            if hour_sample:
                hour_win_rate = sum(1 for pnl in hour_sample if pnl > 0) / len(hour_sample)
                hour_avg_pnl = mean(hour_sample)
                if len(hour_sample) >= 6 and hour_win_rate <= 0.15 and hour_avg_pnl < 0:
                    warnings.append("continuous_learning_bad_hour")
        return warnings

    def observe_trade(
        self,
        report: ExecutionReport,
        signal: StrategySignal,
        assessment: SetupAssessment,
    ) -> None:
        """Update learned state from a finished trade."""
        if not self.config.enabled:
            return

        key = (report.strategy_name, report.regime)
        hour_key = (report.strategy_name, report.regime, self._extract_hour(report.timestamp))
        pnl = float(report.net_pnl)
        self.recent_trades.append(
            {
                "strategy": report.strategy_name,
                "regime": report.regime,
                "pnl": pnl,
                "timestamp": report.timestamp,
                "setup_score": assessment.score,
                "confidence": signal.confidence,
            }
        )
        self.strategy_regime_pnls[key].append(pnl)
        self.strategy_regime_hour_pnls[hour_key].append(pnl)

        if pnl < 0:
            self.strategy_regime_losses[key] += 1
            penalty = self.config.setup_score_penalty if pnl <= self.config.penalize_threshold else self.config.setup_score_penalty / 2
            self.setup_bias[key] = max(
                -0.12,
                self.setup_bias[key] - penalty,
            )
        else:
            self.strategy_regime_wins[key] += 1
            reward = self.config.setup_score_reward if pnl >= self.config.reinforce_threshold else self.config.setup_score_reward / 2
            self.setup_bias[key] = min(
                0.12,
                self.setup_bias[key] + reward,
            )

    def export_state(self, path: str = "storage/state/continuous_learning_state.json") -> None:
        """Persist lightweight learning state for reuse."""
        payload = {
            "setup_bias": {
                f"{strategy}|{regime}": bias
                for (strategy, regime), bias in self.setup_bias.items()
            },
            "loss_clusters": {
                f"{strategy}|{regime}": count
                for (strategy, regime), count in self.strategy_regime_losses.items()
            },
            "win_clusters": {
                f"{strategy}|{regime}": count
                for (strategy, regime), count in self.strategy_regime_wins.items()
            },
            "hourly_pnls": {
                f"{strategy}|{regime}|{hour}": list(pnls)
                for (strategy, regime, hour), pnls in self.strategy_regime_hour_pnls.items()
            },
            "recent_trades": list(self.recent_trades),
        }
        FileUtils.write_json(path, payload)

    def load_state(self, path: str = "storage/state/continuous_learning_state.json") -> None:
        """Restore previously learned state if available."""
        resolved = Path(path)
        if not resolved.exists():
            return
        payload = FileUtils.read_json(path)
        for item in payload.get("recent_trades", []):
            self.recent_trades.append(item)
            strategy = str(item.get("strategy", "unknown"))
            regime = str(item.get("regime", "unknown"))
            pnl = float(item.get("pnl", 0.0))
            self.strategy_regime_pnls[(strategy, regime)].append(pnl)
            hour = self._extract_hour(str(item.get("timestamp", "1970-01-01 00:00:00")))
            self.strategy_regime_hour_pnls[(strategy, regime, hour)].append(pnl)
        self._recompute_from_recent_trades()

    def _extract_hour(self, timestamp: str) -> int:
        """Extract session hour from timestamp strings used by the bot."""
        try:
            return datetime.fromisoformat(timestamp).hour
        except ValueError:
            return int(str(timestamp).split(" ")[1].split(":")[0])

    def _recompute_from_recent_trades(self) -> None:
        """Rebuild rolling learning state from the recent trade window only."""
        self.setup_bias.clear()
        self.strategy_regime_losses.clear()
        self.strategy_regime_wins.clear()

        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for item in self.recent_trades:
            key = (str(item.get("strategy", "unknown")), str(item.get("regime", "unknown")))
            pnl = float(item.get("pnl", 0.0))
            grouped[key].append(pnl)
            if pnl < 0:
                self.strategy_regime_losses[key] += 1
            elif pnl > 0:
                self.strategy_regime_wins[key] += 1

        for key, pnls in grouped.items():
            sample = pnls[-min(len(pnls), 10) :]
            if not sample:
                continue
            avg_pnl = mean(sample)
            win_rate = sum(1 for pnl in sample if pnl > 0) / len(sample)
            if len(sample) >= 4 and win_rate <= 0.35 and avg_pnl < 0:
                self.setup_bias[key] = -0.08
            elif avg_pnl >= max(50.0, self.config.reinforce_threshold / 4):
                self.setup_bias[key] = 0.04
