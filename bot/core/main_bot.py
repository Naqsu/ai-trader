"""Top-level facade for conservative paper futures multi-agent trading."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bot.ai.continuous_learning_ai import ContinuousLearningAI
from bot.ai.market_regime_ai import MarketRegimeAI
from bot.ai.setup_quality_ai import SetupQualityAI
from bot.ai.strategy_weight_ai import StrategyWeightAI
from bot.backtesting.backtest_agent import BacktestAgent
from bot.core.config import BotConfig
from bot.core.orchestrator import Orchestrator
from bot.core.state_manager import StateManager
from bot.data.data_agent import DataAgent
from bot.execution.execution_agent import ExecutionAgent
from bot.indicators.indicator_agent import IndicatorAgent
from bot.memory.decision_logger import DecisionLogger
from bot.memory.episodic_market_memory import EpisodicMarketMemory
from bot.memory.mistake_learner import MistakeLearner
from bot.memory.trade_logger import TradeLogger
from bot.reporting.daily_report import DailyReport
from bot.reporting.performance_report import PerformanceReport
from bot.reporting.runtime_console import RuntimeConsole
from bot.risk.double_check_agent import DoubleCheckAgent
from bot.risk.risk_guardian_agent import RiskGuardianAgent
from bot.risk.risk_manager import RiskManager
from bot.strategies.strategy_agent import StrategyAgent


class PaperFuturesMultiAgentBot:
    """Top-level facade for conservative paper futures multi-agent trading."""

    def __init__(self, config: BotConfig | None = None) -> None:
        self.config = config or self._load_default_config()
        self.data_agent = DataAgent(self.config)
        self.indicator_agent = IndicatorAgent()
        self.market_regime_ai = MarketRegimeAI()
        self.continuous_learning_ai = ContinuousLearningAI(self.config.learning)
        self.continuous_learning_ai.load_state()
        self.strategy_weight_ai = StrategyWeightAI(self.config.strategies, self.config.learning)
        self.strategy_weight_ai.load_state()
        self.strategy_agent = StrategyAgent(
            self.config.strategies,
            self.strategy_weight_ai,
            learner=self.continuous_learning_ai,
        )
        self.setup_quality_ai = SetupQualityAI(learner=self.continuous_learning_ai)
        self.episodic_memory = EpisodicMarketMemory()
        self.mistake_learner = MistakeLearner()
        self.double_check_agent = DoubleCheckAgent(self.config.risk)
        self.risk_guardian = RiskGuardianAgent(self.config, learner=self.continuous_learning_ai)
        self.risk_manager = RiskManager(self.config)
        self.execution_agent = ExecutionAgent(self.config)
        self.trade_logger = TradeLogger()
        self.decision_logger = DecisionLogger()
        self.state_manager = StateManager(self.config.execution.initial_capital)
        self.runtime_console = RuntimeConsole()
        self.orchestrator = Orchestrator(
            strategy_agent=self.strategy_agent,
            setup_quality_ai=self.setup_quality_ai,
            episodic_memory=self.episodic_memory,
            double_check_agent=self.double_check_agent,
            risk_guardian=self.risk_guardian,
            risk_manager=self.risk_manager,
            execution_agent=self.execution_agent,
            trade_logger=self.trade_logger,
            decision_logger=self.decision_logger,
            learner=self.continuous_learning_ai,
            runtime_console=self.runtime_console,
        )
        self.backtest_agent = BacktestAgent(
            config=self.config,
            orchestrator=self.orchestrator,
            state_manager=self.state_manager,
            strategy_weight_ai=self.strategy_weight_ai,
            mistake_learner=self.mistake_learner,
        )
        self.performance_report = PerformanceReport()
        self.daily_report = DailyReport()

    def prepare_dataset(self) -> pd.DataFrame:
        """Load, enrich, and label the research dataset."""
        frame = self.data_agent.build_research_frame()
        frame = self.indicator_agent.enrich(frame)
        frame = self.market_regime_ai.annotate(frame)
        required_columns = [
            "close",
            "vwap",
            "ema_fast",
            "ema_slow",
            "rsi_14",
            "atr_14",
            "momentum_10",
            "market_regime",
        ]
        return frame.dropna(subset=required_columns).copy()

    def run_backtest(self, max_rows: int | None = 20_000) -> dict[str, object]:
        """Run a conservative historical backtest."""
        self.state_manager.reset()
        frame = self.prepare_dataset()
        if max_rows is not None:
            frame = frame.tail(max_rows).copy()
        results = self.backtest_agent.run(frame)
        results["performance_report"] = self.performance_report.build(results["metrics"])
        results["daily_report"] = self.daily_report.build(results["supervisor"])
        return results

    def run_paper_trading(
        self,
        latest_bars: int = 500,
        window_end: int | None = None,
        frame: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        """Run the same decision pipeline on the most recent segment."""
        self.state_manager.reset()
        dataset = frame.copy() if frame is not None else self.prepare_dataset()
        if window_end is None:
            selected_frame = dataset.tail(latest_bars).copy()
        else:
            bounded_end = max(latest_bars, min(window_end, len(dataset)))
            bounded_start = max(0, bounded_end - latest_bars)
            selected_frame = dataset.iloc[bounded_start:bounded_end].copy()
        results = self.backtest_agent.run(selected_frame)
        results["performance_report"] = self.performance_report.build(results["metrics"])
        results["daily_report"] = self.daily_report.build(results["supervisor"])
        return results

    def _load_default_config(self) -> BotConfig:
        """Load config.example.json if available, else defaults."""
        path = Path("config.example.json")
        payload: dict[str, object] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

        override_path = Path("storage/state/active_runtime_overrides.json")
        if override_path.exists():
            with override_path.open("r", encoding="utf-8") as handle:
                override_payload = json.load(handle)
            payload = self._deep_merge(payload, override_payload)

        if payload:
            return BotConfig.from_dict(payload)
        return BotConfig()

    def _deep_merge(self, base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
        """Merge nested config dictionaries conservatively."""
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)  # type: ignore[arg-type]
            else:
                merged[key] = value
        return merged
