"""Coordinate the end-to-end decision flow across all agents."""

from __future__ import annotations

import pandas as pd

from bot.ai.continuous_learning_ai import ContinuousLearningAI
from bot.ai.setup_quality_ai import SetupQualityAI
from bot.core.models import DecisionRecord, StrategySignal
from bot.core.state_manager import RuntimeState
from bot.execution.execution_agent import ExecutionAgent
from bot.memory.decision_logger import DecisionLogger
from bot.memory.episodic_market_memory import EpisodicMarketMemory
from bot.memory.trade_logger import TradeLogger
from bot.reporting.runtime_console import RuntimeConsole
from bot.risk.double_check_agent import DoubleCheckAgent
from bot.risk.risk_guardian_agent import RiskGuardianAgent
from bot.risk.risk_manager import RiskManager
from bot.risk.safety_selector import SafetySelector
from bot.strategies.strategy_agent import StrategyAgent


class Orchestrator:
    """Coordinate the end-to-end decision flow across all agents."""

    def __init__(
        self,
        strategy_agent: StrategyAgent,
        setup_quality_ai: SetupQualityAI,
        episodic_memory: EpisodicMarketMemory,
        double_check_agent: DoubleCheckAgent,
        risk_guardian: RiskGuardianAgent,
        risk_manager: RiskManager,
        execution_agent: ExecutionAgent,
        trade_logger: TradeLogger,
        decision_logger: DecisionLogger,
        learner: ContinuousLearningAI | None = None,
        runtime_console: RuntimeConsole | None = None,
    ) -> None:
        self.strategy_agent = strategy_agent
        self.setup_quality_ai = setup_quality_ai
        self.episodic_memory = episodic_memory
        self.double_check_agent = double_check_agent
        self.risk_guardian = risk_guardian
        self.risk_manager = risk_manager
        self.execution_agent = execution_agent
        self.trade_logger = trade_logger
        self.decision_logger = decision_logger
        self.learner = learner
        self.runtime_console = runtime_console or RuntimeConsole()
        self.safety_selector = SafetySelector()

    def evaluate_and_execute(
        self,
        row: pd.Series,
        future_window: pd.DataFrame,
        state: RuntimeState,
    ):
        """Run the full decision flow for a single timestamp."""
        signal = self.safety_selector.choose(self.strategy_agent.select_best(row))
        if signal is None:
            self.decision_logger.log(
                DecisionRecord(
                    timestamp=str(row.name),
                    status="skipped",
                    regime=str(row.get("market_regime", "unknown")),
                    selected_strategy=None,
                    setup_score=None,
                    veto_reason=None,
                    notes=["no_strategy_signal"],
                )
            )
            return None

        return self._process_signal(signal, row, future_window, state)

    def _process_signal(
        self,
        signal: StrategySignal,
        row: pd.Series,
        future_window: pd.DataFrame,
        state: RuntimeState,
    ):
        assessment = self.setup_quality_ai.assess(signal, row)
        self.runtime_console.log_signal(signal, assessment, row)
        self.episodic_memory.remember(row, assessment, signal)

        valid, reason = self.double_check_agent.validate(signal)
        if not valid:
            self.runtime_console.log_block(str(row.name), signal.strategy_name, reason, ["double_check_failed"])
            self.decision_logger.log(
                DecisionRecord(
                    timestamp=str(row.name),
                    status="blocked",
                    regime=str(row.get("market_regime", "unknown")),
                    selected_strategy=signal.strategy_name,
                    setup_score=assessment.score,
                    veto_reason=reason,
                    notes=["double_check_failed"],
                )
            )
            return None

        guardian_decision = self.risk_guardian.review(
            signal=signal,
            assessment=assessment,
            row=row,
            equity=state.equity,
            peak_equity=state.peak_equity,
            day_start_equity=state.day_start_equity,
        )
        if not guardian_decision.approved:
            self.runtime_console.log_block(
                str(row.name),
                signal.strategy_name,
                guardian_decision.veto_reason,
                guardian_decision.warnings,
            )
            self.decision_logger.log(
                DecisionRecord(
                    timestamp=str(row.name),
                    status="blocked",
                    regime=str(row.get("market_regime", "unknown")),
                    selected_strategy=signal.strategy_name,
                    setup_score=assessment.score,
                    veto_reason=guardian_decision.veto_reason,
                    notes=guardian_decision.warnings,
                )
            )
            return None

        sized_decision = self.risk_manager.size_trade(signal, equity=state.equity)
        if not sized_decision.approved:
            self.runtime_console.log_block(str(row.name), signal.strategy_name, sized_decision.veto_reason, ["risk_manager_failed"])
            self.decision_logger.log(
                DecisionRecord(
                    timestamp=str(row.name),
                    status="blocked",
                    regime=str(row.get("market_regime", "unknown")),
                    selected_strategy=signal.strategy_name,
                    setup_score=assessment.score,
                    veto_reason=sized_decision.veto_reason,
                    notes=["risk_manager_failed"],
                )
            )
            return None

        report = self.execution_agent.execute(signal, sized_decision.position_size, future_window)
        self.trade_logger.log(report)
        if self.learner is not None:
            self.learner.observe_trade(report, signal, assessment)
            self.learner.export_state()
        self.decision_logger.log(
            DecisionRecord(
                timestamp=str(row.name),
                status="approved",
                regime=str(row.get("market_regime", "unknown")),
                selected_strategy=signal.strategy_name,
                setup_score=assessment.score,
                veto_reason=None,
                notes=[f"qty={sized_decision.position_size}"],
            )
        )
        self.runtime_console.log_trade(report, state.equity + report.net_pnl)
        if self.learner is not None:
            self.runtime_console.log_learning(
                signal.strategy_name,
                signal.regime,
                self.learner.score_signal_adjustment(signal),
                self.strategy_agent.weight_ai.get_weight(signal.strategy_name),
            )
        return report
