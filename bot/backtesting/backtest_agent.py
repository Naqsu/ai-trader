"""Run controlled historical simulations against strategy pipelines."""

from __future__ import annotations

import pandas as pd

from bot.ai.meta_supervisor_ai import MetaSupervisorAI
from bot.ai.strategy_weight_ai import StrategyWeightAI
from bot.backtesting.metrics import BacktestMetrics
from bot.backtesting.result_analyzer import ResultAnalyzer
from bot.core.config import BotConfig
from bot.core.orchestrator import Orchestrator
from bot.core.state_manager import StateManager
from bot.memory.mistake_learner import MistakeLearner
from bot.memory.pattern_store import PatternStore
from bot.reporting.runtime_console import RuntimeConsole
from bot.risk.drawdown_guard import DrawdownGuard


class BacktestAgent:
    """Run controlled historical simulations against strategy pipelines."""

    def __init__(
        self,
        config: BotConfig,
        orchestrator: Orchestrator,
        state_manager: StateManager,
        strategy_weight_ai: StrategyWeightAI,
        mistake_learner: MistakeLearner,
    ) -> None:
        self.config = config
        self.orchestrator = orchestrator
        self.state_manager = state_manager
        self.strategy_weight_ai = strategy_weight_ai
        self.mistake_learner = mistake_learner
        self.pattern_store = PatternStore()
        self.metrics = BacktestMetrics()
        self.result_analyzer = ResultAnalyzer()
        self.meta_supervisor = MetaSupervisorAI()
        self.runtime_console = RuntimeConsole()
        self.drawdown_guard = DrawdownGuard(config.risk)

    def run(self, frame: pd.DataFrame) -> dict[str, object]:
        """Run the historical simulation."""
        trades = []
        equity_curve = [self.state_manager.state.equity]
        warmup = self.config.data.warmup_bars
        total_bars = len(frame) - 1
        current_day: pd.Timestamp | None = None

        index = warmup
        while index < len(frame) - 1:
            row = frame.iloc[index]
            row_day = pd.Timestamp(row.name).normalize()
            if current_day is None or row_day != current_day:
                current_day = row_day
                self.state_manager.mark_new_day()
                self.runtime_console.log_info(
                    f"starting session day={row_day.date()} day_start_equity={self.state_manager.state.day_start_equity:.2f}"
                )

            blocked, reason = self.drawdown_guard.is_blocked(
                equity=self.state_manager.state.equity,
                peak_equity=self.state_manager.state.peak_equity,
                day_start_equity=self.state_manager.state.day_start_equity,
            )
            if blocked and reason == "max_daily_drawdown_breached":
                self.runtime_console.log_info(
                    f"daily drawdown lock active for day={row_day.date()}, skipping remaining bars in session"
                )
                index = self._advance_to_next_day(frame, index, row_day)
                continue
            if blocked and reason == "max_total_drawdown_breached":
                self.runtime_console.log_info("total drawdown breached, stopping simulation early")
                break

            if index == warmup or index % self.config.learning.log_every_n_bars == 0:
                self.runtime_console.log_progress(index, total_bars, self.state_manager.state.equity)
            future_window = frame.iloc[index + 1 :]
            report = self.orchestrator.evaluate_and_execute(row, future_window, self.state_manager.state)
            if report is None:
                index += 1
                continue
            trades.append(report)
            self.state_manager.apply_trade_result(report.net_pnl)
            equity_curve.append(self.state_manager.state.equity)
            self.strategy_weight_ai.update_from_trades(trades)
            self.strategy_weight_ai.export_state()
            index += max(1, report.bars_held)

        patterns = self.mistake_learner.review(trades)
        self.pattern_store.save("loss_patterns", patterns)
        metrics = self.metrics.calculate(trades, equity_curve)
        supervisor = self.meta_supervisor.summarize(trades, equity_curve)
        return {
            "metrics": metrics,
            "summary": self.result_analyzer.summarize(metrics),
            "supervisor": supervisor,
            "trades": trades,
            "equity_curve": equity_curve,
        }

    def _advance_to_next_day(self, frame: pd.DataFrame, index: int, current_day: pd.Timestamp) -> int:
        """Move the bar pointer to the first row of the next session day."""
        next_index = index + 1
        while next_index < len(frame):
            next_day = pd.Timestamp(frame.iloc[next_index].name).normalize()
            if next_day != current_day:
                break
            next_index += 1
        return next_index
