"""Convert approved trade intents into simulated or live orders."""

from __future__ import annotations

from bot.backtesting.trade_simulator import TradeSimulator
from bot.core.config import BotConfig
from bot.core.models import ExecutionReport, StrategySignal
from bot.execution.commission_model import CommissionModel
from bot.execution.order_manager import OrderManager
from bot.execution.paper_executor import PaperExecutor
from bot.execution.slippage_model import SlippageModel


class ExecutionAgent:
    """Convert approved trade intents into simulated or live orders."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.order_manager = OrderManager()
        self.paper_executor = PaperExecutor(
            simulator=TradeSimulator(
                point_value=config.execution.point_value,
                max_bars_in_trade=config.execution.max_bars_in_trade,
            ),
            commission_model=CommissionModel(config.execution.commission_per_contract),
            slippage_model=SlippageModel(
                tick_size=config.execution.tick_size,
                slippage_ticks=config.execution.slippage_ticks,
                point_value=config.execution.point_value,
            ),
        )

    def execute(self, signal: StrategySignal, quantity: int, future_window) -> ExecutionReport:
        """Execute an approved trade in paper mode."""
        report = self.paper_executor.execute(signal, quantity, future_window)
        self.order_manager.record(report)
        return report
