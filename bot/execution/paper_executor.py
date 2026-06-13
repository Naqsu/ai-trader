"""Paper-trading executor with conservative simulation assumptions."""

from __future__ import annotations

from bot.backtesting.trade_simulator import TradeSimulator
from bot.core.models import ExecutionReport, StrategySignal
from bot.execution.commission_model import CommissionModel
from bot.execution.slippage_model import SlippageModel


class PaperExecutor:
    """Paper-trading executor with conservative simulation assumptions."""

    def __init__(
        self,
        simulator: TradeSimulator,
        commission_model: CommissionModel,
        slippage_model: SlippageModel,
    ) -> None:
        self.simulator = simulator
        self.commission_model = commission_model
        self.slippage_model = slippage_model

    def execute(self, signal: StrategySignal, quantity: int, future_window) -> ExecutionReport:
        """Execute the trade proposal with simulated fills."""
        slipped_entry = self.slippage_model.adjust_entry(signal.side, signal.entry_price)
        report = self.simulator.simulate(signal, slipped_entry, quantity, future_window)
        commission = self.commission_model.estimate_round_turn(quantity)
        slippage_cost = self.slippage_model.estimate_cost(quantity)
        report.commission = commission
        report.slippage_cost = slippage_cost
        report.net_pnl = report.gross_pnl - commission - slippage_cost
        return report
