"""Simulate fills, stops, targets, and execution frictions."""

from __future__ import annotations

import pandas as pd

from bot.core.models import ExecutionReport, StrategySignal


class TradeSimulator:
    """Simulate fills, stops, targets, and execution frictions."""

    def __init__(self, point_value: float, max_bars_in_trade: int = 60) -> None:
        self.point_value = point_value
        self.max_bars_in_trade = max_bars_in_trade

    def simulate(
        self,
        signal: StrategySignal,
        entry_price: float,
        quantity: int,
        future_window: pd.DataFrame,
    ) -> ExecutionReport:
        """Simulate trade outcome using conservative intrabar assumptions."""
        exit_price = entry_price
        exit_reason = "time_exit"
        bars_held = 0

        for bars_held, (_, row) in enumerate(future_window.head(self.max_bars_in_trade).iterrows(), start=1):
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            if signal.side == "long":
                if low <= signal.stop_loss:
                    exit_price = signal.stop_loss
                    exit_reason = "stop_loss"
                    break
                if high >= signal.take_profit:
                    exit_price = signal.take_profit
                    exit_reason = "take_profit"
                    break
            else:
                if high >= signal.stop_loss:
                    exit_price = signal.stop_loss
                    exit_reason = "stop_loss"
                    break
                if low <= signal.take_profit:
                    exit_price = signal.take_profit
                    exit_reason = "take_profit"
                    break

            exit_price = close

        pnl_points = (
            exit_price - entry_price if signal.side == "long" else entry_price - exit_price
        ) * quantity
        gross_pnl = pnl_points * self.point_value

        return ExecutionReport(
            executed=True,
            side=signal.side,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            quantity=quantity,
            gross_pnl=gross_pnl,
            net_pnl=gross_pnl,
            commission=0.0,
            slippage_cost=0.0,
            exit_reason=exit_reason,
            bars_held=bars_held,
            timestamp=signal.timestamp,
            strategy_name=signal.strategy_name,
            regime=signal.regime,
            metadata=signal.metadata,
        )
