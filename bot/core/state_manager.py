"""Persist and restore runtime state for resilient bot operation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeState:
    """Small mutable state bundle for backtests and paper sessions."""

    equity: float
    peak_equity: float
    day_start_equity: float
    open_positions: int = 0


class StateManager:
    """Persist and restore runtime state for resilient bot operation."""

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self.state = RuntimeState(
            equity=initial_capital,
            peak_equity=initial_capital,
            day_start_equity=initial_capital,
        )

    def reset(self) -> RuntimeState:
        """Reset runtime state for a fresh simulation session."""
        self.state = RuntimeState(
            equity=self.initial_capital,
            peak_equity=self.initial_capital,
            day_start_equity=self.initial_capital,
        )
        return self.state

    def apply_trade_result(self, pnl: float) -> RuntimeState:
        """Update state after a trade closes."""
        self.state.equity += pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)
        return self.state

    def mark_new_day(self) -> None:
        """Reset daily baseline."""
        self.state.day_start_equity = self.state.equity
