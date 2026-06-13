"""Estimate slippage impact for realistic fills."""

from __future__ import annotations


class SlippageModel:
    """Estimate slippage impact for realistic fills."""

    def __init__(self, tick_size: float, slippage_ticks: int, point_value: float) -> None:
        self.tick_size = tick_size
        self.slippage_ticks = slippage_ticks
        self.point_value = point_value

    def adjust_entry(self, side: str, entry_price: float) -> float:
        """Return a slipped entry price."""
        slip = self.tick_size * self.slippage_ticks
        return entry_price + slip if side == "long" else entry_price - slip

    def estimate_cost(self, quantity: int) -> float:
        """Estimate total slippage cost in dollars."""
        return quantity * self.tick_size * self.slippage_ticks * self.point_value
