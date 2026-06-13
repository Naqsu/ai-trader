"""Estimate fees and commissions per trade."""

from __future__ import annotations


class CommissionModel:
    """Estimate fees and commissions per trade."""

    def __init__(self, commission_per_contract: float) -> None:
        self.commission_per_contract = commission_per_contract

    def estimate_round_turn(self, quantity: int) -> float:
        """Estimate full round-turn commission."""
        return quantity * self.commission_per_contract * 2
