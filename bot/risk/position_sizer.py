"""Calculate conservative position size from stop distance and risk budget."""

from __future__ import annotations

import math


class PositionSizer:
    """Calculate conservative position size from stop distance and risk budget."""

    def calculate(
        self,
        equity: float,
        risk_pct: float,
        risk_per_unit: float,
        point_value: float,
    ) -> int:
        """Return the number of contracts to trade."""
        if risk_per_unit <= 0 or point_value <= 0:
            return 0
        dollar_risk = equity * risk_pct
        risk_per_contract = risk_per_unit * point_value
        return max(0, math.floor(dollar_risk / risk_per_contract))
