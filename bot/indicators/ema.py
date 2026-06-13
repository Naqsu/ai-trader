"""EMA indicator placeholder."""

from __future__ import annotations

import pandas as pd


class EMAIndicator:
    """EMA indicator placeholder."""

    def calculate(self, frame: pd.DataFrame, period: int) -> pd.Series:
        """Calculate EMA on close prices."""
        return frame["close"].ewm(span=period, adjust=False).mean()
