"""ATR indicator placeholder."""

from __future__ import annotations

import pandas as pd


class ATRIndicator:
    """ATR indicator placeholder."""

    def calculate(self, frame: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate average true range."""
        previous_close = frame["close"].shift(1)
        tr = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous_close).abs(),
                (frame["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()
