"""RSI indicator placeholder."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RSIIndicator:
    """RSI indicator placeholder."""

    def calculate(self, frame: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Wilder-style RSI."""
        delta = frame["close"].diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)
