"""VWAP indicator placeholder."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VWAPIndicator:
    """VWAP indicator placeholder."""

    def calculate(self, frame: pd.DataFrame) -> pd.Series:
        """Return session-agnostic cumulative VWAP."""
        price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
        volume = frame["volume"].where(frame["volume"] > 0, frame["tick_volume"]).replace(0, np.nan)
        numerator = (price * volume).cumsum()
        denominator = volume.cumsum().replace(0, np.nan)
        return numerator / denominator
