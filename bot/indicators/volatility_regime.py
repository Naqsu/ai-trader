"""Detect high, low, and unstable volatility conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VolatilityRegimeIndicator:
    """Detect high, low, and unstable volatility conditions."""

    def calculate(self, frame: pd.DataFrame) -> pd.Series:
        """Return a coarse categorical volatility regime."""
        atr = frame["atr_14"]
        atr_mean = atr.rolling(50).mean()
        ratio = atr / atr_mean.replace(0, np.nan)
        regime = pd.Series("normal_vol", index=frame.index, dtype="object")
        regime = regime.mask(ratio >= 1.35, "high_vol")
        regime = regime.mask(ratio <= 0.8, "low_vol")
        regime = regime.fillna("normal_vol")
        return regime
