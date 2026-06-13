"""Momentum indicator placeholder."""

from __future__ import annotations

import pandas as pd


class MomentumIndicator:
    """Momentum indicator placeholder."""

    def calculate(self, frame: pd.DataFrame, periods: int = 10) -> pd.Series:
        """Calculate percentage momentum."""
        return frame["close"].pct_change(periods=periods)
