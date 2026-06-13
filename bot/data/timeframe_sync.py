"""Keep multi-timeframe OHLCV slices aligned for analysis and backtesting."""

from __future__ import annotations

import pandas as pd


class TimeframeSynchronizer:
    """Keep multi-timeframe OHLCV slices aligned for analysis and backtesting."""

    RESAMPLE_RULES = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
        "1w": "1W",
        "1month": "1MS",
    }

    def build_multi_timeframe(self, base_frame: pd.DataFrame, timeframes: list[str]) -> dict[str, pd.DataFrame]:
        """Resample 1m data into a multi-timeframe dictionary."""
        result = {"1m": base_frame.copy()}
        for timeframe in timeframes:
            if timeframe == "1m":
                continue
            rule = self.RESAMPLE_RULES[timeframe]
            frame = (
                base_frame.resample(rule)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                        "tick_volume": "sum",
                    }
                )
                .dropna(subset=["open", "high", "low", "close"])
            )
            result[timeframe] = frame
        return result
