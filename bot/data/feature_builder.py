"""Build model-ready features from synchronized multi-timeframe market data."""

from __future__ import annotations

import pandas as pd

from bot.utils.time_utils import TimeUtils


class FeatureBuilder:
    """Build model-ready features from synchronized multi-timeframe market data."""

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Create conservative baseline features from OHLCV data."""
        enriched = frame.copy()
        enriched["return_1"] = enriched["close"].pct_change()
        enriched["return_5"] = enriched["close"].pct_change(5)
        enriched["range"] = enriched["high"] - enriched["low"]
        enriched["body"] = enriched["close"] - enriched["open"]
        enriched["upper_wick"] = enriched["high"] - enriched[["open", "close"]].max(axis=1)
        enriched["lower_wick"] = enriched[["open", "close"]].min(axis=1) - enriched["low"]
        enriched["rolling_high_20"] = enriched["high"].rolling(20).max()
        enriched["rolling_low_20"] = enriched["low"].rolling(20).min()
        enriched["rolling_range_20"] = enriched["range"].rolling(20).mean()
        enriched["volume_proxy"] = enriched["volume"].where(enriched["volume"] > 0, enriched["tick_volume"])
        enriched["volume_ma_20"] = enriched["volume_proxy"].rolling(20).mean()
        enriched["hour"] = enriched.index.hour
        enriched["day_of_week"] = enriched.index.dayofweek
        enriched["session"] = [TimeUtils.session_label(ts) for ts in enriched.index]
        return enriched
