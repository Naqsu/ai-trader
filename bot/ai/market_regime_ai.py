"""Classify trend, range, chop, and volatility regimes."""

from __future__ import annotations

import pandas as pd


class MarketRegimeAI:
    """Classify trend, range, chop, and volatility regimes."""

    def annotate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach a market regime label to the frame."""
        enriched = frame.copy()
        regimes = []
        for _, row in enriched.iterrows():
            regimes.append(self.classify_row(row))
        enriched["market_regime"] = regimes
        return enriched

    def classify_row(self, row: pd.Series) -> str:
        """Classify the current row into a coarse regime."""
        trend_bias = float(row.get("trend_bias", 0.0))
        momentum = float(row.get("momentum_10", 0.0))
        rsi = float(row.get("rsi_14", 50.0))
        vol_regime = str(row.get("vol_regime", "normal_vol"))

        if vol_regime == "high_vol" and abs(momentum) < 0.0008:
            return "chop"
        if vol_regime == "high_vol":
            return "high_volatility"
        if vol_regime == "low_vol" and abs(trend_bias) < max(float(row.get("atr_14", 0.0)) * 0.2, 0.5):
            return "low_volatility"
        if trend_bias > 0 and momentum > 0 and rsi >= 52:
            return "trend_up"
        if trend_bias < 0 and momentum < 0 and rsi <= 48:
            return "trend_down"
        if 42 <= rsi <= 58:
            return "range"
        return "chop"
