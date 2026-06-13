"""Compute and coordinate indicator pipelines for strategy evaluation."""

from __future__ import annotations

import pandas as pd

from bot.indicators.atr import ATRIndicator
from bot.indicators.ema import EMAIndicator
from bot.indicators.momentum import MomentumIndicator
from bot.indicators.rsi import RSIIndicator
from bot.indicators.volatility_regime import VolatilityRegimeIndicator
from bot.indicators.vwap import VWAPIndicator


class IndicatorAgent:
    """Compute and coordinate indicator pipelines for strategy evaluation."""

    def __init__(self) -> None:
        self.vwap = VWAPIndicator()
        self.ema = EMAIndicator()
        self.rsi = RSIIndicator()
        self.atr = ATRIndicator()
        self.momentum = MomentumIndicator()
        self.volatility_regime = VolatilityRegimeIndicator()

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach all baseline indicators to the frame."""
        enriched = frame.copy()
        enriched["vwap"] = self.vwap.calculate(enriched)
        enriched["ema_fast"] = self.ema.calculate(enriched, period=21)
        enriched["ema_slow"] = self.ema.calculate(enriched, period=55)
        enriched["rsi_14"] = self.rsi.calculate(enriched, period=14)
        enriched["atr_14"] = self.atr.calculate(enriched, period=14)
        enriched["momentum_10"] = self.momentum.calculate(enriched, periods=10)
        enriched["trend_bias"] = (enriched["ema_fast"] - enriched["ema_slow"]).fillna(0.0)
        enriched["vol_regime"] = self.volatility_regime.calculate(enriched)
        return enriched
