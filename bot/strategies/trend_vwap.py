"""Trend-following strategy scaffold using VWAP context."""

from __future__ import annotations

import pandas as pd

from bot.core.models import StrategySignal
from bot.strategies.base_strategy import BaseStrategy


class TrendVWAPStrategy(BaseStrategy):
    """Trend-following strategy scaffold using VWAP context."""

    name = "trend_vwap"

    def generate_signal(self, row: pd.Series) -> StrategySignal | None:
        atr = float(row.get("atr_14", 0.0))
        if atr <= 0:
            return None

        close = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        vwap = float(row["vwap"])
        regime = str(row.get("market_regime", "unknown"))
        timestamp = str(row.name)

        if regime == "trend_up" and close > vwap and ema_fast > ema_slow:
            stop = close - atr * 1.1
            target = close + atr * 2.0
            return StrategySignal(
                strategy_name=self.name,
                side="long",
                entry_price=close,
                stop_loss=stop,
                take_profit=target,
                confidence=0.68,
                risk_reward=(target - close) / (close - stop),
                timestamp=timestamp,
                regime=regime,
            )

        if regime == "trend_down" and close < vwap and ema_fast < ema_slow:
            stop = close + atr * 1.1
            target = close - atr * 2.0
            return StrategySignal(
                strategy_name=self.name,
                side="short",
                entry_price=close,
                stop_loss=stop,
                take_profit=target,
                confidence=0.68,
                risk_reward=(close - target) / (stop - close),
                timestamp=timestamp,
                regime=regime,
            )
        return None
