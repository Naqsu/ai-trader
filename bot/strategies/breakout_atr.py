"""Breakout strategy scaffold using ATR expansion context."""

from __future__ import annotations

import pandas as pd

from bot.core.models import StrategySignal
from bot.strategies.base_strategy import BaseStrategy


class BreakoutATRStrategy(BaseStrategy):
    """Breakout strategy scaffold using ATR expansion context."""

    name = "breakout_atr"

    def generate_signal(self, row: pd.Series) -> StrategySignal | None:
        atr = float(row.get("atr_14", 0.0))
        if atr <= 0:
            return None

        close = float(row["close"])
        rolling_high = float(row.get("rolling_high_20", close))
        rolling_low = float(row.get("rolling_low_20", close))
        volume_proxy = float(row.get("volume_proxy", 0.0))
        volume_ma = float(row.get("volume_ma_20", 0.0))
        regime = str(row.get("market_regime", "unknown"))
        timestamp = str(row.name)

        if regime in {"high_volatility", "trend_up"} and close >= rolling_high and volume_proxy >= volume_ma:
            stop = close - atr * 1.2
            target = close + atr * 2.4
            return StrategySignal(
                strategy_name=self.name,
                side="long",
                entry_price=close,
                stop_loss=stop,
                take_profit=target,
                confidence=0.64,
                risk_reward=(target - close) / (close - stop),
                timestamp=timestamp,
                regime=regime,
            )

        if regime in {"high_volatility", "trend_down"} and close <= rolling_low and volume_proxy >= volume_ma:
            stop = close + atr * 1.2
            target = close - atr * 2.4
            return StrategySignal(
                strategy_name=self.name,
                side="short",
                entry_price=close,
                stop_loss=stop,
                take_profit=target,
                confidence=0.64,
                risk_reward=(close - target) / (stop - close),
                timestamp=timestamp,
                regime=regime,
            )
        return None
