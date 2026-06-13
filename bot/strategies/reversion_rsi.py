"""Mean-reversion strategy scaffold using RSI context."""

from __future__ import annotations

import pandas as pd

from bot.core.models import StrategySignal
from bot.strategies.base_strategy import BaseStrategy


class ReversionRSIStrategy(BaseStrategy):
    """Mean-reversion strategy scaffold using RSI context."""

    name = "reversion_rsi"

    def generate_signal(self, row: pd.Series) -> StrategySignal | None:
        atr = float(row.get("atr_14", 0.0))
        if atr <= 0:
            return None

        close = float(row["close"])
        rsi = float(row["rsi_14"])
        ema_fast = float(row["ema_fast"])
        regime = str(row.get("market_regime", "unknown"))
        timestamp = str(row.name)

        if regime in {"range", "low_volatility"} and rsi <= 28 and close < ema_fast - atr * 0.35:
            stop = close - atr * 1.0
            target = close + atr * 1.7
            return StrategySignal(
                strategy_name=self.name,
                side="long",
                entry_price=close,
                stop_loss=stop,
                take_profit=target,
                confidence=0.62,
                risk_reward=(target - close) / (close - stop),
                timestamp=timestamp,
                regime=regime,
            )

        if regime in {"range", "low_volatility"} and rsi >= 72 and close > ema_fast + atr * 0.35:
            stop = close + atr * 1.0
            target = close - atr * 1.7
            return StrategySignal(
                strategy_name=self.name,
                side="short",
                entry_price=close,
                stop_loss=stop,
                take_profit=target,
                confidence=0.62,
                risk_reward=(close - target) / (stop - close),
                timestamp=timestamp,
                regime=regime,
            )
        return None
