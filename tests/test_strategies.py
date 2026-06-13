"""Tests for strategies."""

from __future__ import annotations

import pandas as pd

from bot.ai.market_regime_ai import MarketRegimeAI
from bot.strategies.strategy_agent import StrategyAgent
from bot.ai.strategy_weight_ai import StrategyWeightAI
from bot.core.config import StrategyConfig


def test_strategy_agent_selects_signal_for_trend_context() -> None:
    """Strategy agent should emit a signal in a clean trend row."""
    row = pd.Series(
        {
            "close": 101.0,
            "vwap": 100.4,
            "ema_fast": 100.8,
            "ema_slow": 100.0,
            "atr_14": 1.0,
            "rsi_14": 58.0,
            "momentum_10": 0.01,
            "trend_bias": 0.8,
            "vol_regime": "normal_vol",
            "rolling_high_20": 101.0,
            "rolling_low_20": 98.5,
            "volume_proxy": 80.0,
            "volume_ma_20": 50.0,
        },
        name=pd.Timestamp("2025-01-01 12:00:00"),
    )
    row["market_regime"] = MarketRegimeAI().classify_row(row)

    agent = StrategyAgent(StrategyConfig(), StrategyWeightAI(StrategyConfig()))
    signal = agent.select_best(row)

    assert signal is not None
    assert signal.strategy_name in {"trend_vwap", "breakout_atr"}
