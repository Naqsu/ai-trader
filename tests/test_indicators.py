"""Tests for indicator calculations."""

from __future__ import annotations

import pandas as pd

from bot.indicators.indicator_agent import IndicatorAgent


def test_indicator_agent_adds_expected_columns() -> None:
    """Indicator pipeline should enrich a valid OHLCV frame."""
    index = pd.date_range("2025-01-01", periods=80, freq="min")
    frame = pd.DataFrame(
        {
            "open": [100 + i * 0.1 for i in range(80)],
            "high": [100.5 + i * 0.1 for i in range(80)],
            "low": [99.5 + i * 0.1 for i in range(80)],
            "close": [100.2 + i * 0.1 for i in range(80)],
            "volume": [0 for _ in range(80)],
            "tick_volume": [50 for _ in range(80)],
        },
        index=index,
    )

    enriched = IndicatorAgent().enrich(frame)

    assert {"vwap", "ema_fast", "ema_slow", "rsi_14", "atr_14", "momentum_10", "vol_regime"} <= set(
        enriched.columns
    )
