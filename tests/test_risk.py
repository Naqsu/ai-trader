"""Tests for risk controls."""

from __future__ import annotations

from bot.core.config import BotConfig
from bot.core.models import StrategySignal
from bot.risk.double_check_agent import DoubleCheckAgent
from bot.risk.risk_manager import RiskManager


def test_risk_manager_sizes_trade_with_positive_quantity() -> None:
    """Risk manager should size valid trades conservatively."""
    config = BotConfig()
    signal = StrategySignal(
        strategy_name="trend_vwap",
        side="long",
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        confidence=0.8,
        risk_reward=2.0,
        timestamp="2025-01-01 12:00:00",
        regime="trend_up",
    )

    valid, _ = DoubleCheckAgent(config.risk).validate(signal)
    sized = RiskManager(config).size_trade(signal, equity=100_000.0)

    assert valid is True
    assert sized.approved is True
    assert sized.position_size > 0
