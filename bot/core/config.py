"""Central configuration model for bot runtime and research settings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RiskConfig:
    """Risk constraints for conservative execution."""

    risk_per_trade_min: float = 0.005
    risk_per_trade_max: float = 0.01
    target_risk_per_trade: float = 0.005
    min_risk_reward: float = 1.5
    min_setup_quality: float = 0.55
    min_signal_confidence: float = 0.55
    max_daily_drawdown_pct: float = 0.03
    max_total_drawdown_pct: float = 0.08
    max_open_positions: int = 1
    allow_high_volatility_breakout_only: bool = True


@dataclass
class DataConfig:
    """Market data and feature-generation settings."""

    dataset_name: str = "NAS100 Historical Price Data"
    dataset_path: str = "1m_data.csv"
    primary_market: str = "NQ"
    max_rows: int | None = 100_000
    timeframes: list[str] = field(
        default_factory=lambda: ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )
    warmup_bars: int = 200


@dataclass
class ExecutionConfig:
    """Execution simulation settings."""

    mode: str = "paper"
    enable_live_execution: bool = False
    double_check_required: bool = True
    risk_guardian_veto_enabled: bool = True
    slippage_ticks: int = 1
    commission_per_contract: float = 2.5
    point_value: float = 20.0
    tick_size: float = 0.25
    max_bars_in_trade: int = 60
    initial_capital: float = 50_000.0


@dataclass
class LearningConfig:
    """Continuous learning settings for online adaptation."""

    enabled: bool = True
    online_window_trades: int = 50
    reinforce_threshold: float = 200.0
    penalize_threshold: float = -200.0
    regime_loss_block_threshold: int = 3
    setup_score_penalty: float = 0.08
    setup_score_reward: float = 0.04
    strategy_weight_step: float = 0.02
    log_every_n_bars: int = 500


@dataclass
class StrategyConfig:
    """Strategy selection and weighting settings."""

    enabled_strategies: list[str] = field(
        default_factory=lambda: ["trend_vwap", "reversion_rsi", "breakout_atr"]
    )
    default_weights: dict[str, float] = field(
        default_factory=lambda: {
            "trend_vwap": 1.0,
            "reversion_rsi": 0.8,
            "breakout_atr": 0.7,
        }
    )


@dataclass
class BotConfig:
    """Central configuration model for bot runtime and research settings."""

    environment: str = "paper"
    supported_markets: list[str] = field(
        default_factory=lambda: ["NQ", "GC", "CL", "SI", "HG", "NG"]
    )
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "BotConfig":
        """Load configuration from JSON file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BotConfig":
        """Load configuration from a dictionary."""
        data_payload = payload.get("data", {})
        if "primary_market" in payload and "primary_market" not in data_payload:
            data_payload["primary_market"] = payload["primary_market"]
        return cls(
            environment=payload.get("environment", "paper"),
            supported_markets=payload.get("supported_markets", ["NQ", "GC", "CL", "SI", "HG", "NG"]),
            data=DataConfig(**data_payload),
            risk=RiskConfig(**payload.get("risk", {})),
            execution=ExecutionConfig(**payload.get("execution", {})),
            learning=LearningConfig(**payload.get("learning", {})),
            strategies=StrategyConfig(**payload.get("strategies", {})),
        )
