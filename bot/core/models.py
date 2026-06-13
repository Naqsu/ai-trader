"""Shared domain models used across the trading bot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TradeSide = Literal["long", "short"]
DecisionStatus = Literal["approved", "blocked", "skipped"]


@dataclass(slots=True)
class StrategySignal:
    """Concrete trade proposal produced by a strategy module."""

    strategy_name: str
    side: TradeSide
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_reward: float
    timestamp: str
    regime: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_per_unit(self) -> float:
        """Return absolute stop distance."""
        return abs(self.entry_price - self.stop_loss)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the signal."""
        return asdict(self)


@dataclass(slots=True)
class SetupAssessment:
    """Structured setup quality output."""

    score: float
    label: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the assessment."""
        return asdict(self)


@dataclass(slots=True)
class RiskDecision:
    """Decision emitted by the risk stack."""

    approved: bool
    position_size: int = 0
    risk_pct: float = 0.0
    veto_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision."""
        return asdict(self)


@dataclass(slots=True)
class ExecutionReport:
    """Execution outcome for a single trade."""

    executed: bool
    side: TradeSide
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    quantity: int
    gross_pnl: float
    net_pnl: float
    commission: float
    slippage_cost: float
    exit_reason: str
    bars_held: int
    timestamp: str
    strategy_name: str
    regime: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return asdict(self)


@dataclass(slots=True)
class DecisionRecord:
    """Loggable decision trace for a single bar."""

    timestamp: str
    status: DecisionStatus
    regime: str
    selected_strategy: str | None
    setup_score: float | None
    veto_reason: str | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record."""
        return asdict(self)


@dataclass(slots=True)
class MarketSnapshot:
    """Compact market state snapshot for episodic memory."""

    timestamp: str
    close: float
    rsi: float
    ema_fast: float
    ema_slow: float
    atr: float
    momentum: float
    vol_regime: str
    market_regime: str
    setup_result: str
    strategy_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the snapshot."""
        return asdict(self)
