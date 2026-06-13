"""Store contextual snapshots around each setup and trade decision."""

from __future__ import annotations

import pandas as pd

from bot.core.models import MarketSnapshot, SetupAssessment, StrategySignal


class EpisodicMarketMemory:
    """Store contextual snapshots around each setup and trade decision."""

    def __init__(self) -> None:
        self.snapshots: list[MarketSnapshot] = []

    def remember(self, row: pd.Series, assessment: SetupAssessment, signal: StrategySignal | None) -> None:
        """Store a compact view of the current market context."""
        snapshot = MarketSnapshot(
            timestamp=str(row.name),
            close=float(row["close"]),
            rsi=float(row.get("rsi_14", 50.0)),
            ema_fast=float(row.get("ema_fast", row["close"])),
            ema_slow=float(row.get("ema_slow", row["close"])),
            atr=float(row.get("atr_14", 0.0)),
            momentum=float(row.get("momentum_10", 0.0)),
            vol_regime=str(row.get("vol_regime", "unknown")),
            market_regime=str(row.get("market_regime", "unknown")),
            setup_result=assessment.label,
            strategy_name=signal.strategy_name if signal else None,
        )
        self.snapshots.append(snapshot)
