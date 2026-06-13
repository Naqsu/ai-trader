"""Common strategy contract for all trading approaches."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from bot.core.models import StrategySignal


class BaseStrategy(ABC):
    """Common strategy contract for all trading approaches."""

    name: str

    @abstractmethod
    def generate_signal(self, row: pd.Series) -> StrategySignal | None:
        """Return a trade signal when conditions are met."""
