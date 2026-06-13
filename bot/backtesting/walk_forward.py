"""Manage walk-forward validation splits and evaluation flow."""

from __future__ import annotations

import pandas as pd


class WalkForwardAnalyzer:
    """Manage walk-forward validation splits and evaluation flow."""

    def split(self, frame: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return a simple chronological train/test split."""
        cutoff = int(len(frame) * train_ratio)
        return frame.iloc[:cutoff].copy(), frame.iloc[cutoff:].copy()
