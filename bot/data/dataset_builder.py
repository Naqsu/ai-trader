"""Assemble research and training datasets for AI components."""

from __future__ import annotations

import pandas as pd


class DatasetBuilder:
    """Assemble research and training datasets for AI components."""

    def merge_timeframes(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Forward-fill higher timeframe context onto the 1m frame."""
        base = frames["1m"].copy()
        for timeframe, frame in frames.items():
            if timeframe == "1m":
                continue
            renamed = frame.add_suffix(f"_{timeframe}")
            aligned = renamed.reindex(base.index, method="ffill")
            base = base.join(aligned, how="left")
        return base.ffill().bfill()
