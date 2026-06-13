"""Base loader for standardized OHLCV dataset ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoader:
    """Base loader for standardized OHLCV dataset ingestion."""

    def load(self, path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
        """Load a dataset from disk."""
        raise NotImplementedError
