"""Utility helpers for input and schema validation."""

from __future__ import annotations

import pandas as pd


class ValidationUtils:
    """Utility helpers for input and schema validation."""

    REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "tick_volume"]

    @classmethod
    def validate_ohlcv_frame(cls, frame: pd.DataFrame) -> None:
        """Raise if the frame is missing required columns."""
        missing = [column for column in cls.REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing OHLCV columns: {missing}")

        if frame.empty:
            raise ValueError("OHLCV frame is empty.")
