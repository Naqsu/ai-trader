"""Utility helpers for timestamps and sessions."""

from __future__ import annotations

import pandas as pd


class TimeUtils:
    """Utility helpers for timestamps and sessions."""

    @staticmethod
    def ensure_datetime_index(frame: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
        """Return a copy with a datetime index."""
        result = frame.copy()
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], utc=False)
            result = result.set_index(column)
        result.index = pd.to_datetime(result.index, utc=False)
        result = result.sort_index()
        return result

    @staticmethod
    def session_label(timestamp: pd.Timestamp) -> str:
        """Return a coarse session label."""
        hour = timestamp.hour
        if 0 <= hour < 7:
            return "asia"
        if 7 <= hour < 13:
            return "europe"
        if 13 <= hour < 21:
            return "us"
        return "overnight"
