"""Load NAS100 historical CSV data for initial experiments and backtests."""

from __future__ import annotations

from collections import deque
from io import StringIO
from pathlib import Path

import pandas as pd

from bot.data.data_loader import DataLoader
from bot.utils.time_utils import TimeUtils
from bot.utils.validation import ValidationUtils


class NAS100CSVLoader(DataLoader):
    """Load NAS100 historical CSV data for initial experiments and backtests."""

    COLUMN_MAP = {
        "DateTime": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "TickVolume": "tick_volume",
    }

    def load(self, path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
        """Load and normalize the NAS100 file."""
        raw = self._read_source(path, max_rows=max_rows)
        if len(raw.columns) == 1 and "\t" in raw.columns[0]:
            frame = self._load_tab_wrapped_csv(path, max_rows=max_rows)
        else:
            frame = raw.rename(columns=self.COLUMN_MAP)

        frame["datetime"] = pd.to_datetime(
            frame["datetime"],
            format="%Y.%m.%d %H:%M:%S",
            errors="coerce",
        )
        frame = frame.dropna(subset=["datetime"]).copy()

        numeric_columns = ["open", "high", "low", "close", "volume", "tick_volume"]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()
        frame["volume"] = frame["volume"].fillna(0.0)
        frame["tick_volume"] = frame["tick_volume"].fillna(0.0)

        frame = TimeUtils.ensure_datetime_index(frame, column="datetime")
        frame = frame[~frame.index.duplicated(keep="last")]
        ValidationUtils.validate_ohlcv_frame(frame)
        return frame

    def _load_tab_wrapped_csv(self, path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
        """Parse files where rows are stored as tab-separated text in a single CSV column."""
        with Path(path).open("r", encoding="utf-8") as handle:
            if max_rows is None:
                content = handle.read()
            else:
                header = handle.readline().rstrip("\n")
                lines = deque(handle, maxlen=max_rows)
                content = header + "\n" + "".join(lines)
        frame = pd.read_csv(StringIO(content), sep="\t")
        return frame.rename(columns=self.COLUMN_MAP)

    def _read_source(self, path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
        """Read the raw source with an optional tail limit."""
        if max_rows is None:
            return pd.read_csv(path)
        with Path(path).open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n")
            lines = deque(handle, maxlen=max_rows)
        content = header + "\n" + "".join(lines)
        return pd.read_csv(StringIO(content))
