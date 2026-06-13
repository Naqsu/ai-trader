"""Tests for data loading."""

from __future__ import annotations

from pathlib import Path

from bot.data.nas100_csv_loader import NAS100CSVLoader


def test_nas100_loader_parses_tab_wrapped_csv(tmp_path: Path) -> None:
    """Loader should parse the provided tab-wrapped export format."""
    sample = (
        "DateTime\tOpen\tHigh\tLow\tClose\tVolume\tTickVolume\n"
        "2025.10.01 07:12:00\t24584.9\t24586.5\t24584.5\t24585.7\t0\t60\n"
        "2025.10.01 07:11:00\t24587.7\t24587.7\t24584.7\t24585.0\t0\t50\n"
    )
    path = tmp_path / "sample.csv"
    path.write_text(sample, encoding="utf-8")

    frame = NAS100CSVLoader().load(path)

    assert list(frame.columns) == ["open", "high", "low", "close", "volume", "tick_volume"]
    assert len(frame) == 2
