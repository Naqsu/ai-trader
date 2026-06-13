"""Tests for backtest integration."""

from __future__ import annotations

from pathlib import Path

from bot.core.config import BotConfig, DataConfig
from bot.core.main_bot import PaperFuturesMultiAgentBot


def test_backtest_runs_on_sample_dataset(tmp_path: Path) -> None:
    """Backtest should produce a result dictionary on a small synthetic dataset."""
    rows = ["DateTime\tOpen\tHigh\tLow\tClose\tVolume\tTickVolume"]
    price = 100.0
    for minute in range(300):
        hour = 9 + (minute // 60)
        minute_of_hour = minute % 60
        close = price + 0.08
        high = close + 0.15
        low = price - 0.10
        rows.append(
            f"2025.01.01 {hour:02d}:{minute_of_hour:02d}:00\t{price:.2f}\t{high:.2f}\t{low:.2f}\t{close:.2f}\t0\t50"
        )
        price = close

    path = tmp_path / "sample.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    config = BotConfig(data=DataConfig(dataset_path=str(path), warmup_bars=50))
    bot = PaperFuturesMultiAgentBot(config=config)
    results = bot.run_backtest()

    assert "metrics" in results
    assert "summary" in results
    assert "performance_report" in results
