"""Run the backtesting pipeline from the command line."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.core.main_bot import PaperFuturesMultiAgentBot


def main() -> None:
    """Run the conservative backtest."""
    bot = PaperFuturesMultiAgentBot()
    results = bot.run_backtest()
    print(results["performance_report"])
    print(results["daily_report"])


if __name__ == "__main__":
    main()
