"""Start the conservative paper trading runtime."""

from __future__ import annotations

import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.core.main_bot import PaperFuturesMultiAgentBot
from bot.utils.window_sampling import sample_random_window


WINDOW_SIZE = 500


def main() -> None:
    """Run a random-window paper-trading simulation."""
    bot = PaperFuturesMultiAgentBot()
    frame = bot.prepare_dataset()
    sampled_frame, metadata = sample_random_window(frame, window_size=WINDOW_SIZE, rng=random.Random())
    print(
        "[paper] sampled random window "
        f"rows={metadata.start}:{metadata.end} "
        f"time_range={metadata.start_timestamp}..{metadata.end_timestamp}"
    )
    results = bot.run_paper_trading(
        latest_bars=len(sampled_frame),
        window_end=len(sampled_frame),
        frame=sampled_frame,
    )
    print(results["performance_report"])
    print(results["daily_report"])


if __name__ == "__main__":
    main()
