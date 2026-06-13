"""Prepare initial research datasets from raw market data sources."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.core.main_bot import PaperFuturesMultiAgentBot


def main() -> None:
    """Build and persist the enriched research dataset."""
    bot = PaperFuturesMultiAgentBot()
    frame = bot.prepare_dataset()
    output_path = Path("storage/processed_data/research_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)
    print(f"Saved dataset to {output_path} with {len(frame)} rows.")


if __name__ == "__main__":
    main()
