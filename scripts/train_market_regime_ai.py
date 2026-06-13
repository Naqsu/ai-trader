"""Train the MarketRegimeAI component on prepared datasets."""

from __future__ import annotations

import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.ai.training_pipeline import TrainingPipeline
from bot.core.main_bot import PaperFuturesMultiAgentBot
from bot.utils.window_sampling import sample_training_rows


TRAINING_SAMPLE_ROWS = 40_000


def main() -> None:
    """Train a baseline classifier on heuristic regime labels."""
    bot = PaperFuturesMultiAgentBot()
    frame = bot.prepare_dataset()
    feature_columns = ["return_1", "return_5", "trend_bias", "rsi_14", "atr_14", "momentum_10"]
    dataset = frame.dropna(subset=feature_columns + ["market_regime"]).copy()
    dataset = sample_training_rows(dataset, max_rows=TRAINING_SAMPLE_ROWS, rng=random.Random())
    print(f"[train] sampled {len(dataset)} random rows for market_regime_ai")
    pipeline = TrainingPipeline()
    artifact = pipeline.train_classifier(
        features=dataset[feature_columns],
        target=dataset["market_regime"],
        artifact_path="storage/models/market_regime_ai.joblib",
        model_name="market_regime_ai",
    )
    print(f"Saved market regime model to {artifact}.")


if __name__ == "__main__":
    main()
