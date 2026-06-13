"""Train the SetupQualityAI component on prepared datasets."""

from __future__ import annotations

import pandas as pd
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
    """Train a baseline classifier on proxy setup-quality labels."""
    bot = PaperFuturesMultiAgentBot()
    frame = bot.prepare_dataset()
    feature_columns = ["rsi_14", "trend_bias", "atr_14", "momentum_10", "range", "body"]
    dataset = frame.dropna(subset=feature_columns).copy()
    dataset["setup_quality_label"] = pd.cut(
        dataset["trend_bias"].abs() + dataset["momentum_10"].abs().fillna(0.0),
        bins=[-1.0, 0.5, 2.0, float("inf")],
        labels=["low", "medium", "high"],
    )
    dataset = sample_training_rows(dataset, max_rows=TRAINING_SAMPLE_ROWS, rng=random.Random())
    print(f"[train] sampled {len(dataset)} random rows for setup_quality_ai")
    pipeline = TrainingPipeline()
    artifact = pipeline.train_classifier(
        features=dataset[feature_columns],
        target=dataset["setup_quality_label"],
        artifact_path="storage/models/setup_quality_ai.joblib",
        model_name="setup_quality_ai",
    )
    print(f"Saved setup quality model to {artifact}.")


if __name__ == "__main__":
    main()
