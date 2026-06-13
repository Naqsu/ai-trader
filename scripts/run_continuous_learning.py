"""Run the continuous learning loop until explicitly stopped."""

from __future__ import annotations

import signal
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.ai.training_pipeline import TrainingPipeline
from bot.core.main_bot import PaperFuturesMultiAgentBot
from bot.utils.file_utils import FileUtils
from bot.utils.window_sampling import sample_random_window


STOP_REQUESTED = False
WINDOW_SIZE = 500
DATASET_REFRESH_EVERY = 5
MODEL_RETRAIN_EVERY = 5


def _handle_stop(signum, frame) -> None:  # noqa: ARG001
    """Mark the loop for graceful shutdown."""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("[loop] stop requested, finishing current cycle...")


def main() -> None:
    """Keep running paper simulation and refresh learned artifacts."""
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    rng = random.Random()

    cycle = 0
    while not STOP_REQUESTED:
        cycle += 1
        print(f"[loop] starting cycle={cycle}")
        bot = PaperFuturesMultiAgentBot()
        frame = bot.prepare_dataset()
        window_frame, metadata = sample_random_window(frame, window_size=WINDOW_SIZE, rng=rng)
        print(
            "[loop] using random window "
            f"rows={metadata.start}:{metadata.end} "
            f"time_range={metadata.start_timestamp}..{metadata.end_timestamp}"
        )

        print("[loop] running paper simulation")
        paper_results = bot.run_paper_trading(
            latest_bars=len(window_frame),
            window_end=len(window_frame),
            frame=window_frame,
        )
        print(f"[loop] paper simulation completed: {paper_results['performance_report']}")

        if STOP_REQUESTED:
            break

        if cycle % DATASET_REFRESH_EVERY == 1:
            print("[loop] refreshing processed dataset snapshot")
            output_path = ROOT / "storage/processed_data/research_dataset.csv"
            frame.to_csv(output_path)
            print(f"[loop] dataset refreshed: rows={len(frame)} path={output_path}")

        if STOP_REQUESTED:
            break

        if cycle % MODEL_RETRAIN_EVERY == 1:
            print("[loop] retraining baseline models from latest learned state")
            feature_columns_regime = ["return_1", "return_5", "trend_bias", "rsi_14", "atr_14", "momentum_10"]
            regime_dataset = frame.dropna(subset=feature_columns_regime + ["market_regime"]).copy()
            pipeline = TrainingPipeline()
            pipeline.train_classifier(
                features=regime_dataset[feature_columns_regime],
                target=regime_dataset["market_regime"],
                artifact_path="storage/models/market_regime_ai.joblib",
                model_name="market_regime_ai",
            )
            print("[loop] market regime model refreshed")

            feature_columns_setup = ["rsi_14", "trend_bias", "atr_14", "momentum_10", "range", "body"]
            setup_dataset = frame.dropna(subset=feature_columns_setup).copy()
            setup_dataset["setup_quality_label"] = (
                (setup_dataset["trend_bias"].abs() + setup_dataset["momentum_10"].abs().fillna(0.0))
                .apply(lambda value: "high" if value >= 2.0 else "medium" if value >= 0.5 else "low")
            )
            pipeline.train_classifier(
                features=setup_dataset[feature_columns_setup],
                target=setup_dataset["setup_quality_label"],
                artifact_path="storage/models/setup_quality_ai.joblib",
                model_name="setup_quality_ai",
            )
            print("[loop] setup quality model refreshed")
        else:
            print("[loop] skipping model retrain this cycle")

        if STOP_REQUESTED:
            break

        print("[loop] cycle completed, sleeping 2 seconds before next cycle")
        time.sleep(2)

    print(f"[loop] stopped after cycle={cycle}")


if __name__ == "__main__":
    main()
