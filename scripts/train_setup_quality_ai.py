"""Train the TRL model on prepared market CSV datasets."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "previous-trader-ai" / "training-data"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "storage" / "models" / "best_trl_model.pkl"
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TRL model.")
    parser.add_argument("--steps", type=int, default=100000, help="Number of training steps.")
    parser.add_argument("--fresh", action="store_true", help="Train from scratch instead of resuming.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with CSV training files.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Output model pickle path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from bot.ai.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline(data_dir=args.data_dir, model_path=args.model_path)
    pipeline.train_trl(resume=not args.fresh, total_steps=args.steps)


if __name__ == "__main__":
    main()
