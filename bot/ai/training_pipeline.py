"""Coordinate offline feature generation, training, and artifact storage."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from bot.ai.model_registry import ModelRegistry


class SimpleMajorityClassifier:
    """Minimal fallback classifier when sklearn is unavailable."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "SimpleMajorityClassifier":
        """Store the majority class."""
        _ = features
        self.majority_class = target.mode().iloc[0]
        return self

    def predict(self, features: pd.DataFrame) -> list[str]:
        """Predict the majority class for each row."""
        return [self.majority_class] * len(features)


class TrainingPipeline:
    """Coordinate offline feature generation, training, and artifact storage."""

    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self.model_registry = model_registry or ModelRegistry()

    def train_classifier(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        artifact_path: str,
        model_name: str,
    ) -> str:
        """Train and persist a small baseline classifier."""
        model = self._build_model()
        model.fit(features, target)
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
        self._save_model(model, artifact_path)
        self.model_registry.register(model_name, version="0.1.0", artifact_path=artifact_path)
        return artifact_path

    def _build_model(self):
        """Create the preferred model with a standard-library fallback."""
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            return SimpleMajorityClassifier()
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )

    def _save_model(self, model, artifact_path: str) -> None:
        """Persist the model via joblib when available, else pickle."""
        try:
            import joblib
        except ImportError:
            with open(artifact_path, "wb") as handle:
                pickle.dump(model, handle)
            return
        joblib.dump(model, artifact_path)
