"""Fetch, validate, and prepare market data for downstream agents."""

from __future__ import annotations

import pandas as pd

from bot.core.config import BotConfig
from bot.data.dataset_builder import DatasetBuilder
from bot.data.feature_builder import FeatureBuilder
from bot.data.nas100_csv_loader import NAS100CSVLoader
from bot.data.timeframe_sync import TimeframeSynchronizer


class DataAgent:
    """Fetch, validate, and prepare market data for downstream agents."""

    def __init__(
        self,
        config: BotConfig,
        loader: NAS100CSVLoader | None = None,
        synchronizer: TimeframeSynchronizer | None = None,
        feature_builder: FeatureBuilder | None = None,
        dataset_builder: DatasetBuilder | None = None,
    ) -> None:
        self.config = config
        self.loader = loader or NAS100CSVLoader()
        self.synchronizer = synchronizer or TimeframeSynchronizer()
        self.feature_builder = feature_builder or FeatureBuilder()
        self.dataset_builder = dataset_builder or DatasetBuilder()

    def load_market_data(self) -> dict[str, pd.DataFrame]:
        """Load and resample market data."""
        base_frame = self.loader.load(
            self.config.data.dataset_path,
            max_rows=self.config.data.max_rows,
        )
        enriched = self.feature_builder.enrich(base_frame)
        frames = self.synchronizer.build_multi_timeframe(enriched, self.config.data.timeframes)
        return frames

    def build_research_frame(self) -> pd.DataFrame:
        """Build a synchronized 1m-centric research frame."""
        frames = self.load_market_data()
        return self.dataset_builder.merge_timeframes(frames)
