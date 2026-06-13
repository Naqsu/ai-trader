"""Helpers for efficient random sampling of market-data windows."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class SampledWindow:
    """Metadata for one sampled contiguous market-data window."""

    start: int
    end: int
    start_timestamp: str
    end_timestamp: str


def sample_random_window(
    frame: pd.DataFrame,
    window_size: int,
    rng: random.Random | None = None,
) -> tuple[pd.DataFrame, SampledWindow]:
    """Sample one contiguous random window from a dataframe."""
    if len(frame) <= window_size:
        sampled = frame.copy()
        return sampled, SampledWindow(
            start=0,
            end=len(sampled),
            start_timestamp=str(sampled.index[0]),
            end_timestamp=str(sampled.index[-1]),
        )

    generator = rng or random.Random()
    max_start = len(frame) - window_size
    start = generator.randint(0, max_start)
    end = start + window_size
    sampled = frame.iloc[start:end].copy()
    return sampled, SampledWindow(
        start=start,
        end=end,
        start_timestamp=str(sampled.index[0]),
        end_timestamp=str(sampled.index[-1]),
    )


def sample_random_windows(
    frame: pd.DataFrame,
    window_size: int,
    count: int,
    rng: random.Random | None = None,
) -> list[tuple[pd.DataFrame, SampledWindow]]:
    """Sample several random contiguous windows."""
    generator = rng or random.Random()
    samples: list[tuple[pd.DataFrame, SampledWindow]] = []
    for _ in range(max(1, count)):
        samples.append(sample_random_window(frame, window_size=window_size, rng=generator))
    return samples


def sample_training_rows(
    frame: pd.DataFrame,
    max_rows: int,
    rng: random.Random | None = None,
) -> pd.DataFrame:
    """Sample a random subset of rows without replacement for faster training."""
    if len(frame) <= max_rows:
        return frame.copy()
    generator = rng or random.Random()
    sample_seed = generator.randint(0, 10_000_000)
    return frame.sample(n=max_rows, replace=False, random_state=sample_seed).copy()
