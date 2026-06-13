"""Utility helpers for math and sizing calculations."""

from __future__ import annotations


class MathUtils:
    """Utility helpers for math and sizing calculations."""

    @staticmethod
    def clamp(value: float, minimum: float, maximum: float) -> float:
        """Clamp a float into the target range."""
        return max(minimum, min(value, maximum))

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Divide with a fallback for zero denominators."""
        if denominator == 0:
            return default
        return numerator / denominator
