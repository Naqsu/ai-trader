"""Utility helpers for file and path management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FileUtils:
    """Utility helpers for file and path management."""

    @staticmethod
    def ensure_parent(path: str | Path) -> Path:
        """Ensure the parent directory exists."""
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    @staticmethod
    def write_json(path: str | Path, payload: dict[str, Any]) -> None:
        """Write JSON payload with a stable format."""
        resolved = FileUtils.ensure_parent(path)
        with resolved.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def read_json(path: str | Path) -> dict[str, Any]:
        """Read JSON payload from disk."""
        resolved = Path(path)
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return {}
