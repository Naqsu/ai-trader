"""Store learned market and mistake patterns for later retrieval."""

from __future__ import annotations

from bot.utils.file_utils import FileUtils


class PatternStore:
    """Store learned market and mistake patterns for later retrieval."""

    def __init__(self) -> None:
        self.patterns: dict[str, dict[str, int]] = {}

    def save(self, name: str, payload: dict[str, int], path: str = "storage/state/patterns.json") -> None:
        """Persist a named pattern bundle."""
        self.patterns[name] = payload
        FileUtils.write_json(path, self.patterns)
