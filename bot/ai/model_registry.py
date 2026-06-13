"""Track model versions, metadata, and deployment readiness."""

from __future__ import annotations


class ModelRegistry:
    """Track model versions, metadata, and deployment readiness."""

    def __init__(self) -> None:
        self._models: dict[str, dict[str, str]] = {}

    def register(self, name: str, version: str, artifact_path: str) -> None:
        """Register a saved model artifact."""
        self._models[name] = {"version": version, "artifact_path": artifact_path}

    def get(self, name: str) -> dict[str, str] | None:
        """Return model metadata if available."""
        return self._models.get(name)
