"""Track order lifecycle and execution acknowledgements."""

from __future__ import annotations

from bot.core.models import ExecutionReport


class OrderManager:
    """Track order lifecycle and execution acknowledgements."""

    def __init__(self) -> None:
        self.completed_orders: list[ExecutionReport] = []

    def record(self, report: ExecutionReport) -> None:
        """Track a completed execution."""
        self.completed_orders.append(report)
