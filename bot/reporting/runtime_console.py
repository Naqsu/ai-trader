"""Runtime console progress reporting."""

from __future__ import annotations

from bot.core.models import ExecutionReport, SetupAssessment, StrategySignal


class RuntimeConsole:
    """Emit human-readable progress logs for CLI and dashboard console."""

    def __init__(self) -> None:
        self._last_block_signature: tuple[str | None, str | None] | None = None
        self._last_block_count = 0

    def log_progress(self, current_bar: int, total_bars: int, equity: float) -> None:
        """Print coarse progress information."""
        print(
            f"[progress] bar={current_bar}/{total_bars} "
            f"equity={equity:.2f}"
        )

    def log_info(self, message: str) -> None:
        """Print an informational runtime event."""
        print(f"[info] {message}")

    def log_signal(self, signal: StrategySignal, assessment: SetupAssessment, row) -> None:
        """Print a detailed signal summary."""
        print(
            "[signal] "
            f"time={signal.timestamp} strategy={signal.strategy_name} side={signal.side} "
            f"regime={signal.regime} close={row['close']:.2f} vwap={row.get('vwap', 0.0):.2f} "
            f"ema_fast={row.get('ema_fast', 0.0):.2f} ema_slow={row.get('ema_slow', 0.0):.2f} "
            f"rsi={row.get('rsi_14', 0.0):.2f} atr={row.get('atr_14', 0.0):.4f} "
            f"momentum={row.get('momentum_10', 0.0):.5f} "
            f"entry={signal.entry_price:.2f} stop={signal.stop_loss:.2f} target={signal.take_profit:.2f} "
            f"rr={signal.risk_reward:.2f} confidence={signal.confidence:.2f} "
            f"setup_score={assessment.score:.2f} setup_label={assessment.label}"
        )

    def log_block(self, timestamp: str, strategy_name: str | None, reason: str | None, notes: list[str]) -> None:
        """Print veto or skip information."""
        signature = (strategy_name, reason)
        if signature == self._last_block_signature:
            self._last_block_count += 1
            if self._last_block_count not in {1, 5, 10} and self._last_block_count % 25 != 0:
                return
        else:
            self._last_block_signature = signature
            self._last_block_count = 1
        print(
            f"[block] time={timestamp} strategy={strategy_name or '-'} "
            f"reason={reason or '-'} notes={','.join(notes) if notes else '-'}"
        )

    def log_trade(self, report: ExecutionReport, equity_after: float) -> None:
        """Print trade completion details."""
        print(
            "[trade] "
            f"time={report.timestamp} strategy={report.strategy_name} side={report.side} "
            f"qty={report.quantity} entry={report.entry_price:.2f} exit={report.exit_price:.2f} "
            f"exit_reason={report.exit_reason} gross={report.gross_pnl:.2f} "
            f"net={report.net_pnl:.2f} commission={report.commission:.2f} "
            f"slippage={report.slippage_cost:.2f} bars_held={report.bars_held} "
            f"equity_after={equity_after:.2f}"
        )

    def log_learning(self, strategy_name: str, regime: str, setup_bias: float, weight: float) -> None:
        """Print online learning state update."""
        print(
            f"[learning] strategy={strategy_name} regime={regime} "
            f"setup_bias={setup_bias:.3f} strategy_weight={weight:.3f}"
        )
