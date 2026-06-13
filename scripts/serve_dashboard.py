"""Serve the interactive control panel and expose task APIs."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import Counter, deque
from datetime import datetime
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PORT = 8765
ROOT = Path(__file__).resolve().parents[1]

TASK_SPECS = {
    "prepare_dataset": {
        "label": "Prepare Dataset",
        "command": ["python3", "scripts/prepare_dataset.py"],
        "kind": "data",
        "resets_runtime_logs": False,
    },
    "run_backtest": {
        "label": "Run Backtest",
        "command": ["python3", "scripts/run_backtest.py"],
        "kind": "runtime",
        "resets_runtime_logs": True,
    },
    "run_paper_trading": {
        "label": "Run Paper Simulation",
        "command": ["python3", "scripts/run_paper_trading.py"],
        "kind": "runtime",
        "resets_runtime_logs": True,
    },
    "run_continuous_learning": {
        "label": "Run Continuous Learning",
        "command": ["python3", "scripts/run_continuous_learning.py"],
        "kind": "runtime",
        "resets_runtime_logs": True,
    },
    "run_continuous_rl_gen_learning": {
        "label": "Run Continuous RL+GEN Learning",
        "command": ["python3", "-u", "scripts/run_continuous_rl_gen_learning.py"],
        "kind": "research",
        "resets_runtime_logs": False,
    },
    "train_market_regime_ai": {
        "label": "Train Market Regime AI",
        "command": ["python3", "scripts/train_market_regime_ai.py"],
        "kind": "training",
        "resets_runtime_logs": False,
    },
    "train_setup_quality_ai": {
        "label": "Train Setup Quality AI",
        "command": ["python3", "scripts/train_setup_quality_ai.py"],
        "kind": "training",
        "resets_runtime_logs": False,
    },
}


@dataclass
class TaskRuntime:
    """Track one active or recent task."""

    task_id: str
    label: str
    command: list[str]
    kind: str
    status: str = "idle"
    started_at: float | None = None
    finished_at: float | None = None
    return_code: int | None = None
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=400))
    process: subprocess.Popen[str] | None = None
    stop_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize public task state."""
        return {
            "task_id": self.task_id,
            "label": self.label,
            "command": self.command,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "log_lines": list(self.log_lines),
        }


class TaskManager:
    """Run one dashboard-triggered task at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_task: TaskRuntime | None = None
        self.last_task: TaskRuntime | None = None

    def start(self, task_id: str) -> TaskRuntime:
        """Start a new task."""
        if task_id not in TASK_SPECS:
            raise KeyError(task_id)

        with self._lock:
            if self.active_task and self.active_task.status == "running":
                raise RuntimeError("Another task is already running.")

            spec = TASK_SPECS[task_id]
            if spec["resets_runtime_logs"]:
                self._reset_runtime_outputs()

            runtime = TaskRuntime(
                task_id=task_id,
                label=spec["label"],
                command=spec["command"],
                kind=spec["kind"],
                status="running",
                started_at=time.time(),
            )
            runtime.log_lines.append(f"$ {' '.join(spec['command'])}")
            process = subprocess.Popen(
                spec["command"],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            runtime.process = process
            self.active_task = runtime

            thread = threading.Thread(target=self._watch_task, args=(runtime,), daemon=True)
            thread.start()
            return runtime

    def stop(self) -> TaskRuntime | None:
        """Terminate the active task if any."""
        with self._lock:
            runtime = self.active_task
            if runtime is None or runtime.process is None or runtime.status != "running":
                return None
            runtime.process.terminate()
            runtime.stop_requested = True
            runtime.log_lines.append("Task termination requested from dashboard.")
            return runtime

    def snapshot(self) -> dict[str, Any]:
        """Return task manager state."""
        with self._lock:
            return {
                "active_task": self.active_task.to_dict() if self.active_task else None,
                "last_task": self.last_task.to_dict() if self.last_task else None,
            }

    def _watch_task(self, runtime: TaskRuntime) -> None:
        """Capture stdout and mark completion."""
        assert runtime.process is not None
        try:
            assert runtime.process.stdout is not None
            for line in runtime.process.stdout:
                runtime.log_lines.append(line.rstrip())
            runtime.process.wait()
            runtime.return_code = runtime.process.returncode
            runtime.finished_at = time.time()
            if runtime.stop_requested:
                runtime.status = "stopped"
            else:
                runtime.status = "completed" if runtime.return_code == 0 else "failed"
            runtime.log_lines.append(f"Task finished with return code {runtime.return_code}.")
        finally:
            with self._lock:
                self.last_task = runtime
                self.active_task = None

    def _reset_runtime_outputs(self) -> None:
        """Clear files that represent the current backtest/paper session."""
        for relative_path in (
            "storage/logs/trades.jsonl",
            "storage/logs/decisions.jsonl",
            "storage/state/patterns.json",
        ):
            path = ROOT / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text(json.dumps({"loss_patterns": {}}, indent=2), encoding="utf-8")
            else:
                path.write_text("", encoding="utf-8")


task_manager = TaskManager()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records if the file exists."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    """Load JSON file with fallback."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def summarize_dashboard() -> dict[str, Any]:
    """Build a UI-ready snapshot from logs and artifacts."""
    trades = read_jsonl(ROOT / "storage/logs/trades.jsonl")
    decisions = read_jsonl(ROOT / "storage/logs/decisions.jsonl")
    patterns = read_json(ROOT / "storage/state/patterns.json", {"loss_patterns": {}})
    rl_gen_status = read_json(ROOT / "storage/state/rl_gen_status.json", {})
    config = read_json(ROOT / "config.example.json", {})
    initial_capital = float(config.get("execution", {}).get("initial_capital", 50_000.0))

    approved = sum(1 for item in decisions if item.get("status") == "approved")
    blocked = sum(1 for item in decisions if item.get("status") == "blocked")
    skipped = sum(1 for item in decisions if item.get("status") == "skipped")

    net_pnl = sum(float(trade.get("net_pnl", 0.0)) for trade in trades)
    gross_profit = sum(float(trade.get("net_pnl", 0.0)) for trade in trades if float(trade.get("net_pnl", 0.0)) > 0)
    gross_loss = abs(
        sum(float(trade.get("net_pnl", 0.0)) for trade in trades if float(trade.get("net_pnl", 0.0)) < 0)
    )
    wins = sum(1 for trade in trades if float(trade.get("net_pnl", 0.0)) > 0)
    equity_curve = [initial_capital]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + float(trade.get("net_pnl", 0.0)))

    peak = equity_curve[0]
    max_drawdown = 0.0
    for point in equity_curve:
        peak = max(peak, point)
        max_drawdown = max(max_drawdown, peak - point)

    strategy_counter: dict[str, dict[str, float | int]] = {}
    regime_counter: dict[str, dict[str, float | int]] = {}
    hour_counter: dict[str, dict[str, float | int]] = {}
    for trade in trades:
        name = str(trade.get("strategy_name", "unknown"))
        bucket = strategy_counter.setdefault(name, {"count": 0, "net_pnl": 0.0})
        bucket["count"] = int(bucket["count"]) + 1
        bucket["net_pnl"] = float(bucket["net_pnl"]) + float(trade.get("net_pnl", 0.0))

        regime = str(trade.get("regime", "unknown"))
        regime_bucket = regime_counter.setdefault(regime, {"count": 0, "net_pnl": 0.0})
        regime_bucket["count"] = int(regime_bucket["count"]) + 1
        regime_bucket["net_pnl"] = float(regime_bucket["net_pnl"]) + float(trade.get("net_pnl", 0.0))

        hour = _extract_hour_label(str(trade.get("timestamp", "")))
        hour_bucket = hour_counter.setdefault(hour, {"count": 0, "net_pnl": 0.0})
        hour_bucket["count"] = int(hour_bucket["count"]) + 1
        hour_bucket["net_pnl"] = float(hour_bucket["net_pnl"]) + float(trade.get("net_pnl", 0.0))

    decision_strategy_counter: dict[str, dict[str, int]] = {}
    for decision in decisions:
        strategy_name = str(decision.get("selected_strategy") or "none")
        bucket = decision_strategy_counter.setdefault(
            strategy_name,
            {"approved": 0, "blocked": 0, "skipped": 0},
        )
        status = str(decision.get("status", "skipped"))
        if status in bucket:
            bucket[status] += 1

    veto_counter = Counter(
        str(decision["veto_reason"])
        for decision in decisions
        if decision.get("veto_reason")
    )

    models = []
    for artifact in ("market_regime_ai.joblib", "setup_quality_ai.joblib"):
        path = ROOT / "storage/models" / artifact
        models.append(
            {
                "name": artifact,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )

    return {
        "config": {
            "environment": config.get("environment", "paper"),
            "primary_market": config.get("primary_market", config.get("data", {}).get("primary_market", "NQ")),
            "max_rows": config.get("data", {}).get("max_rows", 100000),
            "risk": config.get("risk", {}),
        },
        "summary": {
            "trade_count": len(trades),
            "win_rate": wins / len(trades) if trades else 0.0,
            "net_pnl": net_pnl,
            "profit_factor": gross_profit / gross_loss if gross_loss else gross_profit,
            "max_drawdown": max_drawdown,
            "ending_equity": equity_curve[-1],
            "approved_count": approved,
            "blocked_count": blocked,
            "skipped_count": skipped,
            "equity_curve": equity_curve,
        },
        "strategy_breakdown": strategy_counter,
        "regime_breakdown": regime_counter,
        "hour_breakdown": hour_counter,
        "decision_strategy_breakdown": decision_strategy_counter,
        "veto_breakdown": dict(veto_counter),
        "patterns": patterns.get("loss_patterns", {}),
        "trades": trades[-40:],
        "decisions": decisions[-60:],
        "models": models,
        "tasks": task_manager.snapshot(),
        "task_catalog": TASK_SPECS,
        "rl_gen_status": rl_gen_status,
        "server_time": time.time(),
    }


def _extract_hour_label(timestamp: str) -> str:
    """Convert runtime timestamps into an hour bucket label."""
    if not timestamp:
        return "unknown"
    try:
        hour = datetime.fromisoformat(timestamp).hour
    except ValueError:
        hour = int(timestamp.split(" ")[1].split(":")[0])
    return f"{hour:02d}:00"


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve static dashboard assets and lightweight JSON APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        """Serve GET endpoints."""
        parsed = urlparse(self.path)
        if parsed.path == "/api/dashboard":
            self._send_json(summarize_dashboard())
            return
        if parsed.path == "/api/tasks":
            self._send_json(task_manager.snapshot())
            return
        super().do_GET()

    def do_POST(self) -> None:
        """Serve POST endpoints."""
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/tasks/") and parsed.path.endswith("/start"):
            task_id = parsed.path.split("/")[3]
            self._handle_start(task_id)
            return
        if parsed.path == "/api/tasks/stop":
            self._handle_stop()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        """Keep server output concise."""
        print(f"[dashboard] {format % args}")

    def _handle_start(self, task_id: str) -> None:
        """Start a registered task."""
        try:
            runtime = task_manager.start(task_id)
        except KeyError:
            self._send_json({"error": f"Unknown task: {task_id}"}, status=HTTPStatus.NOT_FOUND)
            return
        except RuntimeError as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.CONFLICT)
            return
        self._send_json({"ok": True, "task": runtime.to_dict()})

    def _handle_stop(self) -> None:
        """Stop the current task."""
        runtime = task_manager.stop()
        if runtime is None:
            self._send_json({"ok": False, "error": "No running task."}, status=HTTPStatus.CONFLICT)
            return
        self._send_json({"ok": True, "task": runtime.to_dict()})

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        """Return JSON response."""
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    """Start the dashboard server."""
    os.chdir(ROOT)
    with ThreadingHTTPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Dashboard available at http://127.0.0.1:{PORT}/dashboard/")
        print(f"Serving files from {ROOT}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
