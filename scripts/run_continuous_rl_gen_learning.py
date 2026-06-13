"""Run a continuous evolutionary RL-style optimization loop over bot parameters."""

from __future__ import annotations

import copy
import json
import os
import random
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.accelerated.jax_rlgen import JaxPopulationPrefilter, NumpyPopulationPrefilter
from bot.core.config import BotConfig
from bot.core.main_bot import PaperFuturesMultiAgentBot
from bot.utils.file_utils import FileUtils
from bot.utils.math_utils import MathUtils
from bot.utils.window_sampling import sample_random_windows


STOP_REQUESTED = False
POPULATION_SIZE = int(os.getenv("RLGEN_POPULATION_SIZE", "1000"))
ELITE_COUNT = 2
WINDOW_SIZE = int(os.getenv("RLGEN_WINDOW_SIZE", "2500"))
WINDOW_STRIDE = 500
WINDOWS_PER_GEN = int(os.getenv("RLGEN_WINDOWS_PER_GEN", "24"))
MIN_TRADES_PER_CANDIDATE = int(os.getenv("RLGEN_MIN_TRADES", "100"))
MAX_WORKERS = int(os.getenv("RLGEN_MAX_WORKERS", str(min(os.cpu_count() or 4, 12))))
EVAL_BATCH_SIZE = int(os.getenv("RLGEN_EVAL_BATCH_SIZE", "50"))
USE_JAX_PREFILTER = os.getenv("RLGEN_USE_JAX", "0") != "0"
JAX_SHORTLIST_SIZE = int(os.getenv("RLGEN_JAX_SHORTLIST_SIZE", "64"))
USE_NUMPY_PREFILTER = os.getenv("RLGEN_USE_NUMPY_PREFILTER", "1") != "0"
NUMPY_SHORTLIST_SIZE = int(os.getenv("RLGEN_NUMPY_SHORTLIST_SIZE", "96"))
STATE_PATH = ROOT / "storage/state/rl_gen_state.json"
BEST_PATH = ROOT / "storage/state/rl_gen_best_config.json"
ACTIVE_OVERRIDE_PATH = ROOT / "storage/state/active_runtime_overrides.json"
STATUS_PATH = ROOT / "storage/state/rl_gen_status.json"


def _handle_stop(signum, frame) -> None:  # noqa: ARG001
    """Mark the RL+GEN loop for graceful shutdown."""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    _log("[rlgen] stop requested, finishing current generation...")


def _log(message: str) -> None:
    """Emit line-buffered console output for dashboard streaming."""
    print(message, flush=True)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file or return an empty payload."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _base_payload() -> dict[str, Any]:
    """Build the runtime payload with any active overrides already applied."""
    config_payload = _load_json(ROOT / "config.example.json")
    override_payload = _load_json(ACTIVE_OVERRIDE_PATH)
    if override_payload:
        config_payload = _deep_merge(config_payload, override_payload)
    return config_payload


def _load_state(total_rows: int) -> dict[str, int]:
    """Load rolling state for the RL+GEN loop."""
    if not STATE_PATH.exists():
        return {"generation": 0, "window_end": WINDOW_SIZE}
    payload = FileUtils.read_json(STATE_PATH)
    return {
        "generation": int(payload.get("generation", 0)),
        "window_end": max(WINDOW_SIZE, min(int(payload.get("window_end", WINDOW_SIZE)), total_rows)),
    }


def _store_state(generation: int, window_end: int) -> None:
    """Persist cursor and generation counters."""
    FileUtils.write_json(
        STATE_PATH,
        {
            "generation": generation,
            "window_end": window_end,
        },
    )


def _write_status(payload: dict[str, Any]) -> None:
    """Persist a dashboard-readable RL+GEN status snapshot."""
    FileUtils.write_json(STATUS_PATH, payload)


def _load_status() -> dict[str, Any]:
    """Load the last persisted RL+GEN status payload."""
    return _load_json(STATUS_PATH)


def _seed_genome(base_payload: dict[str, Any]) -> dict[str, Any]:
    """Build the current starting genome."""
    return {
        "risk": {
            "min_setup_quality": float(base_payload.get("risk", {}).get("min_setup_quality", 0.55)),
            "target_risk_per_trade": float(base_payload.get("risk", {}).get("target_risk_per_trade", 0.005)),
            "max_daily_drawdown_pct": float(base_payload.get("risk", {}).get("max_daily_drawdown_pct", 0.03)),
        },
        "learning": {
            "regime_loss_block_threshold": int(base_payload.get("learning", {}).get("regime_loss_block_threshold", 3)),
            "setup_score_penalty": float(base_payload.get("learning", {}).get("setup_score_penalty", 0.08)),
            "setup_score_reward": float(base_payload.get("learning", {}).get("setup_score_reward", 0.04)),
        },
        "strategies": {
            "default_weights": copy.deepcopy(
                base_payload.get("strategies", {}).get(
                    "default_weights",
                    {"trend_vwap": 1.0, "reversion_rsi": 0.8, "breakout_atr": 0.7},
                )
            )
        },
    }


def _mutate_genome(genome: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Create a conservative mutation around an existing genome."""
    mutated = copy.deepcopy(genome)
    mutated["risk"]["min_setup_quality"] = round(
        MathUtils.clamp(mutated["risk"]["min_setup_quality"] + rng.uniform(-0.03, 0.03), 0.50, 0.68),
        3,
    )
    mutated["risk"]["target_risk_per_trade"] = round(
        MathUtils.clamp(mutated["risk"]["target_risk_per_trade"] + rng.uniform(-0.001, 0.001), 0.003, 0.008),
        4,
    )
    mutated["risk"]["max_daily_drawdown_pct"] = round(
        MathUtils.clamp(mutated["risk"]["max_daily_drawdown_pct"] + rng.uniform(-0.003, 0.003), 0.02, 0.04),
        4,
    )
    mutated["learning"]["regime_loss_block_threshold"] = int(
        MathUtils.clamp(mutated["learning"]["regime_loss_block_threshold"] + rng.choice([-1, 0, 1]), 2, 6)
    )
    mutated["learning"]["setup_score_penalty"] = round(
        MathUtils.clamp(mutated["learning"]["setup_score_penalty"] + rng.uniform(-0.015, 0.015), 0.03, 0.10),
        3,
    )
    mutated["learning"]["setup_score_reward"] = round(
        MathUtils.clamp(mutated["learning"]["setup_score_reward"] + rng.uniform(-0.01, 0.01), 0.01, 0.06),
        3,
    )
    for name, value in mutated["strategies"]["default_weights"].items():
        mutated["strategies"]["default_weights"][name] = round(
            MathUtils.clamp(float(value) + rng.uniform(-0.12, 0.12), 0.4, 1.2),
            3,
        )
    return mutated


def _build_population(base_genome: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    """Build one generation of candidate genomes."""
    population = [copy.deepcopy(base_genome)]
    while len(population) < POPULATION_SIZE:
        population.append(_mutate_genome(base_genome, rng))
    return population


def _window_slices(frame, window_end: int) -> list[Any]:
    """Create several random windows for one generation."""
    _ = window_end
    return [window for window, _meta in sample_random_windows(frame, window_size=WINDOW_SIZE, count=WINDOWS_PER_GEN, rng=random.Random())]


def _evaluate_candidate(args: tuple[int, dict[str, Any], list[Any], dict[str, Any]]) -> dict[str, Any]:
    """Evaluate one genome across several windows without polluting shared runtime state."""
    candidate_id, genome, windows, base_payload = args
    payload = _deep_merge(base_payload, genome)
    payload.setdefault("learning", {})
    payload["learning"]["enabled"] = False
    config = BotConfig.from_dict(payload)

    net_pnls: list[float] = []
    drawdowns: list[float] = []
    trade_counts: list[int] = []
    win_rates: list[float] = []
    total_trades = 0
    windows_used = 0

    bot = PaperFuturesMultiAgentBot(config=config)
    bot.backtest_agent.pattern_store.save = lambda *args, **kwargs: None  # type: ignore[method-assign]
    bot.continuous_learning_ai.config.enabled = False
    bot.orchestrator.learner = None
    bot.strategy_agent.learner = None
    bot.setup_quality_ai.learner = None
    bot.risk_guardian.risk_filter.learner = None
    bot.strategy_weight_ai.export_state = lambda *args, **kwargs: None  # type: ignore[method-assign]
    bot.trade_logger.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    bot.decision_logger.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    bot.runtime_console.log_progress = lambda *args, **kwargs: None  # type: ignore[method-assign]
    bot.runtime_console.log_signal = lambda *args, **kwargs: None  # type: ignore[method-assign]
    bot.runtime_console.log_block = lambda *args, **kwargs: None  # type: ignore[method-assign]
    bot.runtime_console.log_trade = lambda *args, **kwargs: None  # type: ignore[method-assign]
    bot.runtime_console.log_learning = lambda *args, **kwargs: None  # type: ignore[method-assign]
    bot.runtime_console.log_info = lambda *args, **kwargs: None  # type: ignore[method-assign]
    bot.backtest_agent.runtime_console = bot.runtime_console
    bot.orchestrator.runtime_console = bot.runtime_console

    for window in windows:
        results = bot.run_paper_trading(
            latest_bars=len(window),
            window_end=len(window),
            frame=window,
        )
        metrics = results["metrics"]
        net_pnls.append(float(metrics["net_pnl"]))
        drawdowns.append(float(metrics["max_drawdown"]))
        trade_counts.append(int(metrics["trade_count"]))
        win_rates.append(float(metrics["win_rate"]))
        total_trades += int(metrics["trade_count"])
        windows_used += 1
        if total_trades >= MIN_TRADES_PER_CANDIDATE:
            break

    avg_net = sum(net_pnls) / len(net_pnls) if net_pnls else -9999.0
    avg_dd = sum(drawdowns) / len(drawdowns) if drawdowns else 9999.0
    avg_trades = sum(trade_counts) / len(trade_counts) if trade_counts else 0.0
    avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0.0
    trade_penalty = 0.0
    if total_trades < MIN_TRADES_PER_CANDIDATE:
        trade_penalty = (MIN_TRADES_PER_CANDIDATE - total_trades) * 35.0
    fitness = avg_net - (avg_dd * 0.55) + (avg_win_rate * 300.0) - trade_penalty
    return {
        "candidate_id": candidate_id,
        "fitness": round(fitness, 2),
        "avg_net_pnl": round(avg_net, 2),
        "avg_drawdown": round(avg_dd, 2),
        "avg_trade_count": round(avg_trades, 2),
        "avg_win_rate": round(avg_win_rate, 4),
        "total_trades": total_trades,
        "windows_used": windows_used,
        "genome": genome,
    }


def _next_generation(results: list[dict[str, Any]], rng: random.Random) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select elites and produce the next base genome."""
    ranked = sorted(results, key=lambda item: item["fitness"], reverse=True)
    elites = ranked[:ELITE_COUNT]
    champion = copy.deepcopy(elites[0]["genome"])
    if len(elites) > 1:
        for key in champion["strategies"]["default_weights"]:
            champion["strategies"]["default_weights"][key] = round(
                (
                    champion["strategies"]["default_weights"][key]
                    + elites[1]["genome"]["strategies"]["default_weights"][key]
                )
                / 2,
                3,
            )
    return champion, ranked


def main() -> None:
    """Continuously optimize trading parameters with a generation-based loop."""
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    _log("[rlgen] preparing research frame")
    frame_bot = PaperFuturesMultiAgentBot()
    frame = frame_bot.prepare_dataset()
    base_payload = _base_payload()
    best_payload = _load_json(BEST_PATH)
    seed_payload = _deep_merge(base_payload, best_payload) if best_payload else base_payload
    base_genome = _seed_genome(seed_payload)
    jax_prefilter = JaxPopulationPrefilter(
        initial_capital=frame_bot.config.execution.initial_capital,
        point_value=frame_bot.config.execution.point_value,
    )
    jax_enabled = USE_JAX_PREFILTER and jax_prefilter.is_available()
    numpy_prefilter = NumpyPopulationPrefilter(
        initial_capital=frame_bot.config.execution.initial_capital,
        point_value=frame_bot.config.execution.point_value,
    )
    prefilter_backend = "none"
    if jax_enabled:
        _log(
            "[rlgen] jax prefilter enabled "
            f"backend={jax_prefilter.backend_name()} devices={jax_prefilter.device_summary()} "
            f"shortlist={JAX_SHORTLIST_SIZE}"
        )
        prefilter_backend = "jax"
    elif USE_NUMPY_PREFILTER:
        _log(
            "[rlgen] numpy prefilter enabled "
            f"devices={numpy_prefilter.device_summary()} shortlist={NUMPY_SHORTLIST_SIZE}"
        )
        prefilter_backend = "numpy"
    else:
        _log(
            "[rlgen] approximate prefilter disabled "
            f"use_jax={USE_JAX_PREFILTER} use_numpy={USE_NUMPY_PREFILTER}"
        )
    if jax_prefilter.unsupported_target() is not None:
        _log(
            "[rlgen] rocm target blocked "
            f"target={jax_prefilter.unsupported_target()} "
            "because current JAX ROCm plugin does not officially support it"
        )

    state = _load_state(len(frame))
    generation = state["generation"]
    window_end = state["window_end"]
    rng = random.Random(42 + generation)
    jax_error: str | None = None
    _write_status(
        {
            "status": "running",
            "generation": generation,
            "window_end": window_end,
            "best_candidate": None,
            "history": [],
            "jax_error": None,
            "unsupported_rocm_target": jax_prefilter.unsupported_target(),
        }
    )

    while not STOP_REQUESTED:
        generation += 1
        windows = _window_slices(frame, window_end)
        if not windows:
            window_end = WINDOW_SIZE
            windows = _window_slices(frame, window_end)
        _log(
            f"[rlgen] generation={generation} population={POPULATION_SIZE} "
            f"windows={len(windows)} window_end={window_end} "
            f"target_trades_per_candidate={MIN_TRADES_PER_CANDIDATE}"
        )
        _log("[rlgen] window ranges: " + " | ".join(f"{window.index[0]}..{window.index[-1]}" for window in windows))

        population = _build_population(base_genome, rng)
        selected_population = list(enumerate(population, start=1))
        if jax_enabled:
            try:
                approx_results = jax_prefilter.prefilter(population, windows)
                approx_ranked = sorted(approx_results, key=lambda item: item.fitness, reverse=True)
                shortlist = approx_ranked[: min(JAX_SHORTLIST_SIZE, len(approx_ranked))]
                selected_ids = {result.candidate_id for result in shortlist}
                selected_population = [(index, genome) for index, genome in selected_population if index in selected_ids]
                if shortlist:
                    _log(
                        "[rlgen] jax shortlist "
                        f"selected={len(shortlist)}/{len(population)} "
                        f"best_fitness={shortlist[0].fitness:.2f} "
                        f"best_net={shortlist[0].approx_net_pnl:.2f} "
                        f"best_dd={shortlist[0].approx_drawdown:.2f} "
                        f"best_trades={shortlist[0].approx_trade_count:.2f}"
                )
                jax_error = None
            except Exception as exc:
                jax_enabled = False
                jax_error = str(exc)
                _log(f"[rlgen] jax prefilter failed, falling back to cpu exact-only mode: {exc}")
        if not jax_enabled and USE_NUMPY_PREFILTER:
            approx_results = numpy_prefilter.prefilter(population, windows)
            approx_ranked = sorted(approx_results, key=lambda item: item.fitness, reverse=True)
            shortlist = approx_ranked[: min(NUMPY_SHORTLIST_SIZE, len(approx_ranked))]
            selected_ids = {result.candidate_id for result in shortlist}
            selected_population = [(index, genome) for index, genome in selected_population if index in selected_ids]
            if shortlist:
                _log(
                    "[rlgen] numpy shortlist "
                    f"selected={len(shortlist)}/{len(population)} "
                    f"best_fitness={shortlist[0].fitness:.2f} "
                    f"best_net={shortlist[0].approx_net_pnl:.2f} "
                    f"best_dd={shortlist[0].approx_drawdown:.2f} "
                    f"best_trades={shortlist[0].approx_trade_count:.2f}"
                )
            prefilter_backend = "numpy"
        tasks = [(index, genome, windows, base_payload) for index, genome in selected_population]

        results: list[dict[str, Any]] = []
        total_batches = (len(tasks) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE
        executor_label = "process_pool"
        try:
            executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
            with executor:
                _log(
                    f"[rlgen] evaluating exact shortlist with {executor_label} workers={MAX_WORKERS} "
                    f"batch_size={EVAL_BATCH_SIZE} candidates={len(tasks)}"
                )
                for batch_index, batch_start in enumerate(range(0, len(tasks), EVAL_BATCH_SIZE), start=1):
                    batch = tasks[batch_start : batch_start + EVAL_BATCH_SIZE]
                    _log(
                        f"[rlgen] batch {batch_index}/{total_batches} "
                        f"candidates={batch_start + 1}-{batch_start + len(batch)}"
                    )
                    results.extend(executor.map(_evaluate_candidate, batch))
                    if STOP_REQUESTED:
                        break
        except PermissionError as exc:
            executor_label = "thread_pool"
            _log(f"[rlgen] process pool unavailable, falling back to thread pool: {exc}")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                _log(
                    f"[rlgen] evaluating exact shortlist with {executor_label} workers={MAX_WORKERS} "
                    f"batch_size={EVAL_BATCH_SIZE} candidates={len(tasks)}"
                )
                for batch_index, batch_start in enumerate(range(0, len(tasks), EVAL_BATCH_SIZE), start=1):
                    batch = tasks[batch_start : batch_start + EVAL_BATCH_SIZE]
                    _log(
                        f"[rlgen] batch {batch_index}/{total_batches} "
                        f"candidates={batch_start + 1}-{batch_start + len(batch)}"
                    )
                    results.extend(executor.map(_evaluate_candidate, batch))
                    if STOP_REQUESTED:
                        break

        champion, ranked = _next_generation(results, rng)
        winner = ranked[0]
        _log(
            "[rlgen] best candidate "
            f"id={winner['candidate_id']} fitness={winner['fitness']:.2f} "
            f"net={winner['avg_net_pnl']:.2f} dd={winner['avg_drawdown']:.2f} "
            f"avg_trades={winner['avg_trade_count']:.2f} total_trades={winner['total_trades']} "
            f"win_rate={winner['avg_win_rate']:.2%}"
        )

        FileUtils.write_json(BEST_PATH, champion)
        FileUtils.write_json(ACTIVE_OVERRIDE_PATH, champion)
        base_genome = champion
        status_payload = _load_status()
        history = list(status_payload.get("history", []))
        history.append(
            {
                "generation": generation,
                "fitness": winner["fitness"],
                "avg_net_pnl": winner["avg_net_pnl"],
                "avg_drawdown": winner["avg_drawdown"],
                "avg_trade_count": winner["avg_trade_count"],
                "avg_win_rate": winner["avg_win_rate"],
            }
        )
        history = history[-120:]
        _write_status(
            {
                "status": "running",
                "generation": generation,
                "window_end": window_end,
                "best_candidate": {
                    "candidate_id": winner["candidate_id"],
                    "fitness": winner["fitness"],
                    "avg_net_pnl": winner["avg_net_pnl"],
                    "avg_drawdown": winner["avg_drawdown"],
                    "avg_trade_count": winner["avg_trade_count"],
                    "avg_win_rate": winner["avg_win_rate"],
                    "total_trades": winner["total_trades"],
                    "windows_used": winner["windows_used"],
                },
                "active_override": champion,
                "history": history,
                "population_size": POPULATION_SIZE,
                "target_trades_per_candidate": MIN_TRADES_PER_CANDIDATE,
                "jax_prefilter_enabled": jax_enabled,
                "jax_backend": jax_prefilter.backend_name(),
                "jax_shortlist_size": JAX_SHORTLIST_SIZE if jax_enabled else 0,
                "prefilter_backend": prefilter_backend,
                "numpy_shortlist_size": NUMPY_SHORTLIST_SIZE if prefilter_backend == "numpy" else 0,
                "jax_error": jax_error,
                "unsupported_rocm_target": jax_prefilter.unsupported_target(),
            }
        )

        window_end += WINDOW_STRIDE
        if window_end > len(frame):
            window_end = WINDOW_SIZE
        _store_state(generation, window_end)

        if STOP_REQUESTED:
            break

        _log("[rlgen] generation complete, sleeping 2 seconds")
        time.sleep(2)

    _write_status(
        {
            "status": "stopped",
            "generation": generation,
            "window_end": window_end,
            "best_candidate": _load_status().get("best_candidate"),
            "active_override": _load_json(ACTIVE_OVERRIDE_PATH),
            "history": _load_status().get("history", []),
            "population_size": POPULATION_SIZE,
            "target_trades_per_candidate": MIN_TRADES_PER_CANDIDATE,
            "jax_prefilter_enabled": jax_enabled,
            "jax_backend": jax_prefilter.backend_name(),
            "jax_shortlist_size": JAX_SHORTLIST_SIZE if jax_enabled else 0,
            "prefilter_backend": prefilter_backend,
            "numpy_shortlist_size": NUMPY_SHORTLIST_SIZE if prefilter_backend == "numpy" else 0,
            "jax_error": jax_error,
            "unsupported_rocm_target": jax_prefilter.unsupported_target(),
        }
    )
    _log(f"[rlgen] stopped after generation={generation}")


if __name__ == "__main__":
    main()
