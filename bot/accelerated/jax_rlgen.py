"""JAX-backed population prefilter for RL+GEN candidate evaluation."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_AMDGPU_TARGETS = {
    "gfx900",
    "gfx906",
    "gfx908",
    "gfx90a",
    "gfx942",
    "gfx950",
    "gfx1030",
    "gfx1100",
    "gfx1101",
    "gfx1103",
    "gfx1150",
    "gfx1151",
    "gfx1200",
    "gfx1201",
}


def _detect_rocm_target() -> str | None:
    """Try to discover the active AMDGPU target from rocminfo output."""
    try:
        result = subprocess.run(
            ["/opt/rocm/bin/rocminfo"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    haystack = "\n".join(part for part in [result.stdout, result.stderr] if part)
    match = re.search(r"\bgfx[0-9a-f]+\b", haystack)
    return match.group(0) if match else None


UNSUPPORTED_GFX_TARGET = None
if os.environ.get("RLGEN_ALLOW_UNSUPPORTED_ROCM", "0") != "1":
    detected_target = _detect_rocm_target()
    if detected_target is not None and detected_target not in SUPPORTED_AMDGPU_TARGETS:
        UNSUPPORTED_GFX_TARGET = detected_target
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

if "JAX_PLATFORMS" not in os.environ and not Path("/dev/kfd").exists():
    # This session cannot see ROCm devices. Force CPU so JAX still runs.
    os.environ["JAX_PLATFORMS"] = "cpu"
if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if "JAX_ASYNC_DISPATCH" not in os.environ:
    os.environ["JAX_ASYNC_DISPATCH"] = "false"

try:
    import jax
    import jax.numpy as jnp

    JAX_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on local runtime
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    JAX_IMPORT_ERROR = exc


REGIME_TO_CODE = {
    "trend_up": 0,
    "trend_down": 1,
    "range": 2,
    "high_volatility": 3,
    "low_volatility": 4,
    "chop": 5,
}


@dataclass(slots=True)
class JaxEvalResult:
    """Compact prefilter output for one candidate genome."""

    candidate_id: int
    fitness: float
    approx_net_pnl: float
    approx_drawdown: float
    approx_trade_count: float
    approx_win_rate: float
    genome: dict[str, Any]


class JaxPopulationPrefilter:
    """Vectorized prefilter that ranks many genomes before exact CPU backtests."""

    FEATURE_COLUMNS = (
        "close",
        "high",
        "low",
        "vwap",
        "ema_fast",
        "ema_slow",
        "rsi_14",
        "atr_14",
        "momentum_10",
        "rolling_high_20",
        "rolling_low_20",
    )

    def __init__(
        self,
        *,
        initial_capital: float = 50_000.0,
        point_value: float = 20.0,
        hold_bars: int = 8,
        decision_stride: int = 4,
    ) -> None:
        self.initial_capital = initial_capital
        self.point_value = point_value
        self.hold_bars = hold_bars
        self.decision_stride = decision_stride
        self._compiled = None

    def is_available(self) -> bool:
        """Return true if JAX is importable."""
        return jax is not None and jnp is not None

    def backend_name(self) -> str:
        """Return the detected JAX backend."""
        if UNSUPPORTED_GFX_TARGET is not None:
            return "cpu"
        if not self.is_available():
            return "unavailable"
        try:
            devices = jax.devices()
        except Exception:  # pragma: no cover - runtime dependent
            return "error"
        if not devices:
            return "none"
        return devices[0].platform

    def device_summary(self) -> str:
        """Return a short device summary for logging."""
        if UNSUPPORTED_GFX_TARGET is not None:
            return (
                f"cpu-forced (unsupported amdgpu target {UNSUPPORTED_GFX_TARGET}; "
                "set RLGEN_ALLOW_UNSUPPORTED_ROCM=1 to override)"
            )
        if not self.is_available():
            return f"unavailable ({JAX_IMPORT_ERROR})"
        try:
            devices = jax.devices()
        except Exception as exc:  # pragma: no cover - runtime dependent
            return f"error ({exc})"
        if not devices:
            return "no-devices"
        return ", ".join(f"{device.platform}:{device.device_kind}" for device in devices)

    def unsupported_target(self) -> str | None:
        """Return the blocked ROCm target, if any."""
        return UNSUPPORTED_GFX_TARGET

    def prefilter(
        self,
        population: list[dict[str, Any]],
        windows: list[pd.DataFrame],
    ) -> list[JaxEvalResult]:
        """Score a full population with one batched JAX pass."""
        if not self.is_available():
            raise RuntimeError(f"JAX prefilter is unavailable: {JAX_IMPORT_ERROR}")
        if not windows:
            return []

        feature_tensor, regime_tensor = self._prepare_windows(windows)
        genome_matrix = self._prepare_population(population)
        metrics = self._evaluate(genome_matrix, feature_tensor, regime_tensor)

        results: list[JaxEvalResult] = []
        for index, genome in enumerate(population, start=1):
            results.append(
                JaxEvalResult(
                    candidate_id=index,
                    fitness=float(metrics["fitness"][index - 1]),
                    approx_net_pnl=float(metrics["net"][index - 1]),
                    approx_drawdown=float(metrics["drawdown"][index - 1]),
                    approx_trade_count=float(metrics["trade_count"][index - 1]),
                    approx_win_rate=float(metrics["win_rate"][index - 1]),
                    genome=genome,
                )
            )
        return results

    def _prepare_windows(self, windows: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
        """Convert a list of windows into dense tensors."""
        feature_stack = []
        regime_stack = []
        for window in windows:
            dense = window.loc[:, self.FEATURE_COLUMNS].astype("float32").to_numpy(copy=True)
            dense[:, 7] = np.maximum(dense[:, 7], 1e-3)  # atr_14
            feature_stack.append(dense)
            regime_stack.append(
                window["market_regime"]
                .map(REGIME_TO_CODE)
                .fillna(REGIME_TO_CODE["chop"])
                .astype("int32")
                .to_numpy(copy=True)
            )
        return np.stack(feature_stack, axis=0), np.stack(regime_stack, axis=0)

    def _prepare_population(self, population: list[dict[str, Any]]) -> np.ndarray:
        """Convert nested genomes into a dense candidate matrix."""
        rows: list[list[float]] = []
        for genome in population:
            rows.append(
                [
                    float(genome["risk"]["min_setup_quality"]),
                    float(genome["risk"]["target_risk_per_trade"]),
                    float(genome["risk"]["max_daily_drawdown_pct"]),
                    float(genome["learning"]["regime_loss_block_threshold"]),
                    float(genome["learning"]["setup_score_penalty"]),
                    float(genome["learning"]["setup_score_reward"]),
                    float(genome["strategies"]["default_weights"]["trend_vwap"]),
                    float(genome["strategies"]["default_weights"]["reversion_rsi"]),
                    float(genome["strategies"]["default_weights"]["breakout_atr"]),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    def _evaluate(
        self,
        genome_matrix: np.ndarray,
        feature_tensor: np.ndarray,
        regime_tensor: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run the compiled evaluator and return numpy arrays."""
        if self._compiled is None:
            self._compiled = jax.jit(self._evaluate_impl)
        outputs = self._compiled(
            jnp.asarray(genome_matrix),
            jnp.asarray(feature_tensor),
            jnp.asarray(regime_tensor),
        )
        return {key: np.asarray(value) for key, value in outputs.items()}

    def _evaluate_impl(self, genomes, features, regimes):  # pragma: no cover - exercised indirectly
        """JAX kernel for fast approximate ranking."""
        close = features[:, :, 0]
        high = features[:, :, 1]
        low = features[:, :, 2]
        vwap = features[:, :, 3]
        ema_fast = features[:, :, 4]
        ema_slow = features[:, :, 5]
        rsi = features[:, :, 6]
        atr = jnp.maximum(features[:, :, 7], 1e-3)
        momentum = features[:, :, 8]
        rolling_high = features[:, :, 9]
        rolling_low = features[:, :, 10]

        valid_len = close.shape[1] - self.hold_bars
        idx = jnp.arange(valid_len)
        trade_slots = (idx % self.decision_stride) == 0

        close_now = close[:, :valid_len]
        high_future = jnp.max(
            jnp.stack([high[:, offset : offset + valid_len] for offset in range(1, self.hold_bars + 1)], axis=-1),
            axis=-1,
        )
        low_future = jnp.min(
            jnp.stack([low[:, offset : offset + valid_len] for offset in range(1, self.hold_bars + 1)], axis=-1),
            axis=-1,
        )
        close_future = close[:, self.hold_bars :]
        vwap_now = vwap[:, :valid_len]
        ema_fast_now = ema_fast[:, :valid_len]
        ema_slow_now = ema_slow[:, :valid_len]
        rsi_now = rsi[:, :valid_len]
        atr_now = atr[:, :valid_len]
        momentum_now = momentum[:, :valid_len]
        rolling_high_now = rolling_high[:, :valid_len]
        rolling_low_now = rolling_low[:, :valid_len]
        regimes_now = regimes[:, :valid_len]

        trend_up = (regimes_now == REGIME_TO_CODE["trend_up"]).astype(jnp.float32)
        trend_down = (regimes_now == REGIME_TO_CODE["trend_down"]).astype(jnp.float32)
        ranging = ((regimes_now == REGIME_TO_CODE["range"]) | (regimes_now == REGIME_TO_CODE["chop"])).astype(
            jnp.float32
        )
        high_vol = (regimes_now == REGIME_TO_CODE["high_volatility"]).astype(jnp.float32)
        low_vol = (regimes_now == REGIME_TO_CODE["low_volatility"]).astype(jnp.float32)

        price_vs_vwap = (close_now - vwap_now) / atr_now
        ema_spread = (ema_fast_now - ema_slow_now) / atr_now
        rsi_centered = (rsi_now - 50.0) / 20.0
        momentum_scaled = momentum_now * 4000.0

        trend_long = jnp.clip(
            0.32 * jax.nn.sigmoid(price_vs_vwap)
            + 0.28 * jax.nn.sigmoid(ema_spread)
            + 0.18 * jnp.clip(rsi_centered, 0.0, 1.0)
            + 0.12 * jnp.clip(momentum_scaled, 0.0, 1.0)
            + 0.10 * trend_up,
            0.0,
            1.0,
        )
        trend_short = jnp.clip(
            0.32 * jax.nn.sigmoid(-price_vs_vwap)
            + 0.28 * jax.nn.sigmoid(-ema_spread)
            + 0.18 * jnp.clip(-rsi_centered, 0.0, 1.0)
            + 0.12 * jnp.clip(-momentum_scaled, 0.0, 1.0)
            + 0.10 * trend_down,
            0.0,
            1.0,
        )
        trend_score = jnp.maximum(trend_long, trend_short)
        trend_side = jnp.where(trend_long >= trend_short, 1.0, -1.0)

        reversion_long = jnp.clip(
            0.34 * jnp.clip((35.0 - rsi_now) / 20.0, 0.0, 1.0)
            + 0.24 * jax.nn.sigmoid(-price_vs_vwap)
            + 0.22 * ranging
            + 0.10 * low_vol
            + 0.10 * jnp.clip(-momentum_scaled, 0.0, 1.0),
            0.0,
            1.0,
        )
        reversion_short = jnp.clip(
            0.34 * jnp.clip((rsi_now - 65.0) / 20.0, 0.0, 1.0)
            + 0.24 * jax.nn.sigmoid(price_vs_vwap)
            + 0.22 * ranging
            + 0.10 * low_vol
            + 0.10 * jnp.clip(momentum_scaled, 0.0, 1.0),
            0.0,
            1.0,
        )
        reversion_score = jnp.maximum(reversion_long, reversion_short)
        reversion_side = jnp.where(reversion_long >= reversion_short, 1.0, -1.0)

        breakout_long = jnp.clip(
            0.28 * jnp.clip((close_now - rolling_high_now + atr_now * 0.25) / atr_now, 0.0, 1.0)
            + 0.20 * jax.nn.sigmoid(price_vs_vwap)
            + 0.20 * jax.nn.sigmoid(ema_spread)
            + 0.16 * high_vol
            + 0.16 * jnp.clip(momentum_scaled, 0.0, 1.0),
            0.0,
            1.0,
        )
        breakout_short = jnp.clip(
            0.28 * jnp.clip((rolling_low_now - close_now + atr_now * 0.25) / atr_now, 0.0, 1.0)
            + 0.20 * jax.nn.sigmoid(-price_vs_vwap)
            + 0.20 * jax.nn.sigmoid(-ema_spread)
            + 0.16 * high_vol
            + 0.16 * jnp.clip(-momentum_scaled, 0.0, 1.0),
            0.0,
            1.0,
        )
        breakout_score = jnp.maximum(breakout_long, breakout_short)
        breakout_side = jnp.where(breakout_long >= breakout_short, 1.0, -1.0)

        strategy_scores = jnp.stack([trend_score, reversion_score, breakout_score], axis=0)  # [3, W, T]
        strategy_sides = jnp.stack([trend_side, reversion_side, breakout_side], axis=0)

        weights = genomes[:, 6:9]  # [C, 3]
        min_setup = genomes[:, 0][:, None, None]
        risk_pct = genomes[:, 1][:, None, None]
        dd_limit = genomes[:, 2][:, None, None]
        loss_threshold = genomes[:, 3][:, None, None]
        penalty = genomes[:, 4][:, None, None]
        reward = genomes[:, 5][:, None, None]

        weighted_scores = strategy_scores[None, :, :, :] * weights[:, :, None, None]
        best_strategy = jnp.argmax(weighted_scores, axis=1)
        best_score = jnp.take_along_axis(weighted_scores, best_strategy[:, None, :, :], axis=1)[:, 0, :, :]
        best_side = jnp.take_along_axis(strategy_sides[None, :, :, :].repeat(genomes.shape[0], axis=0), best_strategy[:, None, :, :], axis=1)[:, 0, :, :]

        setup_gate = best_score >= min_setup
        slot_gate = trade_slots[None, None, :]

        future_return = (close_future - close_now) / close_now
        move_in_atr = ((close_future - close_now) / atr_now) * best_side
        move_high = ((high_future - close_now) / atr_now) * best_side
        move_low = ((low_future - close_now) / atr_now) * best_side

        rr_cap = jnp.where(best_strategy == 2, 2.0, jnp.where(best_strategy == 1, 1.6, 1.82))
        effective_move = jnp.where(
            best_side > 0,
            jnp.clip(jnp.minimum(move_in_atr, move_high), -1.0, rr_cap),
            jnp.clip(jnp.minimum(-move_in_atr, -move_low), -1.0, rr_cap),
        )

        adaptive_gate = jnp.where(
            best_strategy == 1,
            best_score + reward * ranging - penalty * high_vol,
            best_score + reward * (trend_up + trend_down) - penalty * ranging,
        )
        trade_mask = setup_gate & (adaptive_gate >= min_setup) & slot_gate

        risk_dollars = self.initial_capital * risk_pct
        gross_pnl = effective_move * risk_dollars
        fee_penalty = (40.0 + 20.0 * risk_pct * 1000.0)
        net_pnl = jnp.where(trade_mask, gross_pnl - fee_penalty, 0.0)
        wins = jnp.where(trade_mask, net_pnl > 0.0, False)

        flat_pnl = net_pnl.reshape(genomes.shape[0], -1)
        equity_curve = self.initial_capital + jnp.cumsum(flat_pnl, axis=1)
        peak_equity = jnp.maximum.accumulate(equity_curve, axis=1)
        drawdown_curve = peak_equity - equity_curve
        max_drawdown = jnp.max(drawdown_curve, axis=1)

        trade_count = jnp.sum(trade_mask, axis=(1, 2))
        win_count = jnp.sum(wins, axis=(1, 2))
        win_rate = jnp.where(trade_count > 0, win_count / trade_count, 0.0)
        net_total = jnp.sum(flat_pnl, axis=1)

        dd_breach = max_drawdown > (self.initial_capital * dd_limit[:, 0, 0])
        loss_cluster_penalty = jnp.maximum(0.0, loss_threshold[:, 0, 0] - win_rate * 10.0) * 55.0
        fitness = net_total - (max_drawdown * 0.55) + (win_rate * 800.0) - loss_cluster_penalty
        fitness = jnp.where(dd_breach, fitness - 1200.0, fitness)

        return {
            "fitness": fitness,
            "net": net_total,
            "drawdown": max_drawdown,
            "trade_count": trade_count.astype(jnp.float32),
            "win_rate": win_rate.astype(jnp.float32),
        }


class NumpyPopulationPrefilter(JaxPopulationPrefilter):
    """Fast CPU prefilter for large populations when JAX is unavailable or disabled."""

    def is_available(self) -> bool:
        """NumPy prefilter is always available."""
        return True

    def backend_name(self) -> str:
        """Return the active backend name."""
        return "numpy"

    def device_summary(self) -> str:
        """Return the active backend summary."""
        return "cpu:numpy"

    def prefilter(
        self,
        population: list[dict[str, Any]],
        windows: list[pd.DataFrame],
    ) -> list[JaxEvalResult]:
        """Score a full population with vectorized NumPy."""
        if not windows:
            return []
        feature_tensor, regime_tensor = self._prepare_windows(windows)
        genome_matrix = self._prepare_population(population)
        metrics = self._evaluate_numpy(genome_matrix, feature_tensor, regime_tensor)
        return [
            JaxEvalResult(
                candidate_id=index,
                fitness=float(metrics["fitness"][index - 1]),
                approx_net_pnl=float(metrics["net"][index - 1]),
                approx_drawdown=float(metrics["drawdown"][index - 1]),
                approx_trade_count=float(metrics["trade_count"][index - 1]),
                approx_win_rate=float(metrics["win_rate"][index - 1]),
                genome=genome,
            )
            for index, genome in enumerate(population, start=1)
        ]

    def _evaluate_numpy(
        self,
        genomes: np.ndarray,
        features: np.ndarray,
        regimes: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Approximate ranking logic implemented in NumPy."""
        close = features[:, :, 0]
        high = features[:, :, 1]
        low = features[:, :, 2]
        vwap = features[:, :, 3]
        ema_fast = features[:, :, 4]
        ema_slow = features[:, :, 5]
        rsi = features[:, :, 6]
        atr = np.maximum(features[:, :, 7], 1e-3)
        momentum = features[:, :, 8]
        rolling_high = features[:, :, 9]
        rolling_low = features[:, :, 10]

        valid_len = close.shape[1] - self.hold_bars
        idx = np.arange(valid_len)
        trade_slots = (idx % self.decision_stride) == 0

        close_now = close[:, :valid_len]
        high_future = np.max(
            np.stack([high[:, offset : offset + valid_len] for offset in range(1, self.hold_bars + 1)], axis=-1),
            axis=-1,
        )
        low_future = np.min(
            np.stack([low[:, offset : offset + valid_len] for offset in range(1, self.hold_bars + 1)], axis=-1),
            axis=-1,
        )
        close_future = close[:, self.hold_bars :]
        vwap_now = vwap[:, :valid_len]
        ema_fast_now = ema_fast[:, :valid_len]
        ema_slow_now = ema_slow[:, :valid_len]
        rsi_now = rsi[:, :valid_len]
        atr_now = atr[:, :valid_len]
        momentum_now = momentum[:, :valid_len]
        rolling_high_now = rolling_high[:, :valid_len]
        rolling_low_now = rolling_low[:, :valid_len]
        regimes_now = regimes[:, :valid_len]

        trend_up = (regimes_now == REGIME_TO_CODE["trend_up"]).astype(np.float32)
        trend_down = (regimes_now == REGIME_TO_CODE["trend_down"]).astype(np.float32)
        ranging = ((regimes_now == REGIME_TO_CODE["range"]) | (regimes_now == REGIME_TO_CODE["chop"])).astype(np.float32)
        high_vol = (regimes_now == REGIME_TO_CODE["high_volatility"]).astype(np.float32)
        low_vol = (regimes_now == REGIME_TO_CODE["low_volatility"]).astype(np.float32)

        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

        price_vs_vwap = (close_now - vwap_now) / atr_now
        ema_spread = (ema_fast_now - ema_slow_now) / atr_now
        rsi_centered = (rsi_now - 50.0) / 20.0
        momentum_scaled = momentum_now * 4000.0

        trend_long = np.clip(
            0.32 * sigmoid(price_vs_vwap)
            + 0.28 * sigmoid(ema_spread)
            + 0.18 * np.clip(rsi_centered, 0.0, 1.0)
            + 0.12 * np.clip(momentum_scaled, 0.0, 1.0)
            + 0.10 * trend_up,
            0.0,
            1.0,
        )
        trend_short = np.clip(
            0.32 * sigmoid(-price_vs_vwap)
            + 0.28 * sigmoid(-ema_spread)
            + 0.18 * np.clip(-rsi_centered, 0.0, 1.0)
            + 0.12 * np.clip(-momentum_scaled, 0.0, 1.0)
            + 0.10 * trend_down,
            0.0,
            1.0,
        )
        trend_score = np.maximum(trend_long, trend_short)
        trend_side = np.where(trend_long >= trend_short, 1.0, -1.0)

        reversion_long = np.clip(
            0.34 * np.clip((35.0 - rsi_now) / 20.0, 0.0, 1.0)
            + 0.24 * sigmoid(-price_vs_vwap)
            + 0.22 * ranging
            + 0.10 * low_vol
            + 0.10 * np.clip(-momentum_scaled, 0.0, 1.0),
            0.0,
            1.0,
        )
        reversion_short = np.clip(
            0.34 * np.clip((rsi_now - 65.0) / 20.0, 0.0, 1.0)
            + 0.24 * sigmoid(price_vs_vwap)
            + 0.22 * ranging
            + 0.10 * low_vol
            + 0.10 * np.clip(momentum_scaled, 0.0, 1.0),
            0.0,
            1.0,
        )
        reversion_score = np.maximum(reversion_long, reversion_short)
        reversion_side = np.where(reversion_long >= reversion_short, 1.0, -1.0)

        breakout_long = np.clip(
            0.28 * np.clip((close_now - rolling_high_now + atr_now * 0.25) / atr_now, 0.0, 1.0)
            + 0.20 * sigmoid(price_vs_vwap)
            + 0.20 * sigmoid(ema_spread)
            + 0.16 * high_vol
            + 0.16 * np.clip(momentum_scaled, 0.0, 1.0),
            0.0,
            1.0,
        )
        breakout_short = np.clip(
            0.28 * np.clip((rolling_low_now - close_now + atr_now * 0.25) / atr_now, 0.0, 1.0)
            + 0.20 * sigmoid(-price_vs_vwap)
            + 0.20 * sigmoid(-ema_spread)
            + 0.16 * high_vol
            + 0.16 * np.clip(-momentum_scaled, 0.0, 1.0),
            0.0,
            1.0,
        )
        breakout_score = np.maximum(breakout_long, breakout_short)
        breakout_side = np.where(breakout_long >= breakout_short, 1.0, -1.0)

        strategy_scores = np.stack([trend_score, reversion_score, breakout_score], axis=0)
        strategy_sides = np.stack([trend_side, reversion_side, breakout_side], axis=0)

        weights = genomes[:, 6:9]
        min_setup = genomes[:, 0][:, None, None]
        risk_pct = genomes[:, 1][:, None, None]
        dd_limit = genomes[:, 2][:, None, None]
        loss_threshold = genomes[:, 3][:, None, None]
        penalty = genomes[:, 4][:, None, None]
        reward = genomes[:, 5][:, None, None]

        weighted_scores = strategy_scores[None, :, :, :] * weights[:, :, None, None]
        best_strategy = np.argmax(weighted_scores, axis=1)
        best_score = np.take_along_axis(weighted_scores, best_strategy[:, None, :, :], axis=1)[:, 0, :, :]
        repeated_sides = np.repeat(strategy_sides[None, :, :, :], genomes.shape[0], axis=0)
        best_side = np.take_along_axis(repeated_sides, best_strategy[:, None, :, :], axis=1)[:, 0, :, :]

        setup_gate = best_score >= min_setup
        slot_gate = trade_slots[None, None, :]

        move_in_atr = ((close_future - close_now) / atr_now) * best_side
        move_high = ((high_future - close_now) / atr_now) * best_side
        move_low = ((low_future - close_now) / atr_now) * best_side

        rr_cap = np.where(best_strategy == 2, 2.0, np.where(best_strategy == 1, 1.6, 1.82))
        effective_move = np.where(
            best_side > 0,
            np.clip(np.minimum(move_in_atr, move_high), -1.0, rr_cap),
            np.clip(np.minimum(-move_in_atr, -move_low), -1.0, rr_cap),
        )

        adaptive_gate = np.where(
            best_strategy == 1,
            best_score + reward * ranging - penalty * high_vol,
            best_score + reward * (trend_up + trend_down) - penalty * ranging,
        )
        trade_mask = setup_gate & (adaptive_gate >= min_setup) & slot_gate

        risk_dollars = self.initial_capital * risk_pct
        gross_pnl = effective_move * risk_dollars
        fee_penalty = (40.0 + 20.0 * risk_pct * 1000.0)
        net_pnl = np.where(trade_mask, gross_pnl - fee_penalty, 0.0)
        wins = np.where(trade_mask, net_pnl > 0.0, False)

        flat_pnl = net_pnl.reshape(genomes.shape[0], -1)
        equity_curve = self.initial_capital + np.cumsum(flat_pnl, axis=1)
        peak_equity = np.maximum.accumulate(equity_curve, axis=1)
        drawdown_curve = peak_equity - equity_curve
        max_drawdown = np.max(drawdown_curve, axis=1)

        trade_count = np.sum(trade_mask, axis=(1, 2)).astype(np.float32)
        win_count = np.sum(wins, axis=(1, 2)).astype(np.float32)
        win_rate = np.divide(
            win_count,
            trade_count,
            out=np.zeros_like(win_count, dtype=np.float32),
            where=trade_count > 0,
        ).astype(np.float32)
        net_total = np.sum(flat_pnl, axis=1)

        dd_breach = max_drawdown > (self.initial_capital * dd_limit[:, 0, 0])
        loss_cluster_penalty = np.maximum(0.0, loss_threshold[:, 0, 0] - win_rate * 10.0) * 55.0
        fitness = net_total - (max_drawdown * 0.55) + (win_rate * 800.0) - loss_cluster_penalty
        fitness = np.where(dd_breach, fitness - 1200.0, fitness)

        return {
            "fitness": fitness.astype(np.float32),
            "net": net_total.astype(np.float32),
            "drawdown": max_drawdown.astype(np.float32),
            "trade_count": trade_count,
            "win_rate": win_rate,
        }
