"""Coordinate offline TRL training and artifact storage."""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import device_put, jit, lax, vmap
from jax import random as jrandom

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "previous-trader-ai" / "training-data"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "storage" / "processed_data" / "npy_cache"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "storage" / "models" / "best_trl_model.pkl"

HIDDEN_DIM = 256
CHUNK_LEN = 100


def load_single_file_numpy(csv_path: Path, max_rows: int = 15000):
    import pandas as pd

    df = pd.read_csv(csv_path)
    close_col = next((c for c in ["close", "Close", "CLOSE", "adj close"] if c in df.columns), df.columns[-1])
    close_vals = df[close_col].dropna().values
    close_vals = np.clip(close_vals, 1e-5, None)

    target_len = max_rows + 1
    if len(close_vals) < target_len:
        pad_size = target_len - len(close_vals)
        close_vals = np.concatenate([np.repeat(close_vals[0], pad_size), close_vals])
    else:
        close_vals = close_vals[-target_len:]

    returns = np.log(close_vals[1:] / (close_vals[:-1] + 1e-8))
    vol = np.sqrt(np.convolve(returns**2, np.ones(20) / 20, mode="same"))
    trend = np.convolve(returns, np.ones(10) / 10, mode="same")
    momentum = np.concatenate([np.zeros(5), returns[:-5]])

    close_1 = close_vals[1:]
    running_max = np.maximum.accumulate(close_1)
    running_min = np.minimum.accumulate(close_1)
    recent_range = np.convolve(running_max - running_min, np.ones(25) / 25, mode="same") / (close_1 + 1e-6)

    def norm(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-6)

    features = np.stack(
        [
            norm(returns[:-1]),
            norm(vol[:-1]),
            norm(trend[:-1]),
            norm(momentum[:-1]),
            norm(recent_range[:-1]),
        ],
        axis=-1,
    ).astype(np.float32)
    targets = norm(returns[1:]).reshape(-1, 1).astype(np.float32)
    return features, targets


def load_all_files_to_pool(file_paths: list[Path], cache_dir: Path = DEFAULT_CACHE_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)

    def load_one(path: Path):
        cache_name = path.with_suffix(".npz").name
        cache_path = cache_dir / cache_name
        if cache_path.exists():
            try:
                with np.load(cache_path) as data:
                    return data["features"], data["targets"]
            except Exception:
                pass
        features, targets = load_single_file_numpy(path)
        try:
            np.savez_compressed(cache_path, features=features, targets=targets)
        except Exception:
            pass
        return features, targets

    print(f"Loading {len(file_paths)} CSV files into memory.")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(load_one, file_paths))

    xs_list = [r[0] for r in results]
    ys_list = [r[1] for r in results]
    return np.stack(xs_list), np.stack(ys_list)


def init_trl_params(key, input_dim=5, hidden_dim=HIDDEN_DIM):
    k1, k2, k3, k4 = jrandom.split(key, 4)
    return {
        "W_ih": jrandom.normal(k1, (hidden_dim, input_dim)) * 0.06,
        "W_hh": jrandom.normal(k2, (hidden_dim, hidden_dim)) * 0.06,
        "W_renorm": jrandom.normal(k3, (hidden_dim // 2, hidden_dim)) * 0.06,
        "W_out": jrandom.normal(k4, (1, hidden_dim)) * 0.06,
        "b_out": jnp.zeros((1,)),
        "dissipation_raw": jnp.ones((hidden_dim,)) * 0.8,
        "temperature": jnp.array(0.06),
    }


def get_dissipation(params):
    return jax.nn.softplus(params["dissipation_raw"]) + 1e-5


def trl_cell(params, x, h):
    d = get_dissipation(params)
    t = params["temperature"]
    h_cand = jnp.tanh(jnp.dot(params["W_ih"], x) + jnp.dot(params["W_hh"], h))
    friction = d * h_cand
    h_new = h_cand - friction * 0.1
    ep = jnp.sum(friction * h_cand) * 0.1 / (jnp.abs(t) + 1e-5)
    hs = jnp.tanh(jnp.dot(params["W_renorm"], h_new))
    return h_new, ep, hs


def trl_model(params, xs):
    h0 = jnp.zeros((params["W_hh"].shape[0],))

    def step(h, x):
        h_new, ep, hs = trl_cell(params, x, h)
        pred = jnp.dot(params["W_out"], h_new) + params["b_out"]
        return h_new, {"pred": pred, "ep": ep, "hs": hs}

    _final_h, aux = lax.scan(step, h0, xs)
    return aux["pred"], aux


def trl_chunk_loss_bf16(params, xs, ys):
    xs_bf = xs.astype(jnp.bfloat16)
    p_bf = jax.tree_util.tree_map(lambda param: param.astype(jnp.bfloat16), params)

    xs_t = jnp.transpose(xs_bf, (1, 0, 2))
    batch_size = xs.shape[0]
    h_init = jnp.zeros((batch_size, HIDDEN_DIM), dtype=jnp.bfloat16)

    def step(h, x):
        d = jax.nn.softplus(p_bf["dissipation_raw"]) + jnp.bfloat16(1e-5)
        t = p_bf["temperature"]

        W_ih_T = jnp.transpose(p_bf["W_ih"])
        W_hh_T = jnp.transpose(p_bf["W_hh"])

        h_cand = jnp.tanh(jnp.dot(x, W_ih_T) + jnp.dot(h, W_hh_T))

        friction = d * h_cand
        h_new = h_cand - friction * jnp.bfloat16(0.1)

        ep = jnp.sum(friction * h_cand, axis=-1) * jnp.bfloat16(0.1) / (
            jnp.abs(t) + jnp.bfloat16(1e-5)
        )

        W_renorm_T = jnp.transpose(p_bf["W_renorm"])
        hs = jnp.tanh(jnp.dot(h_new, W_renorm_T))

        W_out_T = jnp.transpose(p_bf["W_out"])
        pred = jnp.dot(h_new, W_out_T) + p_bf["b_out"]

        return h_new, {"pred": pred, "ep": ep, "hs": hs}

    _, aux = lax.scan(step, h_init, xs_t)

    preds = jnp.transpose(aux["pred"], (1, 0, 2)).astype(jnp.float32)
    ys_f = ys.astype(jnp.float32)
    ep = aux["ep"].astype(jnp.float32)
    hs = jnp.transpose(aux["hs"], (1, 0, 2)).astype(jnp.float32)

    return jnp.mean((preds - ys_f) ** 2) + 0.06 * jnp.mean(ep) + 0.006 * jnp.mean(hs**2)


opt = optax.adam(1e-3)


@partial(jit, static_argnums=(5, 6))
def train_step_fused(params, opt_state, key, files_xs, files_ys, batch_size, chunk_len):
    k1, k2, next_key = jax.random.split(key, 3)
    file_indices = jax.random.randint(k1, (batch_size,), 0, files_xs.shape[0])
    start_indices = jax.random.randint(k2, (batch_size,), 0, files_xs.shape[1] - chunk_len)

    def slice_one(file_idx, start):
        slice_x = lax.dynamic_slice(files_xs, (file_idx, start, 0), (1, chunk_len, 5))
        return jnp.squeeze(slice_x, axis=0)

    def slice_one_y(file_idx, start):
        slice_y = lax.dynamic_slice(files_ys, (file_idx, start, 0), (1, chunk_len, 1))
        return jnp.squeeze(slice_y, axis=0)

    batch_xs = vmap(slice_one)(file_indices, start_indices)
    batch_ys = vmap(slice_one_y)(file_indices, start_indices)

    loss, grads = jax.value_and_grad(trl_chunk_loss_bf16)(params, batch_xs, batch_ys)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, next_key


@jit
def val_step(p, x, y):
    return trl_chunk_loss_bf16(p, x, y)


def auto_tune_batch_size(params, opt_state, train_xs, train_ys):
    print()
    print("Running hardware benchmark for batch size auto-tuning.")
    print("Using hardware GEMM and Tensor Cores.")

    test_sizes = [128, 256, 512, 1024, 2048]
    best_bs = 256
    max_throughput = 0
    key = jrandom.PRNGKey(0)

    for bs in test_sizes:
        try:
            p, o, l, k = train_step_fused(params, opt_state, key, train_xs, train_ys, bs, CHUNK_LEN)
            jax.block_until_ready(p)

            steps = 50
            start_t = time.time()
            for _ in range(steps):
                p, o, l, k = train_step_fused(params, opt_state, key, train_xs, train_ys, bs, CHUNK_LEN)
            jax.block_until_ready(p)

            elapsed = time.time() - start_t
            throughput = (steps * bs * CHUNK_LEN) / elapsed
            its = steps / elapsed

            print(f"Batch size {bs:>4} | {its:>5.1f} it/s | Throughput {throughput / 1000:>5.0f}k bars/s")

            if throughput > max_throughput:
                max_throughput = throughput
                best_bs = bs
            elif throughput < max_throughput * 1.05:
                print("Scaling slowed. Peak throughput reached.")
                break

        except Exception:
            print(f"Batch size {bs} failed or exceeded available VRAM.")
            break

    print(f"Selected batch size: {best_bs}")
    print()
    return best_bs


class TrainingPipeline:
    """Run the TRL training loop from the previous monolithic implementation."""

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        model_path: Path | str = DEFAULT_MODEL_PATH,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.model_path = Path(model_path)

    def train_trl(self, resume: bool = True, total_steps: int = 100000):
        key = jrandom.PRNGKey(42)
        params = init_trl_params(key)
        opt_state = opt.init(params)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        if resume and self.model_path.exists():
            with self.model_path.open("rb") as f:
                params = pickle.load(f)
            opt_state = opt.init(params)
            print("Loaded saved model.")

        all_files = [path for path in self.data_dir.iterdir() if path.suffix == ".csv"]
        all_files.sort()

        if not all_files:
            print(f"No CSV files found in: {self.data_dir}")
            return None

        split_idx = int(len(all_files) * 0.9)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:] if split_idx < len(all_files) else train_files

        train_xs, train_ys = load_all_files_to_pool(train_files, self.cache_dir)
        gpu_train_xs = device_put(train_xs)
        gpu_train_ys = device_put(train_ys)

        val_xs_pool, val_ys_pool = load_all_files_to_pool(val_files, self.cache_dir)

        batch_size = auto_tune_batch_size(params, opt_state, gpu_train_xs, gpu_train_ys)

        print("Building static validation set.")
        val_batches_xs = []
        val_batches_ys = []
        np_key = np.random.default_rng(42)
        for i in range(20):
            file_indices = np_key.choice(len(val_files), size=batch_size, replace=True)
            start_indices = np_key.integers(0, 15000 - CHUNK_LEN, size=batch_size)

            batch_xs, batch_ys = [], []
            for f_idx, s_idx in zip(file_indices, start_indices):
                batch_xs.append(val_xs_pool[f_idx, s_idx : s_idx + CHUNK_LEN])
                batch_ys.append(val_ys_pool[f_idx, s_idx : s_idx + CHUNK_LEN])
            val_batches_xs.append(np.stack(batch_xs))
            val_batches_ys.append(np.stack(batch_ys))

        gpu_val_xs = device_put(np.stack(val_batches_xs))
        gpu_val_ys = device_put(np.stack(val_batches_ys))

        best_loss = float("inf")
        best_params = params
        ma_loss = None
        t_start = time.time()
        loop_key = key

        print(f"Starting training. Steps: {total_steps} | Batch: {batch_size} | Hidden dim: {HIDDEN_DIM}")

        for step in range(1, total_steps + 1):
            params, opt_state, loss_val, loop_key = train_step_fused(
                params, opt_state, loop_key, gpu_train_xs, gpu_train_ys, batch_size, CHUNK_LEN
            )

            if step % 100 == 0:
                loss_cpu = float(loss_val)
                if ma_loss is None:
                    ma_loss = loss_cpu
                else:
                    ma_loss = 0.95 * ma_loss + 0.05 * loss_cpu

            if step % 500 == 0:
                val_idx = (step // 500) % 20
                val_loss = float(val_step(params, gpu_val_xs[val_idx], gpu_val_ys[val_idx]))
                elapsed = time.time() - t_start
                rate = step / elapsed

                print(
                    f"  [{step:>7}/{total_steps}] "
                    f"Loss: {ma_loss:.5f} | Val: {val_loss:.5f} | {rate:>5.0f} it/s"
                )

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
                    with self.model_path.open("wb") as f:
                        pickle.dump(best_params, f)

        jax.block_until_ready(params)
        print()
        print(f"Training complete. Best validation loss: {best_loss:.5f}")
        return best_params


def train_trl(
    resume: bool = True,
    total_steps: int = 100000,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    model_path: Path | str = DEFAULT_MODEL_PATH,
):
    return TrainingPipeline(data_dir=data_dir, cache_dir=cache_dir, model_path=model_path).train_trl(
        resume=resume,
        total_steps=total_steps,
    )
