"""Inspect local JAX/ROCm availability for the accelerated RL+GEN path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.accelerated.jax_rlgen import JAX_IMPORT_ERROR, JaxPopulationPrefilter


def main() -> None:
    """Print a concise runtime report."""
    evaluator = JaxPopulationPrefilter()
    print(f"jax_available={evaluator.is_available()}")
    print(f"backend={evaluator.backend_name()}")
    print(f"devices={evaluator.device_summary()}")
    print(f"unsupported_rocm_target={evaluator.unsupported_target()}")
    if JAX_IMPORT_ERROR is not None:
        print(f"import_error={JAX_IMPORT_ERROR}")


if __name__ == "__main__":
    main()
