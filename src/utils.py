from __future__ import annotations

import os
import random
import re
import time
from typing import Any, Tuple

import numpy as np
import torch
import yaml


# ═══════════════════════════════════════════════
#  config & path helpers
# ═══════════════════════════════════════════════

def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | None, base_dir: str) -> str | None:
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def safe_name(value: str | None) -> str:
    value = value or "none"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower() or "unknown"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ═══════════════════════════════════════════════
#  seed
# ═══════════════════════════════════════════════

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)


def choose_seed(config_seed: Any, cli_seed: int | None) -> int:
    if cli_seed is not None:
        return cli_seed
    if config_seed is not None:
        return int(config_seed)
    return random.SystemRandom().randint(0, 2**31 - 1)


# ═══════════════════════════════════════════════
#  hardware-aware resolution / steps
# ═══════════════════════════════════════════════

def nearest_multiple(value: int, base: int) -> int:
    return max(base, (value // base) * base)


def get_resolution(
    perf_index: float, min_res: int = 384, max_res: int = 1024, multiple: int = 64,
) -> int:
    perf = max(0.0, min(1.0, perf_index))
    res = int(min_res + (max_res - min_res) * perf)
    return nearest_multiple(res, multiple)


def get_steps(perf_index: float, min_steps: int = 10, max_steps: int = 50) -> int:
    perf = max(0.0, min(1.0, perf_index))
    return max(min_steps, int(min_steps + (max_steps - min_steps) * perf))


def pick_jitter(perf_index: float, low: float, mid: float, high: float) -> float:
    if perf_index < 0.3:
        return low
    if perf_index < 0.7:
        return mid
    return high


# ═══════════════════════════════════════════════
#  benchmark
# ═══════════════════════════════════════════════

def benchmark_device(
    model,
    device: torch.device,
    ref_time: float = 0.08,
    repeats: int = 3,
    steps: int = 100,
) -> Tuple[float, float]:
    """Lightweight matmul benchmark — fast, model-independent, differentiates CPU vs GPU."""
    model.eval()
    size = 2048
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    durations = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            torch.mm(a, b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        durations.append(time.perf_counter() - start)
    avg_time = sum(durations) / len(durations)
    perf_index = min(1.0, max(0.05, ref_time / max(avg_time, 1e-6)))
    return perf_index, avg_time
