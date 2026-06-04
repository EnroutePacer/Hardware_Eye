from __future__ import annotations

import os
import random
import time
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def nearest_multiple(value: int, base: int) -> int:
    return max(base, (value // base) * base)


def get_resolution(
    perf_index: float, min_res: int = 384, max_res: int = 1024, multiple: int = 64
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    if hasattr(vae, "config") and hasattr(vae.config, "scaling_factor"):
        latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    return images


def tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]


def benchmark_device(
    model,
    device: torch.device,
    ref_time: float = 0.08,
    repeats: int = 3,
    steps: int = 100,
    latent_size: int = 128,
    cond_dim: int = 1152,
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
        start = time.perf_counter()
        for _ in range(steps):
            torch.mm(a, b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    avg_time = sum(durations) / len(durations)
    perf_index = min(1.0, max(0.05, ref_time / max(avg_time, 1e-6)))
    return perf_index, avg_time
