from __future__ import annotations

import colorsys
import hashlib
import random
from typing import Iterable

import torch

HARDWARE_COLOR_MAP = {
    "intel": (0.45, 0.55, 0.70),
    "amd": (0.85, 0.35, 0.20),
    "nvidia": (0.20, 0.80, 0.25),
    "apple": (0.72, 0.72, 0.78),
    "qualcomm": (0.35, 0.55, 0.90),
}


def _normalize_brand(name: str | None) -> str:
    if not name:
        return "unknown"
    return name.split()[0].strip().lower()


# Keyword-based brand extraction: handles full hardware names like
# "Intel(R) Core i9-13900K" → "intel", "NVIDIA GeForce RTX 4090" → "nvidia"
BRAND_KEYWORDS: dict[str, list[str]] = {
    "intel":     ["intel", "xeon", "pentium", "celeron", "atom"],
    "amd":       ["amd", "ryzen", "epyc", "threadripper", "radeon"],
    "nvidia":    ["nvidia", "geforce", "quadro", "rtx", "gtx", "tesla", "a100", "h100"],
    "apple":     ["apple", "m1", "m2", "m3", "m4"],
    "qualcomm":  ["qualcomm", "snapdragon"],
}


def map_hardware_to_brand(name: str) -> str:
    """Extract standard brand name from full hardware string."""
    name_lower = name.lower()
    for brand, keywords in BRAND_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return brand
    return "unknown"


def _hash_to_color(key: str) -> tuple[float, float, float]:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    hue = int(digest[:6], 16) / 0xFFFFFF
    sat = 0.55 + (int(digest[6:8], 16) / 0xFF) * 0.25
    val = 0.65 + (int(digest[8:10], 16) / 0xFF) * 0.25
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return r, g, b


def get_brand_pool() -> Iterable[str]:
    return list(HARDWARE_COLOR_MAP.keys())


def get_base_color(brand_name: str | None) -> tuple[float, float, float]:
    brand = map_hardware_to_brand(brand_name) if brand_name else "unknown"
    return HARDWARE_COLOR_MAP.get(brand, _hash_to_color(brand))


def mix_colors(
    cpu_color: tuple[float, float, float],
    gpu_color: tuple[float, float, float],
    weight: float,
) -> tuple[float, float, float]:
    w = max(0.0, min(1.0, weight))
    return tuple((1.0 - w) * cpu_color[i] + w * gpu_color[i] for i in range(3))


def apply_jitter(
    color: tuple[float, float, float], jitter: float
) -> tuple[float, float, float]:
    return tuple(
        max(0.0, min(1.0, c + random.uniform(-jitter, jitter))) for c in color
    )


def get_hardware_conditions(
    hw_profile: dict, perf_index: float, jitter: float = 0.08
) -> dict:
    cpu_name = hw_profile.get("cpu_name", "")
    gpu_name = hw_profile.get("gpu_name", "")
    
    # RGB color: still computed for color_loss supervision in training
    cpu_color = get_base_color(cpu_name)
    gpu_color = get_base_color(gpu_name) if gpu_name else cpu_color
    mixed = mix_colors(cpu_color, gpu_color, perf_index)
    final_color = apply_jitter(mixed, jitter)
    
    # Brand names for T5 embedding cache lookup (keyword-based extraction)
    cpu_brand = map_hardware_to_brand(cpu_name)
    gpu_brand = map_hardware_to_brand(gpu_name) if gpu_name else None
    
    # Hardware identity hash (unchanged)
    identity_str = f"{cpu_name} {gpu_name}".strip()
    identity_hash = int(hashlib.md5(identity_str.encode("utf-8")).hexdigest(), 16)
    
    # Style probability (unchanged)
    landscape_prob = (identity_hash % 1000) / 1000.0
    abstract_prob = 1.0 - landscape_prob
    
    return {
        "identity_hash": identity_hash,
        "cpu_brand": cpu_brand,
        "gpu_brand": gpu_brand,
        "color_rgb": torch.tensor(final_color, dtype=torch.float32),
        "style_vector": torch.tensor([landscape_prob, abstract_prob], dtype=torch.float32),
        "perf_index": perf_index
    }
