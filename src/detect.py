from __future__ import annotations

import platform

import psutil
import torch
from cpuinfo import get_cpu_info


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cpu_name() -> str:
    info = get_cpu_info()
    name = info.get("brand_raw") or platform.processor() or "Unknown CPU"
    return name.strip()


def get_gpu_name() -> str | None:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None


def get_memory_gb() -> float:
    return psutil.virtual_memory().total / (1024**3)


def get_hardware_profile() -> dict:
    return {
        "cpu_name": get_cpu_name(),
        "gpu_name": get_gpu_name(),
        "memory_gb": round(get_memory_gb(), 2),
    }
