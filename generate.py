from __future__ import annotations

import argparse
import os
import random
import re
from datetime import datetime
from typing import Any

import torch
import yaml
from diffusers import PixArtSigmaPipeline, Transformer2DModel, AutoencoderKL, DPMSolverMultistepScheduler

from color_map import get_hardware_conditions
from detect import get_device, get_hardware_profile
from utils import benchmark_device, ensure_dir, get_resolution, get_steps, pick_jitter, seed_everything

BASE_DIR = os.path.dirname(__file__)
DEFAULT_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "pixart_sigma")
CACHE_PATH     = os.path.join(BASE_DIR, "colorEmb_cache", "brand_style_embeds.pt")
FALLBACK_BRAND = "unknown"


# ── helpers ──────────────────────────────────────────────

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


def choose_seed(config_seed: Any, cli_seed: int | None) -> int:
    if cli_seed is not None:
        return cli_seed
    if config_seed is not None:
        return int(config_seed)
    return random.SystemRandom().randint(0, 2**31 - 1)


def load_prompt_cache() -> dict[str, dict[str, torch.Tensor]]:
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"Cache not found at {CACHE_PATH}. "
            "Run: python colorEmb_cache/cache_generate.py"
        )
    return torch.load(CACHE_PATH, map_location="cpu", weights_only=False)


# ── main ─────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG scale (default: random")
    parser.add_argument("--landscape", action="store_true", default=False,
                        help="use landscape prompt instead of hardware brand")
    parser.add_argument("--empty", action="store_true", default=False,
                        help="use empty prompt (unconditional baseline)")
    parser.add_argument("--no-negative", action="store_true", default=False,
                        help="replace negative prompt with empty embedding")
    parser.add_argument("--cpu-brand", type=str, default=None,
                        choices=["intel", "amd", "apple", "qualcomm", "unknown"],
                        help="override CPU brand")
    parser.add_argument("--gpu-brand", type=str, default=None,
                        choices=["nvidia", "amd", "intel", "apple", "qualcomm", "unknown"],
                        help="override GPU brand")
    args = parser.parse_args()

    # guidance_scale: random (1.5, 5.0) if not explicitly set
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else random.uniform(0.1, 2.5)
    print(f"guidance_scale: {guidance_scale:.2f}" + (" (random)" if args.guidance_scale is None else " (manual)"))

    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    config = load_config(config_path)

    seed = choose_seed(config["generate"].get("seed"), args.seed)
    seed_everything(seed)

    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipeline_path = resolve_path(
        config["model"].get("pipeline_path", DEFAULT_PIPELINE_PATH), config_dir,
    )

    pipe = PixArtSigmaPipeline(
        transformer=Transformer2DModel.from_pretrained(
            os.path.join(pipeline_path, "transformer"), torch_dtype=dtype,
        ),
        vae=AutoencoderKL.from_pretrained(
            os.path.join(pipeline_path, "vae"), torch_dtype=dtype,
        ),
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            os.path.join(pipeline_path, "scheduler"),
        ),
        text_encoder=None,
        tokenizer=None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    # ── benchmark & hardware-aware resolution / steps ──
    perf_index, avg_time = benchmark_device(
        pipe.transformer, device,
        ref_time=float(config["benchmark"]["ref_time"]),
        repeats=int(config["benchmark"]["repeats"]),
        steps=int(config["benchmark"]["steps"]),
    )

    res = get_resolution(
        perf_index,
        min_res=int(config["generate"]["min_res"]),
        max_res=int(config["generate"]["max_res"]),
    )
    steps = args.steps if args.steps is not None else get_steps(
        perf_index,
        min_steps=int(config["generate"]["min_steps"]),
        max_steps=int(config["generate"]["max_steps"]),
    )
    if steps <= 0:
        raise ValueError("--steps must be greater than 0")

    jitter = pick_jitter(
        perf_index,
        low=float(config["generate"]["jitter_low"]),
        mid=float(config["generate"]["jitter_mid"]),
        high=float(config["generate"]["jitter_high"]),
    )

    hw_profile = get_hardware_profile()
    conditions = get_hardware_conditions(hw_profile, perf_index, jitter=jitter)
    cpu_brand = conditions.get("cpu_brand", FALLBACK_BRAND)
    gpu_brand = conditions.get("gpu_brand", FALLBACK_BRAND)

    # CLI override
    if args.cpu_brand:
        cpu_brand = args.cpu_brand
        print(f"CPU brand override: {cpu_brand}")
    if args.gpu_brand:
        gpu_brand = args.gpu_brand
        print(f"GPU brand override: {gpu_brand}")

    color_rgb = conditions["color_rgb"]

    # ── load cache & select prompt embedding ──
    prompt_cache = load_prompt_cache()
    neg = prompt_cache["_negative_"]

    if args.landscape:
        entry = dict(prompt_cache["_landscape_direction_"])
        print("Prompt: landscape")
    elif args.empty:
        entry = dict(prompt_cache["_empty_"])
        print("Prompt: empty (unconditional baseline)")
    else:
        # Default brand mode: GPU preferred with tiny CPU blend for randomness
        gpu_valid = gpu_brand and gpu_brand != "unknown" and gpu_brand in prompt_cache
        cpu_brand = cpu_brand if cpu_brand in prompt_cache else FALLBACK_BRAND

        if gpu_valid:
            gpu_entry = prompt_cache[gpu_brand]
            cpu_entry = prompt_cache[cpu_brand]
            entry = {
                "prompt_embeds": 0.995 * gpu_entry["prompt_embeds"] + 0.005 * cpu_entry["prompt_embeds"],
                "prompt_mask": (gpu_entry["prompt_mask"] + cpu_entry["prompt_mask"]).clamp(0, 1),
            }
            print(f"Prompt: GPU={gpu_brand} + 0.5% CPU={cpu_brand}")
        else:
            cpu_entry = prompt_cache[cpu_brand]
            landscape_entry = prompt_cache["_landscape_direction_"]
            entry = {
                "prompt_embeds": 0.995 * cpu_entry["prompt_embeds"] + 0.005 * landscape_entry["prompt_embeds"],
                "prompt_mask": (cpu_entry["prompt_mask"] + landscape_entry["prompt_mask"]).clamp(0, 1),
            }
            print(f"Prompt: CPU={cpu_brand} + 0.5% landscape")

    prompt_embeds   = entry["prompt_embeds"].to(device=device, dtype=dtype)
    prompt_mask     = entry["prompt_mask"].to(device=device)
    negative_embeds = neg["prompt_embeds"].to(device=device, dtype=dtype)
    negative_mask   = neg["prompt_mask"].to(device=device)

    if args.no_negative:
        empty_entry = prompt_cache["_empty_"]
        negative_embeds = empty_entry["prompt_embeds"].to(device=device, dtype=dtype)
        negative_mask   = empty_entry["prompt_mask"].to(device=device)
        print("Negative: disabled (empty)")

    # ── generate ──
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_mask,
        negative_prompt_embeds=negative_embeds,
        negative_prompt_attention_mask=negative_mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=res,
        width=res,
        generator=generator,
        output_type="pil",
    )
    image = result.images[0]

    # ── save ──
    output_dir = resolve_path(config["generate"]["output_dir"], config_dir)
    ensure_dir(output_dir)
    ts = datetime.now().strftime("%H-%M")
    file_name = (
        f"{ts}_seed-{seed}_steps-{steps}_cfg-{guidance_scale:.1f}"
        f"_perf-{perf_index:.2f}({safe_name(cpu_brand)}_&_{safe_name(gpu_brand)}).png"
    )
    output_path = os.path.join(output_dir, file_name)
    image.save(output_path)

    print(f"Hardware: {hw_profile}")
    print(f"Brands:  cpu={cpu_brand}  gpu={gpu_brand or 'none'}  rgb={color_rgb.tolist()}")
    print(f"Shape:   embeds={tuple(prompt_embeds.shape)}  mask={tuple(prompt_mask.shape)}")
    print(f"Params:  perf={perf_index:.3f}  res={res}x{res}  steps={steps}  seed={seed}  jitter={jitter:.3f}")
    print(f"Saved:   {output_path}")


if __name__ == "__main__":
    main()
