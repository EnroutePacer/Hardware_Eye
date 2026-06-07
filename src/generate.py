from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from io import BytesIO
from typing import Any

# ensure project root is on path (supports both `python src/generate.py` and `python -m src.generate`)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import PixArtSigmaPipeline, Transformer2DModel, AutoencoderKL, DPMSolverMultistepScheduler
from huggingface_hub import snapshot_download
from PIL import Image

from src.color_map import get_hardware_conditions
from src.detect import get_device, get_hardware_profile
from src.utils import (
    benchmark_device, choose_seed, ensure_dir, get_resolution, get_steps,
    load_config, pick_jitter, resolve_path, safe_name, seed_everything,
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root (parent of src/)
DEFAULT_PIPELINE_PATH = os.path.join(BASE_DIR, "models", "pixart_sigma")
CACHE_PATH     = os.path.join(BASE_DIR, "colorEmb_cache", "brand_style_embeds.pt")
FALLBACK_BRAND = "unknown"


def load_prompt_cache() -> dict[str, dict[str, torch.Tensor]]:
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"Cache not found at {CACHE_PATH}. Run: python colorEmb_cache/cache_generate.py")
    return torch.load(CACHE_PATH, map_location="cpu", weights_only=False)


HF_PIPELINE_ID = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
# Sub-folders we actually need (skip text_encoder, tokenizer — we use cached prompt embeds)
HF_SUBFOLDERS = ["transformer", "vae", "scheduler"]


def _ensure_models(pipeline_path: str) -> None:
    """Download transformer/vae/scheduler from HuggingFace if not present locally."""
    if os.path.isdir(os.path.join(pipeline_path, "transformer")):
        return  # already present

    print(f"Models not found at {pipeline_path} — downloading from HuggingFace …")
    os.makedirs(pipeline_path, exist_ok=True)
    for sub in HF_SUBFOLDERS:
        snapshot_download(
            HF_PIPELINE_ID,
            allow_patterns=[f"{sub}/**"],
            local_dir=pipeline_path,
            local_dir_use_symlinks=False,
        )
    print("Download complete.")


# ═══════════════════════════════════════════════════════════
#  pipeline context (initialised once, reused across calls)
# ═══════════════════════════════════════════════════════════

def init_pipeline(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load model + cache + hardware profile. Call once at startup."""
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config = load_config(config_path)

    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipeline_path = resolve_path(
        config["model"].get("pipeline_path", DEFAULT_PIPELINE_PATH), config_dir,
    )

    _ensure_models(pipeline_path)

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

    perf_index, avg_time = benchmark_device(
        pipe.transformer, device,
        ref_time=float(config["benchmark"]["ref_time"]),
        repeats=int(config["benchmark"]["repeats"]),
        steps=int(config["benchmark"]["steps"]),
    )

    prompt_cache = load_prompt_cache()
    hw_profile = get_hardware_profile()

    return {
        "pipe": pipe,
        "device": device,
        "dtype": dtype,
        "perf_index": perf_index,
        "avg_time": avg_time,
        "config": config,
        "config_dir": config_dir,
        "prompt_cache": prompt_cache,
        "hw_profile": hw_profile,
    }


# ═══════════════════════════════════════════════════════════
#  single generation
# ═══════════════════════════════════════════════════════════

def generate_image(
    ctx: dict[str, Any],
    *,
    seed: int | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
    landscape: bool = False,
    empty: bool = False,
    no_negative: bool = False,
    cpu_brand: str | None = None,
    gpu_brand: str | None = None,
    progress_callback = None,  # callable(stage, pct, msg)
) -> dict[str, Any]:
    """Run one generation and return image bytes + metadata."""
    config = ctx["config"]
    pipe = ctx["pipe"]
    device = ctx["device"]
    dtype = ctx["dtype"]
    perf_index = ctx["perf_index"]
    prompt_cache = ctx["prompt_cache"]
    hw_profile = ctx["hw_profile"]

    # seed
    seed = choose_seed(config["generate"].get("seed"), seed)
    seed_everything(seed)

    # guidance_scale
    if guidance_scale is None:
        guidance_scale = random.uniform(0.05, 1.8)

    # resolution / steps / jitter
    res = get_resolution(
        perf_index,
        min_res=int(config["generate"]["min_res"]),
        max_res=int(config["generate"]["max_res"]),
    )
    steps = steps if steps is not None else get_steps(
        perf_index,
        min_steps=int(config["generate"]["min_steps"]),
        max_steps=int(config["generate"]["max_steps"]),
    )
    jitter = pick_jitter(
        perf_index,
        low=float(config["generate"]["jitter_low"]),
        mid=float(config["generate"]["jitter_mid"]),
        high=float(config["generate"]["jitter_high"]),
    )

    conditions = get_hardware_conditions(hw_profile, perf_index, jitter=jitter)
    _cpu_brand = conditions.get("cpu_brand", FALLBACK_BRAND)
    _gpu_brand = conditions.get("gpu_brand", FALLBACK_BRAND)

    if cpu_brand:
        _cpu_brand = cpu_brand
    if gpu_brand:
        _gpu_brand = gpu_brand

    color_rgb = conditions["color_rgb"]
    neg = prompt_cache["_negative_"]

    # prompt selection
    if landscape:
        entry = dict(prompt_cache["_landscape_direction_"])
    elif empty:
        entry = dict(prompt_cache["_empty_"])
    else:
        gpu_valid = _gpu_brand and _gpu_brand != "unknown" and _gpu_brand in prompt_cache
        _cpu_brand = _cpu_brand if _cpu_brand in prompt_cache else FALLBACK_BRAND

        if gpu_valid:
            gpu_entry = prompt_cache[_gpu_brand]
            cpu_entry = prompt_cache[_cpu_brand]
            entry = {
                "prompt_embeds": 0.995 * gpu_entry["prompt_embeds"] + 0.005 * cpu_entry["prompt_embeds"],
                "prompt_mask": (gpu_entry["prompt_mask"] + cpu_entry["prompt_mask"]).clamp(0, 1),
            }
        else:
            cpu_entry = prompt_cache[_cpu_brand]
            landscape_entry = prompt_cache["_landscape_direction_"]
            entry = {
                "prompt_embeds": 0.995 * cpu_entry["prompt_embeds"] + 0.005 * landscape_entry["prompt_embeds"],
                "prompt_mask": (cpu_entry["prompt_mask"] + landscape_entry["prompt_mask"]).clamp(0, 1),
            }

    prompt_embeds   = entry["prompt_embeds"].to(device=device, dtype=dtype)
    prompt_mask     = entry["prompt_mask"].to(device=device)
    negative_embeds = neg["prompt_embeds"].to(device=device, dtype=dtype)
    negative_mask   = neg["prompt_mask"].to(device=device)

    if no_negative:
        empty_entry = prompt_cache["_empty_"]
        negative_embeds = empty_entry["prompt_embeds"].to(device=device, dtype=dtype)
        negative_mask   = empty_entry["prompt_mask"].to(device=device)

    # progress: prep done
    if progress_callback:
        progress_callback("prep", 5, f"cfg={guidance_scale:.1f}  res={res}  steps={steps}")

    # per-step callback (diffusers 0.38 signature: step, timestep, latents)
    _on_step = None
    if progress_callback:
        _step_counter = [0]
        def _on_step(i, t, latents):
            _step_counter[0] += 1
            pct = int(_step_counter[0] / steps * 100)
            progress_callback("diffuse", pct, f"step {_step_counter[0]}/{steps}")


    # generate
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=None, negative_prompt=None,
        prompt_embeds=prompt_embeds, prompt_attention_mask=prompt_mask,
        negative_prompt_embeds=negative_embeds, negative_prompt_attention_mask=negative_mask,
        num_inference_steps=steps, guidance_scale=guidance_scale,
        height=res, width=res, generator=generator, output_type="pil",
        callback=_on_step,
        callback_steps=1,
    )
    image = result.images[0]

    if progress_callback:
        progress_callback("post", 95, "applying hardware tint …")

    # subtle hardware-color tint (jittered CPU+GPU brand colour → image post-process)
    color_strength = 0.08
    tint = Image.new("RGB", image.size, tuple(int(c * 255) for c in color_rgb))
    image = Image.blend(image, tint, color_strength)

    # save to disk + return bytes
    output_dir = resolve_path(config["generate"]["output_dir"], ctx["config_dir"])
    ensure_dir(output_dir)
    ts = datetime.now().strftime("%H-%M")
    filename = (
        f"{ts}_seed-{seed}_steps-{steps}_cfg-{guidance_scale:.1f}"
        f"_perf-{perf_index:.2f}({safe_name(_cpu_brand)}_&_{safe_name(_gpu_brand)}).png"
    )
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)

    buf = BytesIO()
    image.save(buf, format="PNG")

    return {
        "image_bytes": buf.getvalue(),
        "filepath": filepath,
        "cpu_brand": _cpu_brand,
        "gpu_brand": _gpu_brand,
        "color_rgb": color_rgb.tolist(),
        "seed": seed,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "perf_index": perf_index,
        "resolution": res,
        "jitter": jitter,
    }


# ═══════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--landscape", action="store_true", default=False)
    parser.add_argument("--empty", action="store_true", default=False)
    parser.add_argument("--no-negative", action="store_true", default=False)
    parser.add_argument("--cpu-brand", type=str, default=None,
                        choices=["intel", "amd", "apple", "qualcomm", "unknown"])
    parser.add_argument("--gpu-brand", type=str, default=None,
                        choices=["nvidia", "amd", "intel", "apple", "qualcomm", "unknown"])
    args = parser.parse_args()

    print("Initializing pipeline …")
    ctx = init_pipeline(args.config)

    print(f"Generating (perf={ctx['perf_index']:.3f}) …")
    result = generate_image(
        ctx,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        landscape=args.landscape,
        empty=args.empty,
        no_negative=args.no_negative,
        cpu_brand=args.cpu_brand,
        gpu_brand=args.gpu_brand,
    )

    print(f"Hardware: {ctx['hw_profile']}")
    print(f"Brands:  cpu={result['cpu_brand']}  gpu={result['gpu_brand'] or 'none'}  rgb={result['color_rgb']}")
    print(f"Params:  perf={result['perf_index']:.3f}  res={result['resolution']}  "
          f"steps={result['steps']}  seed={result['seed']}  jitter={result['jitter']:.3f}")
    print(f"Saved:   {result['filepath']}")


if __name__ == "__main__":
    main()
