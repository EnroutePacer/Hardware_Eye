"""
Pre-compute final prompt_embeds per brand using PixArt's native encode_prompt().
Run ONCE: python cache_generate.py
Output: colorEmb_cache/brand_prompt_embeds.pt

This will download the whole model on your pc, so don't run it unless neccessary.
After this, generate.py can skip T5 entirely and pipe() with cached embeddings.
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from diffusers import PixArtSigmaPipeline

PIPELINE_PATH = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "brand_style_embeds.pt")
MAX_SEQ_LEN   = 300

# Each brand → a descriptive prompt. These get encoded ONCE and cached.
BRAND_PROMPTS: dict[str, str] = {
    "intel":     "a casual scene with partial blue and white elements",
    "amd":       "a casual scene with partial red and orange elements",
    "nvidia":    "a casual scene with partial neon green and black elements",
    "apple":     "a casual scene with partial silver and white elements",
    "qualcomm":  "a casual scene with partial blue and teal elements",
    "unknown":   "a casual scene with balanced colors, neutral aesthetic",
}

NEGATIVE_PROMPT = "uniform color"
EMPTY_PROMPT    = ""
LANDSCAPE_PROMPT = "landscape"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading PixArtSigmaPipeline from {PIPELINE_PATH} ...")
    pipe = PixArtSigmaPipeline.from_pretrained(PIPELINE_PATH, torch_dtype=dtype).to(device)

    print("Encoding negative prompt (shared)...")
    neg_embeds, neg_mask, _, _ = pipe.encode_prompt(
        prompt=NEGATIVE_PROMPT,
        do_classifier_free_guidance=False,
        negative_prompt="",
        num_images_per_prompt=1,
        device=device,
        clean_caption=False,
        max_sequence_length=MAX_SEQ_LEN,
    )

    cache: dict[str, dict[str, torch.Tensor]] = {
        "_negative_": {
            "prompt_embeds": neg_embeds.cpu(),
            "prompt_mask": neg_mask.cpu(),
        }
    }

    # ── Landscape style direction: landscape - empty ──
    print("Encoding empty prompt (baseline)...")
    empty_embeds, empty_mask, _, _  = pipe.encode_prompt(
        prompt=EMPTY_PROMPT, do_classifier_free_guidance=False, negative_prompt="",
        num_images_per_prompt=1, device=device, clean_caption=False, max_sequence_length=MAX_SEQ_LEN,
    )
    # Save empty prompt as baseline for unconditional/dismiss mode
    cache["_empty_"] = {
        "prompt_embeds": empty_embeds.cpu(),
        "prompt_mask":   empty_mask.cpu(),
    }
    print(f"  empty prompt  embeds={list(empty_embeds.shape)}  norm={empty_embeds.norm():.2f}")

    print(f"Encoding landscape prompt: \"{LANDSCAPE_PROMPT}\"")
    landscape_embeds, landscape_mask, _, _ = pipe.encode_prompt(
        prompt=LANDSCAPE_PROMPT, do_classifier_free_guidance=False, negative_prompt="",
        num_images_per_prompt=1, device=device, clean_caption=False, max_sequence_length=MAX_SEQ_LEN,
    )
    cache["_landscape_direction_"] = {
        "prompt_embeds": landscape_embeds.cpu(),
        "prompt_mask":   landscape_mask.cpu(),
    }
    print(f"  landscape direction  embeds={list(landscape_embeds.shape)}  norm={landscape_embeds.norm():.2f}")

    print(f"Encoding {len(BRAND_PROMPTS)} brand prompts...")
    with torch.no_grad():
        for brand, prompt in BRAND_PROMPTS.items():
            print(f"  [{brand}] \"{prompt}\"")
            pos_embeds, pos_mask, _, _ = pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                negative_prompt="",
                num_images_per_prompt=1,
                device=device,
                clean_caption=False,
                max_sequence_length=MAX_SEQ_LEN,
            )
            cache[brand] = {
                "prompt_embeds": pos_embeds.cpu(),
                "prompt_mask": pos_mask.cpu(),
            }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(cache, OUTPUT_PATH)

    print(f"\nSaved to {OUTPUT_PATH}")
    for key, entry in cache.items():
        e = entry["prompt_embeds"]
        m = entry["prompt_mask"]
        print(f"  {key:14s}  embeds={list(e.shape)}  mask_tokens={int(m.sum())}")
    print("Done. generate.py will load these directly (no T5 needed at runtime).")


if __name__ == "__main__":
    main()