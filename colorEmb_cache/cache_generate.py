"""
Pre-compute final prompt_embeds per brand using PixArt's native encode_prompt().
Run ONCE: python cache_generate.py
Output: colorEmb_cache/brand_prompt_embeds.pt

After this, generate.py can skip T5 entirely and pipe() with cached embeddings.
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from diffusers import PixArtSigmaPipeline

SNAPSHOT_DIR = (
    r"C:\Users\matebook\.cache\huggingface\hub"
    r"\models--PixArt-alpha--PixArt-Sigma-XL-2-1024-MS"
    r"\snapshots\e102b3591cc82e97071b8b4cb90d834d0c487207"
)
PIPELINE_PATH = SNAPSHOT_DIR
OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), "brand_style_embeds.pt")
MAX_SEQ_LEN   = 300

# Each brand → a descriptive prompt. These get encoded ONCE and cached.
BRAND_PROMPTS: dict[str, str] = {
    "intel":     "random scene with partial blue and white tones",
    "amd":       "random scene with partial red and orange tones",
    "nvidia":    "random scene with partial neon green and black tones",
    "apple":     "random scene with partial silver and white tones",
    "qualcomm":  "random scene with partial blue and teal tones",
    "unknown":   "random scene with balanced colors, neutral aesthetic",
}

NEGATIVE_PROMPT = "single color, single object"


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
    print("Done. generate.py will load these directly.")


if __name__ == "__main__":
    main()