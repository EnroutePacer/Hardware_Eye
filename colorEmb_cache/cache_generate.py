"""
Generate pre-computed T5 color embeddings for each brand.
Run ONCE: python cache_generate.py
Output: colorEmb_cache/brand_color_embeddings.pt
"""

from __future__ import annotations

import os
import torch
from transformers import T5Tokenizer, T5EncoderModel

# ---- config ----
SNAPSHOT_DIR = (
    r"C:\Users\matebook\.cache\huggingface\hub"
    r"\models--PixArt-alpha--PixArt-Sigma-XL-2-1024-MS"
    r"\snapshots\e102b3591cc82e97071b8b4cb90d834d0c487207"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "brand_color_embeddings.pt")

# Each brand gets a color-description prompt that T5 can actually understand
BRAND_COLORS: dict[str, str] = {
    "intel":     "random scene with partail blue and white",
    "amd":       "random scene with partail red and orange",
    "nvidia":    "random scene with partail green and black",
    "apple":     "random scene with partail silver and white",
    "qualcomm":  "random scene with partail blue and teal",
    # fallback for unknown brands — neutral abstract
    "unknown":   "random scene with partail random colors",
}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Loading T5 tokenizer + encoder from {SNAPSHOT_DIR} ...")
    tokenizer = T5Tokenizer.from_pretrained(SNAPSHOT_DIR, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        SNAPSHOT_DIR, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    text_encoder.eval()

    cache: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for brand, prompt in BRAND_COLORS.items():
            print(f"  Encoding [{brand}]: \"{prompt}\"")
            tokens = tokenizer(
                prompt, return_tensors="pt", padding="max_length",
                max_length=128, truncation=True,
            ).to(device)
            emb = text_encoder(**tokens).last_hidden_state  # (1, 128, 4096)
            cache[brand] = emb.cpu()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(cache, OUTPUT_PATH)
    print(f"\nSaved {len(cache)} brand embeddings to {OUTPUT_PATH}")

    for brand, emb in cache.items():
        print(f"  {brand:12s}  shape={list(emb.shape)}  dtype={emb.dtype}")


if __name__ == "__main__":
    main()
