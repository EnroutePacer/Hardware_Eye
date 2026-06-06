"""Scan blend strength to find the breaking point."""
from __future__ import annotations
import torch, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from diffusers import PixArtSigmaPipeline, Transformer2DModel, AutoencoderKL, DPMSolverMultistepScheduler
from detect import get_device

BASE_DIR = os.path.dirname(__file__)
LOCAL_PATH = os.path.join(BASE_DIR, "models", "pixart_sigma")
CACHE_PATH = os.path.join(BASE_DIR, "colorEmb_cache", "brand_style_embeds.pt")
SEED = 42

def main():
    device = get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    cpu_e = cache["intel"]["prompt_embeds"]
    cpu_m = cache["intel"]["prompt_mask"]
    gpu_e = cache["nvidia"]["prompt_embeds"]
    gpu_m = cache["nvidia"]["prompt_mask"]
    neg_e = cache["_negative_"]["prompt_embeds"]
    neg_m = cache["_negative_"]["prompt_mask"]
    empty_e = cache["_empty_"]["prompt_embeds"]
    empty_m = cache["_empty_"]["prompt_mask"]

    pipe = PixArtSigmaPipeline(
        transformer=Transformer2DModel.from_pretrained(os.path.join(LOCAL_PATH, "transformer"), torch_dtype=dtype),
        vae=AutoencoderKL.from_pretrained(os.path.join(LOCAL_PATH, "vae"), torch_dtype=dtype),
        scheduler=DPMSolverMultistepScheduler.from_pretrained(os.path.join(LOCAL_PATH, "scheduler")),
        text_encoder=None, tokenizer=None,
    ).to(device)

    # Baseline: pure cpu
    strengths = [0.0]

    for s in strengths:
        if s == 0.0:
            pe, pm = cpu_e, cpu_m
            label = "pure CPU"
        else:
            pe = (1-s) * cpu_e + s * gpu_e
            pm = (cpu_m + gpu_m).clamp(0, 1)
            label = f"CPU + {s:.2f}*GPU"

        pe = pe.to(device, dtype)
        pm = pm.to(device)
        ne = empty_e.to(device, dtype)  # use empty as negative for clean comparison
        nm = empty_m.to(device)

        print(f"\n[{label}]  norm={pe.norm():.2f}  mask={int(pm.sum())}")
        gen = torch.Generator(device=device).manual_seed(SEED)
        r = pipe(prompt=None, negative_prompt=None,
                 prompt_embeds=pe, prompt_attention_mask=pm,
                 negative_prompt_embeds=ne, negative_prompt_attention_mask=nm,
                 num_inference_steps=12, guidance_scale=4.5, height=512, width=512,
                 generator=gen, output_type="pil")
        fname = f"blend_{str(s).replace('.', '_')}.png"
        out = os.path.join(BASE_DIR, "outputs", f"scan_{fname}")
        r.images[0].save(out)
        print(f"  Saved: {out}")

    print("\nDone — check outputs/scan_blend_*.png")

if __name__ == "__main__":
    main()
