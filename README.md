# 🖥 Hardware Eye

**Your machine, visualised.**  
No text prompt. No labels. Just your hardware with its eye.

Detects your own **CPU and GPU**, maps them to cached T5 prompt embeddings, and **generates a unique image** via the transformer of PixArt-Sigma — entirely on your own device.

---

## Quick start

```bash
git clone https://github.com/EnroutePacer/Hardware_Eye.git
cd Hardware_Eye
pip install -r requirements.txt
python server.py
```

Open `http://127.0.0.1:8000` → click **Generate**.  
First run downloads ~2.6 GB of model weights (one-time).

---

## Usage

```bash
python src/generate.py                     # default: auto-detect hardware brand
python src/generate.py --landscape         # landscape prompt
python src/generate.py --empty             # unconditional baseline
python src/generate.py --cpu-brand amd --gpu-brand nvidia
python src/generate.py --seed 42 --steps 12 --guidance-scale 3.0
```

---

## Architecture

**Denoise-related factor :** brand style, 
```
detect.py                  color_map.py               generate.py
─────────                  ────────────               ───────────
cpu_name → map_to_brand()  brand + jitter → tint →   post-process
gpu_name → map_to_brand()  prompt cache lookup    →   PixArt-Sigma
                                                       ↓
                                                    image
```

| File | Role |
|---|---|
| `server.py` | FastAPI + SSE progress streaming |
| `docs/index.html` | Web frontend |
| `src/generate.py` | Pipeline init, prompt selection, generation, CLI |
| `src/detect.py` | CPU/GPU name detection |
| `src/color_map.py` | Brand → color mapping + hardware-aware tint |
| `src/utils.py` | Config, seed, benchmark, resolution scaling |
| `colorEmb_cache/` | Pre-computed T5 prompt embeddings per brand |
| `models/pixart_sigma/` | Auto-downloaded model weights |

---

## Configuration

`config.yaml`:

```yaml
model:
  pipeline_path: "models/pixart_sigma"

generate:
  output_dir: "outputs"
  min_res: 512
  max_res: 1024
  min_steps: 10
  max_steps: 50
  jitter_low: 0.02
  jitter_mid: 0.05
  jitter_high: 0.10

benchmark:
  ref_time: 0.08
  repeats: 3
  steps: 100
```

---

## How it works

1. **Detect** — `detect.py` reads CPU/GPU brand via `py-cpuinfo` / `torch.cuda`
2. **Map** — `color_map.py` converts brand names to RGB colors + applies jitter
3. **Embed** — pre-computed T5 prompt embeddings loaded from `brand_style_embeds.pt`
4. **Generate** — PixArt-Sigma DiT denoises from random latent
5. **Tint** — hardware-specific color blended onto the output image at 8% strength
