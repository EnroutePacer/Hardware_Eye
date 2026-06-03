# Hardware Eye 2.0

Hardware Eye generates an abstract image that reflects the current device — **no text prompt, no class label, no T5 encoder**. The machine's hardware identity (CPU + GPU brand) is converted into a 20-token condition sequence and injected into a frozen **PixArt-Sigma** transformer via its built-in cross-attention and FiLM modulation. Resolution and denoising steps scale smoothly with a lightweight matrix-multiplication benchmark.

---

## Quick start

```bash
pip install -r requirements.txt
python generate.py --config config.yaml
```

(Optional) Fine-tune the adapter layers:
```bash
python train.py --config config.yaml
```

## Backbone model

**PixArt-alpha/PixArt-Sigma-XL-2-1024-MS** — a 28-layer Diffusion Transformer pre-trained on aesthetic images with T5 text conditioning. We discard the T5 text encoder entirely and replace its `encoder_hidden_states` with our own hardware-derived tokens.

Local copy (T5 stripped): `models/pixart_sigma/` (~2.8 GB, transformer + VAE only).

---

## Architecture deep dive

### Complete data flow

```
detect.py: get_device() + get_hardware_profile()
    |
    v  {"cpu_name": "Intel...", "gpu_name": "NVIDIA..."}
    |
color_map.py: get_hardware_conditions()
    |
    v  {"identity_hash": 8172..., "brand_name": "nvidia",
    |    "color_rgb": tensor([0.2,0.8,0.25]),   ← still used for color_loss
    |    "style_vector": tensor([0.73,0.27]), "perf_index": 0.77}
    |
    |                         colorEmb_cache/brand_color_embeddings.pt
    |                         ┌─────────────────────────────────────┐
    |  brand_name="nvidia" →  │ T5("green and black, neon, gaming") │
    |                         │ → (1, 128, 4096) pre-computed       │
    |                         └─────────────────────────────────────┘
    |
cond_encoder.py: HardwareConditionEncoder.forward()
    |
    |  8  identity tokens  <-  embed_identity(hash % 10000)         [FROZEN in train]
    |  4  color tokens     <-  color_emb_proj(T5_emb_mean)          [TRAINABLE]
    |  4  style tokens     <-  fc_style([landscape_prob, abstr])    [TRAINABLE]
    |  4  global tokens    <-  fc_global(concat of above 16)        [TRAINABLE]
    |
    v  cond: tensor (batch=1, seq=20, dim=1152)
    |
model.py: HardwareAwareDiT.forward_with_cond()
    |
    |  Step 1 - FiLM modulation:                                   [FROZEN in train]
    |    gamma * latents + beta   (brand color → global tone)
    |
    |  Step 2 - PixArt cross-attention:
    |    encoder_hidden_states = cond (20 tokens)
    |    → all 28 layers attend to hardware identity + style + color
    |
    v  noise_pred → DDIM step → VAE decode → PNG
```
    |        encoder_hidden_states=cond     <-- hardware tokens injected HERE
    |    ).sample
    |    # Inside PixArt: all 28 layers attend to the 20 hardware tokens
    |
    |  Step 3 - channel split:                                   [line 74]
    |    noise_pred, _ = raw_out.chunk(2, dim=1)  # 8->4  (discard variance)
    |
    v  noise_pred: (1, 4, H, W)  ->  DDIM scheduler step
    |
VAE decode -> PNG
```

### What model.py does (line-by-line)

**`load_pretrained_transformer()` (line 12):**
Loads PixArt-Sigma's `Transformer2DModel` from local path. No `subfolder` needed — `config.yaml` points directly at `transformer/`.

**`HardwareAwareDiT.__init__()` (line 16-42):**
- **Line 27:** Reads `cross_attention_dim=1152` dynamically from `transformer.config`. This is the dimension PixArt's internal cross-attention layers expect for `encoder_hidden_states`. Our `cond_encoder` is built to output exactly this dimension.
- **Line 29:** Creates `HardwareConditionEncoder(hidden_dim=1152)` — matches PixArt.
- **Line 34-35:** Creates `film_gamma` and `film_beta`, each a `Linear(1152 -> 4)` that maps the global condition vector to per-channel scale and shift.
- **Line 37:** Freezes all 28 layers of the PixArt transformer via `freeze_backbone()`. Only our adapter layers (cond_encoder + film_gamma/beta) are trainable.

**`forward_with_cond()` (line 48-75):**
This is the core of the project. Every DDIM denoising step calls this function.

| Step | Code (line) | What happens |
|------|-------------|--------------|
| FiLM | 63-66 | Global condition -> per-channel gamma, beta -> `gamma*latent + beta`. Sets the **overall color tone** of the latent (e.g. NVIDIA -> greenish gamma, Intel -> bluish gamma). |
| PixArt | 69-71 | Passes `encoder_hidden_states=cond`. PixArt's 28 transformer blocks each contain a cross-attention layer that attends to these 20 tokens. **This is where hardware identity shapes structure and texture.** |
| Chunk | 74 | PixArt outputs 8 channels (4 mean + 4 learned variance). DDIM only needs the mean, so we discard the variance half. |

**`trainable_state_dict()` (line 77-82):**
Only three components are saved — `cond_encoder`, `film_gamma`, `film_beta`. The 2.4 GB PixArt transformer is always loaded fresh from disk and kept frozen.

### What generate.py does (line-by-line)

| Line(s) | Code | Purpose |
|---------|------|---------|
| 55-56 | `load_pretrained_transformer(...)` -> `HardwareAwareDiT(transformer)` | Load frozen PixArt + attach trainable adapters |
| 61 | `AutoencoderKL.from_pretrained(vae_path)` | Load SDXL VAE for latent<->pixel conversion |
| 63-72 | `benchmark_device(model, device, ...)` | Run 100x `torch.mm(2048x2048)` -> `perf_index` (GPU ~0.8, CPU ~0.2) |
| 74-78 | `get_resolution(perf_index, 512, 1024)` | Linear interpolation: perf=0.5 -> res=768 (9 smooth steps via `nearest_multiple(res, 64)`) |
| 80-84 | `get_steps(perf_index, 10, 50)` | Same interpolation for DDIM denoising steps |
| 90-92 | `get_hardware_conditions(hw_profile, perf_index)` -> `cond_encoder(...)` | Hardware identity -> 20 tokens -> cond tensor `(1, 20, 1152)` |
| 96-106 | DDIM loop | Standard diffusion reverse process, calling `forward_with_cond()` at each step |

### Parameters dynamically adapted to PixArt-Sigma

| Parameter | Source | Value | Why dynamic |
|-----------|--------|-------|-------------|
| `cond_dim` | `transformer.config.cross_attention_dim` | **1152** | Our 20 tokens must match this dimension for PixArt's built-in cross-attention |
| `in_channels` | `transformer.config.in_channels` | **4** | VAE latent channel count; FiLM gamma/beta output this many channels |
| `out_channels` | `transformer.config.out_channels` | **8** | We chunk out channels [4:8] because PixArt outputs learned variance |
| `sample_size` | `transformer.config.sample_size` | **128** | Max latent resolution = 1024/8; affects max image resolution |
| VAE | separate subfolder | SDXL VAE | Different scaling factor from SD VAE; `decode_latents` handles this generically |

---

## Training strategy: style-only

Training focuses exclusively on teaching the model to recognize **landscape vs abstract** style, bound to each device's identity hash.

### What is trained vs frozen

| Component | Status | Reason |
|-----------|--------|--------|
| `embed_identity` (hash → 8 tokens) | **Trained** | Different devices → different visual identity |
| `fc_style` (style_vector → 4 tokens) | **Trained** | Core target: landscape vs abstract |
| `fc_global` (cross-modal fusion) | **Trained** | Integrates identity + style signals |
| `color_emb_proj` (T5 emb → 4 tokens) | **Frozen** | Pre-computed T5 color embeddings are already semantically meaningful |
| `film_gamma`, `film_beta` | **Frozen** | Color tone modulation uses pre-computed T5 semantics |
| PixArt transformer (28 layers) | **Frozen** | Always |

### Training data

Two folders:
```
data/
├── landscape/    ← 5000+ landscape/nature images
└── abstract/     ← 5000+ abstract/geometric images
```

Each training step:
1. Generate random hardware conditions → `style_vector`
2. If `style_vector[0] > 0.5`: fetch image from `landscape/`, else from `abstract/`
3. Diffusion MSE loss teaches: "these identity+style tokens → this kind of image structure"
4. `color_loss` (L1 between generated-mean-RGB and target-RGB) still runs but gradients only affect identity/style tokens through the cross-attention path

### Run training

```bash
# Step 1: generate color embedding cache (run once)
cd colorEmb_cache
python cache_generate.py

# Step 2: train style recognition
cd ..
python train.py --config config.yaml
```

---

## Current limitations

1. **color_emb_proj is frozen**: The T5 color embeddings ("green and black, neon, gaming") carry rich semantics but the projection layer is not fine-tuned. Colors will influence generation but may not perfectly match brand expectations without training this layer.
2. **Style learning is indirect**: No explicit "landscape-ness" loss — the model learns style only through the pairing of tokens with folder-matched images. Needs sufficient data (5k+ per folder).
3. **Identity via hash only**: "Intel Core i9-13900K" and "Intel Core i9-13900KS" map to unrelated hash buckets. No semantic understanding of hardware tiers or generations.

## Training stages (future)

| Stage | What to train | Goal |
|-------|--------------|------|
| 1 (current) | embed_identity, fc_style, fc_global | Style recognition |
| 2 | + color_emb_proj | Fine-tune color projection |
| 3 | + film_gamma, film_beta | Brand color modulation |
| 4 | + TinyCLIP replacing hash | Semantic hardware understanding |

### Mechanism 1: FiLM (Feature-wise Linear Modulation)

```
cond (1, 20, 1152)
    |
    v mean(dim=1)
cond_global (1, 1152)
    |
    |-- film_gamma: Linear(1152->4) -> gamma (1, 4, 1, 1)
    |-- film_beta:  Linear(1152->4) -> beta  (1, 4, 1, 1)
            |
            v
latent <- gamma * latent + beta
```

This is a **global, channel-wise modulation** applied BEFORE the latent enters PixArt. It is equivalent to saying "make this entire image warmer/cooler/greener/bluer" — perfect for encoding brand color preferences (NVIDIA -> green gamma, AMD -> red gamma, Intel -> blue gamma).

FiLM is the standard approach used in StyleGAN, Muse, and Imagen for this type of conditioning. It is implemented in `model.py` lines 63-66.

### Mechanism 2: encoder_hidden_states hijacking

PixArt-Sigma was designed to take T5 text embeddings as `encoder_hidden_states`. We **replace** those text embeddings with our 20 hardware-derived tokens:

```
                    PixArt Transformer (28 layers, frozen)
                    +------------------------------------------+
  modulated latent  |  Block 0: self-attn -> cross-attn       |
  ----------------->|           Q = image patches              |
                    |           K,V = hardware tokens (20)     |  <- built into PixArt
                    |  Block 1: self-attn -> cross-attn       |
                    |           ...each layer re-attends...    |
                    |  Block 27: self-attn -> cross-attn      |
                    +------------------------------------------+
                                    |
                                    v
                              noise prediction
```

Inside every one of PixArt's 28 transformer blocks, the cross-attention layer computes:

```
Attention(Q_image, K_hardware, V_hardware) = softmax(Q*K^T / sqrt(d)) * V
```

- **Q (Query)** = every spatial position in the latent image
- **K, V (Key, Value)** = our 20 hardware tokens (8 identity + 4 color + 4 style + 4 global)

This means **every pixel region in the image independently decides which hardware tokens to attend to**. A region that "wants" color information attends more to the 4 color tokens; a region that "wants" structural identity attends to the 8 identity tokens.

This is implemented in `model.py` line 69-71 by passing `encoder_hidden_states=cond` to `self.transformer()`.

### Why two mechanisms?

FiLM and encoder_hidden_states serve different purposes:

| | FiLM | encoder_hidden_states (cross-attn) |
|---|---|---|
| **Scope** | Global (whole image) | Local (per spatial position) |
| **What it controls** | Overall color tone, brightness, contrast | Texture, structure, spatial layout |
| **How** | Channel-wise scale & shift: y=g*x+b | Dot-product attention between image patches and hardware tokens |
| **Trainable params** | 2 Linear layers (1152->4 each) | None (PixArt's cross-attn weights are frozen) |

---

## Project layout

```
hardware_eye/
|-- detect.py          # CPU/GPU detection
|-- color_map.py       # Brand -> RGB color + identity hash + style vector
|-- cond_encoder.py    # 4 raw values -> 20 tokens (batch, 20, 1152)
|-- model.py           # FiLM + PixArt wrapper (HardwareAwareDiT)
|-- generate.py        # Inference entry point
|-- train.py           # Adapter fine-tuning with color_loss
|-- utils.py           # Benchmark, resolution/step interpolation, VAE decode
|-- config.yaml        # All tunable parameters
|-- requirements.txt
|-- models/
    |-- pixart_sigma/  # Local PixArt-Sigma (transformer + VAE, ~2.8 GB)
        |-- transformer/
        |-- vae/
        |-- scheduler/
```

## File responsibilities

| File | Role |
|------|------|
| `detect.py` | Reads CPU/GPU model names via `psutil` + `py-cpuinfo` |
| `color_map.py` | Brand -> RGB color (preset or hash), identity hash (MD5 of CPU+GPU string), style vector (device-personalized landscape/abstract probability) |
| `cond_encoder.py` | `HardwareConditionEncoder`: expands 4 raw scalars into 20 tokens at `hidden_dim=1152` (dynamically matched to PixArt's `cross_attention_dim`) |
| `model.py` | `HardwareAwareDiT`: wraps frozen PixArt transformer, adds FiLM modulation + routes hardware tokens as `encoder_hidden_states` |
| `generate.py` | Orchestrates benchmark -> resolution/steps calculation -> condition encoding -> DDIM loop -> VAE decode -> save |
| `train.py` | Trains only the adapter layers (cond_encoder + film_gamma/beta) with diffusion MSE loss + optional L1 color constraint |
| `utils.py` | `benchmark_device` (matmul-based), `get_resolution`/`get_steps` (linear interpolation), `decode_latents` (VAE decode with scaling_factor) |
| `config.yaml` | Model paths, benchmark params, resolution range (512-1024), steps range (10-50), color loss weight |

## Current limitations

1. **No text semantics**: Identity is encoded via MD5 hash -> Embedding lookup, not a text encoder. "Intel Core i9-13900K" and "Intel Core i9-13900KS" map to completely unrelated hash buckets.
2. **Untrained adapters**: `cond_encoder`, `film_gamma`, `film_beta` are randomly initialized. Without training, hardware conditions produce random perturbations rath6er than meaningful stylistic control.
3. **color_loss is crude**: L1 between generated-mean-RGB and target-RGB can push the image toward a flat color wash.
4. **PixArt was trained with T5 text**: We are hijacking a pathway designed for rich semantic embeddings with our sparse 20-token hardware sequence. The model may not respond strongly without fine-tuning.

## Training plan

1. **Dataset**: 10k+ abstract/aesthetic images (LAION-Aesthetics, WikiArt, or self-curated)
2. **Stage 1** (color_loss=0): Train adapter layers on pure diffusion MSE — teach the model to map hardware tokens to visual structure
3. **Stage 2** (color_loss=0.05): Enable light color constraint — teach brand colors without washing out image detail
4. **Future**: Replace MD5 hash embedding with a lightweight text encoder (e.g. TinyCLIP) for semantic brand understanding
