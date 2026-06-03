from __future__ import annotations

import argparse
import os
import random
from itertools import cycle
from typing import List

import torch
import torch.nn.functional as F
import yaml
from diffusers import AutoencoderKL, DDPMScheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from color_map import get_brand_pool, get_hardware_conditions
from detect import get_device
from model import HardwareAwareDiT, load_pretrained_transformer
from utils import decode_latents, ensure_dir, seed_everything

COLOR_CACHE_PATH = os.path.join(os.path.dirname(__file__), "colorEmb_cache", "brand_color_embeddings.pt")
_FALLBACK_BRAND = "unknown"


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: str, resolution: int) -> None:
        self.image_paths = self._collect_image_paths(image_dir)
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    @staticmethod
    def _collect_image_paths(image_dir: str) -> List[str]:
        paths: List[str] = []
        for root, _, files in os.walk(image_dir):
            for name in files:
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    paths.append(os.path.join(root, name))
        return paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sample_conditions(
    batch_size: int, device: torch.device, jitter: float
) -> tuple[torch.Tensor, list[dict]]:
    brands = list(get_brand_pool())
    perf = torch.rand(batch_size, device=device) * 0.8 + 0.2
    conditions_batch = []
    for idx in range(batch_size):
        cpu_brand = random.choice(brands)
        gpu_brand = random.choice(brands)
        profile = {"cpu_name": cpu_brand, "gpu_name": gpu_brand}
        cond_dict = get_hardware_conditions(profile, float(perf[idx].item()), jitter=jitter)
        conditions_batch.append(cond_dict)
    return perf, conditions_batch


def compute_color_loss(
    vae: AutoencoderKL, latents: torch.Tensor, target_colors: torch.Tensor
) -> torch.Tensor:
    images = decode_latents(vae, latents)
    mean_color = images.mean(dim=(2, 3))
    return F.l1_loss(mean_color, target_colors)


def save_checkpoint(model: HardwareAwareDiT, output_dir: str, step: int, config: dict) -> None:
    ensure_dir(output_dir)
    payload = {
        "hardware_eye": model.trainable_state_dict(),
        "config": config,
        "step": step,
    }
    step_path = os.path.join(output_dir, f"hardware_eye_step_{step}.pt")
    latest_path = os.path.join(output_dir, "hardware_eye_step_latest.pt")
    torch.save(payload, step_path)
    torch.save(payload, latest_path)


def make_infinite_loader(dataset: Dataset, batch_size: int, **kwargs):
    """Yields infinite batches from a dataset (cycles forever)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return cycle(loader)


def fetch_style_aligned_batch(
    landscape_iter,
    abstract_iter,
    conditions_batch: list[dict],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    For each sample in conditions_batch, fetch an image from the folder
    that matches its style_vector: landscape if style[0] > 0.5, else abstract.
    """
    frames = []
    for i in range(batch_size):
        cond = conditions_batch[i]
        if cond["style_vector"][0] > 0.5:
            img = next(landscape_iter)
        else:
            img = next(abstract_iter)
        # img is (B, C, H, W); take first item in batch
        frames.append(img[0:1] if img.dim() == 4 else img.unsqueeze(0))
    return torch.cat(frames, dim=0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["generate"]["seed"]))

    device = get_device()
    transformer = load_pretrained_transformer(config["model"]["pretrained_transformer"])
    model = HardwareAwareDiT(transformer).to(device)

    vae = AutoencoderKL.from_pretrained(config["model"]["vae_path"]).to(device)
    vae.requires_grad_(False)
    vae.eval()

    scheduler = DDPMScheduler(num_train_timesteps=int(config["diffusion"]["train_timesteps"]))

    resolution = int(config["train"]["resolution"])
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"]["num_workers"])

    # ---- dual-folder datasets: landscape/ and abstract/ ----
    landscape_dataset = ImageFolderDataset(config["train"]["landscape_dir"], resolution)
    abstract_dataset  = ImageFolderDataset(config["train"]["abstract_dir"], resolution)

    landscape_iter = make_infinite_loader(
        landscape_dataset, batch_size,
        num_workers=num_workers, pin_memory=device.type == "cuda"
    )
    abstract_iter = make_infinite_loader(
        abstract_dataset, batch_size,
        num_workers=num_workers, pin_memory=device.type == "cuda"
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(config["train"]["lr"]))
    use_amp = str(config["train"].get("mixed_precision", "no")).lower() == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    max_steps = int(config["train"]["num_steps"])
    grad_accum = int(config["train"]["gradient_accumulation"])
    save_every = int(config["train"]["save_every"])
    color_weight = float(config["train"]["color_loss_weight"])
    jitter = float(config["train"]["jitter"])

    # Load pre-computed T5 color embeddings
    color_cache = torch.load(COLOR_CACHE_PATH, map_location="cpu")

    # ---- Freeze FiLM: only train style + identity ----
    # (caption_projection is PixArt's pre-trained layer, always frozen)
    model.film_gamma.requires_grad_(False)
    model.film_beta.requires_grad_(False)
    print("Trainable: embed_identity, fc_style, fc_global only")

    print(f"Landscape images: {len(landscape_dataset)}, Abstract images: {len(abstract_dataset)}")

    global_step = 0
    model.train()

    progress = tqdm(total=max_steps, desc="training", dynamic_ncols=True)
    while global_step < max_steps:
        # ---- Phase 1: generate conditions first ----
        perf_index, conditions_batch = sample_conditions(batch_size, device, jitter)

        # ---- Phase 2: fetch images matching the style ----
        batch = fetch_style_aligned_batch(
            landscape_iter, abstract_iter, conditions_batch, device, batch_size
        )

        with torch.no_grad():
            latents = vae.encode(batch).latent_dist.sample()
            if hasattr(vae.config, "scaling_factor"):
                latents = latents * vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (latents.size(0),), device=device
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        with torch.cuda.amp.autocast(enabled=use_amp):
            # Blend CPU + GPU color embeddings: 4:6 ratio
            color_embs = []
            for c in conditions_batch:
                cpu_emb = color_cache.get(c.get("cpu_brand", _FALLBACK_BRAND),
                                          color_cache[_FALLBACK_BRAND])
                gpu_brand = c.get("gpu_brand")
                if gpu_brand:
                    gpu_emb = color_cache.get(gpu_brand, color_cache[_FALLBACK_BRAND])
                    blended = 0.4 * cpu_emb + 0.6 * gpu_emb
                else:
                    blended = cpu_emb
                color_embs.append(blended.squeeze(0))  # (128, 4096)
            color_emb_batch = torch.stack(color_embs).to(device)  # (B, 128, 4096)

            cond = model.cond_encoder(conditions_batch, perf_index)
            noise_pred = model.forward_with_cond(noisy_latents, timesteps, cond, color_emb_batch)

            # diffusion MSE loss
            loss = F.mse_loss(noise_pred, noise)

            # color_loss: push average image color toward target RGB
            if color_weight > 0:
                alphas = scheduler.alphas_cumprod.to(device)[timesteps].view(-1, 1, 1, 1)
                pred_x0 = (noisy_latents - (1 - alphas).sqrt() * noise_pred) / alphas.sqrt()
                target_colors = torch.stack([c["color_rgb"] for c in conditions_batch]).to(device)
                loss = loss + color_weight * compute_color_loss(vae, pred_x0, target_colors)

        scaler.scale(loss / grad_accum).backward()

        if (global_step + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        progress.update(1)
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

        if global_step % save_every == 0:
            save_checkpoint(model, config["train"]["output_dir"], global_step, config)

    progress.close()
    save_checkpoint(model, config["train"]["output_dir"], global_step, config)


if __name__ == "__main__":
    main()
