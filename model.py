from __future__ import annotations

from typing import Any, Dict

import torch
from diffusers import Transformer2DModel
from torch import nn

from cond_encoder import HardwareConditionEncoder


def load_pretrained_transformer(model_id: str, revision: str | None = None) -> Transformer2DModel:
    """Load PixArt-Sigma transformer from local path or HuggingFace hub."""
    return Transformer2DModel.from_pretrained(model_id, revision=revision)


class HardwareAwareDiT(nn.Module):
    def __init__(
        self,
        transformer: Transformer2DModel,
    ) -> None:
        super().__init__()
        self.transformer = transformer

        cond_dim = getattr(transformer.config, "cross_attention_dim",
                          getattr(transformer.config, "hidden_size", 1152))
        caption_dim = getattr(transformer.config, "caption_channels", 4096)

        self.cond_dim = cond_dim
        self.caption_dim = caption_dim
        self.cond_encoder = HardwareConditionEncoder(hidden_dim=cond_dim)

        in_channels = getattr(transformer.config, "in_channels", 4)

        # FiLM: pre-transformer channel-wise modulation
        self.film_gamma = nn.Linear(cond_dim, in_channels)
        self.film_beta = nn.Linear(cond_dim, in_channels)

        # Project our 1152-dim cond tokens → 4096-dim T5 embedding space,
        # so PixArt's native caption_projection can process them uniformly
        self.cond_to_t5_proj = nn.Linear(cond_dim, caption_dim)

        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward_with_cond(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
        color_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            latents:  (batch, 4, H, W)  — VAE-encoded noise
            timesteps: (batch,)          — diffusion timestep
            cond:      (batch, 16, cond_dim) — identity + style + global tokens
            color_emb: (batch, 128, 4096) or None — raw T5 color embeddings
        Returns:
            noise_pred: (batch, 4, H, W) — predicted noise
        """
        # 1) FiLM modulation: uses only identity/style/global tokens (cond_dim=1152)
        cond_global = cond.mean(dim=1)
        gamma = self.film_gamma(cond_global).unsqueeze(-1).unsqueeze(-1)
        beta  = self.film_beta(cond_global).unsqueeze(-1).unsqueeze(-1)
        modulated = gamma * latents + beta

        # 2) Build encoder_hidden_states in T5 space (4096-dim)
        #    Project our 16 hardware tokens into T5 embedding space
        cond_t5 = self.cond_to_t5_proj(cond)  # (B, 16, 4096)

        if color_emb is not None:
            # Raw T5 color embedding — already 4096-dim, no pre-projection needed
            encoder_hidden_states = torch.cat([cond_t5, color_emb], dim=1)  # (B, 144, 4096)
        else:
            encoder_hidden_states = cond_t5  # (B, 16, 4096)

        # 3) PixArt transformer internally does:
        #    encoder_hidden_states = caption_projection(encoder_hidden_states)
        #    — this maps (B, *, 4096) → (B, *, 1152) uniformly for ALL tokens
        raw_out = self.transformer(
            modulated, timestep=timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample

        # 4) PixArt outputs 8 channels → keep only mean
        noise_pred, _ = raw_out.chunk(2, dim=1)
        return noise_pred

    def trainable_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "cond_encoder": self.cond_encoder.state_dict(),
            "film_gamma": self.film_gamma.state_dict(),
            "film_beta": self.film_beta.state_dict(),
            "cond_to_t5_proj": self.cond_to_t5_proj.state_dict(),
        }

    def load_trainable_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        self.cond_encoder.load_state_dict(state["cond_encoder"], strict=True)
        if "film_gamma" in state:
            self.film_gamma.load_state_dict(state["film_gamma"], strict=True)
        if "film_beta" in state:
            self.film_beta.load_state_dict(state["film_beta"], strict=True)
        if "cond_to_t5_proj" in state:
            self.cond_to_t5_proj.load_state_dict(state["cond_to_t5_proj"], strict=True)
