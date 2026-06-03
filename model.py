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

        # PixArt's built-in cross-attention expects encoder_hidden_states
        # of this dimension. We match our cond tokens to it.
        cond_dim = getattr(transformer.config, "cross_attention_dim",
                          getattr(transformer.config, "hidden_size", 1152))

        self.cond_dim = cond_dim
        self.cond_encoder = HardwareConditionEncoder(hidden_dim=cond_dim)

        in_channels = getattr(transformer.config, "in_channels", 4)

        # FiLM: pre-transformer channel-wise modulation
        self.film_gamma = nn.Linear(cond_dim, in_channels)
        self.film_beta = nn.Linear(cond_dim, in_channels)

        # Use PixArt's pre-trained caption_projection for T5 color embeddings
        self.caption_projection = transformer.caption_projection

        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.transformer.parameters():
            param.requires_grad = False

    def _project_color_emb(self, color_emb: torch.Tensor) -> torch.Tensor:
        """
        Project T5 color embedding through PixArt's pre-trained caption_projection.
        Preserves the full 128-token sequence — the exact format PixArt was trained with.
        color_emb: (batch, 128, 4096) — cached T5 embeddings
        Returns: (batch, 128, cond_dim) — full projected sequence
        """
        # PixArt's caption_projection is the SAME layer used during training
        # to map T5 outputs → cross-attention space.
        # Input:  (B, 128, 4096) — raw T5 hidden states
        # Output: (B, 128, cond_dim) — projected into PixArt's native space
        return self.caption_projection(color_emb)

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
            color_emb: (batch, 128, 4096) or None — T5 color embeddings
        Returns:
            noise_pred: (batch, 4, H, W) — predicted noise
        """
        # 1) Append color tokens via PixArt's pre-trained caption_projection
        if color_emb is not None:
            col_tokens = self._project_color_emb(color_emb)  # (B, 4, cond_dim)
            cond = torch.cat([cond, col_tokens], dim=1)       # (B, 20, cond_dim)

        # 2) FiLM modulation: condition decides global channel scale & shift
        cond_global = cond.mean(dim=1)  # (batch, cond_dim)
        gamma = self.film_gamma(cond_global).unsqueeze(-1).unsqueeze(-1)  # (B,4,1,1)
        beta  = self.film_beta(cond_global).unsqueeze(-1).unsqueeze(-1)
        modulated = gamma * latents + beta

        # 3) PixArt transformer: hardware tokens -> built-in cross-attention
        raw_out = self.transformer(
            modulated, timesteps, encoder_hidden_states=cond
        ).sample

        # 4) PixArt outputs 8 channels (4 mean + 4 variance); keep only mean
        noise_pred, _ = raw_out.chunk(2, dim=1)
        return noise_pred

    def trainable_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "cond_encoder": self.cond_encoder.state_dict(),
            "film_gamma": self.film_gamma.state_dict(),
            "film_beta": self.film_beta.state_dict(),
        }

    def load_trainable_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        self.cond_encoder.load_state_dict(state["cond_encoder"], strict=True)
        if "film_gamma" in state:
            self.film_gamma.load_state_dict(state["film_gamma"], strict=True)
        if "film_beta" in state:
            self.film_beta.load_state_dict(state["film_beta"], strict=True)
