from __future__ import annotations

from typing import Any, Dict

import torch
from diffusers import Transformer2DModel
from torch import nn

from src.cond_encoder import HardwareConditionEncoder


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

        self.cond_dim = cond_dim
        self.cond_encoder = HardwareConditionEncoder(hidden_dim=cond_dim)

        in_channels = getattr(transformer.config, "in_channels", 4)

        # FiLM: pre-transformer channel-wise modulation
        self.film_gamma = nn.Linear(cond_dim, in_channels)
        self.film_beta = nn.Linear(cond_dim, in_channels)

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
        encoder_attention_mask: torch.Tensor | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            latents:  (batch, 4, H, W)  — VAE-encoded noise
            timesteps: (batch,)          — diffusion timestep
            cond:      (batch, 16, cond_dim) — identity + style + global tokens
            color_emb: (batch, 128, 4096) or None — raw T5 color embeddings
        """
        # 1) FiLM modulation (disabled during inference: random weights degrade output)
        #    During training, uncomment to learn meaningful color shifts.
        # cond_global = cond.mean(dim=1)
        # gamma = self.film_gamma(cond_global).unsqueeze(-1).unsqueeze(-1)
        # beta  = self.film_beta(cond_global).unsqueeze(-1).unsqueeze(-1)
        # modulated = gamma * latents + beta
        modulated = latents  # pass-through until FiLM is trained

        # 2) Build encoder_hidden_states at 1152-dim (cross_attention_dim).
        if color_emb is not None:
            color_projected = self.transformer.caption_projection(color_emb)
            if cond is not None and cond.shape[1] > 0:
                encoder_hidden_states = torch.cat([cond, color_projected], dim=1)
                if encoder_attention_mask is not None:
                    cond_mask = torch.ones(
                        cond.shape[:2],
                        dtype=encoder_attention_mask.dtype,
                        device=encoder_attention_mask.device,
                    )
                    encoder_attention_mask = torch.cat([cond_mask, encoder_attention_mask], dim=1)
            else:
                encoder_hidden_states = color_projected
        elif cond is not None:
            encoder_hidden_states = cond
        else:
            raise ValueError("Either cond or color_emb must be provided")

        # 3) Temporarily replace caption_projection with identity (pass-through)
        #    since our encoder_hidden_states is already at 1152-dim.
        original_caption_proj = self.transformer.caption_projection
        self.transformer.caption_projection = nn.Identity()

        try:
            raw_out = self.transformer(
                modulated,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        finally:
            self.transformer.caption_projection = original_caption_proj

        # 4) PixArt outputs 8 channels → keep only mean
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
