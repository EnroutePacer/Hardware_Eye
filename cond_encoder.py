from __future__ import annotations

import torch
from torch import nn


class HardwareConditionEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 512, hash_buckets: int = 10000) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.hash_buckets = hash_buckets
        
        # 8 identity + 4 style + 4 global = 16 tokens.
        # Color tokens (4) are produced separately by PixArt's caption_projection.
        
        # 1. Identity encoder (embeds the hash into 8 tokens)
        self.embed_identity = nn.Sequential(
            nn.Embedding(hash_buckets, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 8)
        )
        
        # 2. Style encoder (prob_vec -> 4 tokens)
        self.fc_style = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 4),
        )
        
        # 3. Global encoder (combines identity + style -> 4 tokens)
        self.fc_global = nn.Sequential(
            nn.Linear(hidden_dim * 12, hidden_dim * 4), 
        )

    def forward(
        self,
        conditions_batch: list[dict],
        perf_index_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns: tensor of shape (batch, 16, hidden_dim)
        8 identity + 4 style + 4 global tokens
        """
        device = next(self.parameters()).device
        out_tokens = []
        
        for i, cond in enumerate(conditions_batch):
            identity_hash = cond["identity_hash"] % self.hash_buckets
            hash_idx = torch.tensor([identity_hash], device=device)
            ident_tokens = self.embed_identity(hash_idx).reshape(8, self.hidden_dim)
            
            style_vec = cond["style_vector"].to(device)
            style_tokens = self.fc_style(style_vec).reshape(4, self.hidden_dim)
            
            combined = torch.cat([ident_tokens, style_tokens], dim=0).flatten()
            global_tokens = self.fc_global(combined).reshape(4, self.hidden_dim)
            
            seq = torch.cat([ident_tokens, style_tokens, global_tokens], dim=0)
            out_tokens.append(seq)
            
        return torch.stack(out_tokens)
