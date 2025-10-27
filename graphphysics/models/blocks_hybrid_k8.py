from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from graphphysics.models.local_flash_attn_k8 import LocalFlashK8


class HybridGraphBlockK8(nn.Module):
    """LayerNorm → LocalFlashK8 → residual → MLP → residual block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        mlp_ratio: float = 3.0,
        chunk_nodes: int | None = None,
        include_self: bool = True,
        sort_neighbors: bool = True,
        use_triton: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = LocalFlashK8(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            include_self=include_self,
            sort_neighbors=sort_neighbors,
            chunk_nodes=chunk_nodes,
            use_triton=use_triton,
            use_flash_attn=use_flash_attn,
        )
        self.norm_mlp = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp_fc1 = nn.Linear(d_model, hidden_dim)
        self.mlp_fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        idx_k8: torch.Tensor | None = None,
        rowptr: torch.Tensor | None = None,
        col: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.norm_attn(x)
        attn_out = self.attn(
            x_norm,
            idx_k8=idx_k8,
            rowptr=rowptr,
            col=col,
            edge_index=edge_index,
        )
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.norm_mlp(x)
        x = self.mlp_fc1(x_norm)
        x = F.gelu(x)
        x = self.dropout(self.mlp_fc2(x))
        x = residual + x
        return x


__all__ = ["HybridGraphBlockK8"]
