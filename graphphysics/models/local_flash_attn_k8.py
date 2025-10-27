"""
Local FlashAttention layer with fixed K=8 neighbourhoods.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import profiler as autograd_profiler

from graphphysics.kernels.triton_gather_pack_k8 import gather_pack_k8_triton
from graphphysics.models.utils_csr import (
    build_fixed_fanout_k8,
    edge_index_to_csr,
)

try:  # pragma: no cover - optional dependency
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
except ImportError:  # pragma: no cover
    try:
        from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore

        HAS_FLASH_ATTN = True
    except ImportError:
        flash_attn_func = None
        HAS_FLASH_ATTN = False


def _manual_pack(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Reference gather used when Triton is disabled."""
    idx_long = idx.to(torch.long, device=x.device)
    gathered = x.index_select(0, idx_long.reshape(-1))
    gathered = gathered.reshape(idx.shape[0], idx.shape[1], x.shape[1], x.shape[2])
    return gathered.transpose(1, 2).contiguous()


class LocalFlashK8(nn.Module):
    """Local attention over fixed fanout K=8 neighbourhoods."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        include_self: bool = True,
        sort_neighbors: bool = True,
        chunk_nodes: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        use_triton: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        if d_model != n_heads * head_dim:
            raise ValueError("d_model must equal n_heads * head_dim.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.include_self = include_self
        self.sort_neighbors = sort_neighbors
        self.chunk_nodes = chunk_nodes
        self.use_triton = use_triton
        self.use_flash_attn = use_flash_attn

        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_dropout = nn.Dropout(dropout)

        if dtype is not None:
            self._set_dtype(dtype)

    def _set_dtype(self, dtype: torch.dtype) -> None:
        self.qkv.to(dtype=dtype)
        self.out_proj.to(dtype=dtype)

    def _maybe_build_idx(
        self,
        x: torch.Tensor,
        idx_k8: Optional[torch.Tensor],
        rowptr: Optional[torch.Tensor],
        col: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if idx_k8 is not None:
            return idx_k8.to(device=x.device, dtype=torch.int32)

        if rowptr is not None and col is not None:
            idx = build_fixed_fanout_k8(
                rowptr=rowptr,
                col=col,
                include_self=self.include_self,
                sort_neighbors=self.sort_neighbors,
                device=x.device,
            )
            return idx

        if edge_index is not None:
            num_nodes = x.size(0)
            rowptr_cpu, col_cpu = edge_index_to_csr(
                edge_index=edge_index,
                num_nodes=num_nodes,
            )
            idx = build_fixed_fanout_k8(
                rowptr=rowptr_cpu,
                col=col_cpu,
                include_self=self.include_self,
                sort_neighbors=self.sort_neighbors,
                device=x.device,
            )
            return idx

        raise ValueError(
            "Provide either idx_k8, (rowptr & col), or edge_index for neighbour definition."
        )

    def _gather(self, tensor: torch.Tensor, idx_k8: torch.Tensor) -> torch.Tensor:
        if self.use_triton:
            return gather_pack_k8_triton(tensor, idx_k8)
        return _manual_pack(tensor, idx_k8)

    def _attention_chunk(
        self,
        q_chunk: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        idx_chunk: torch.Tensor,
    ) -> torch.Tensor:
        with autograd_profiler.record_function("local_flash_k8.pack_k"):
            k_packed = self._gather(k_tensor, idx_chunk)
        with autograd_profiler.record_function("local_flash_k8.pack_v"):
            v_packed = self._gather(v_tensor, idx_chunk)

        dropout_p = self.dropout if self.training else 0.0

        if (
            self.use_flash_attn
            and HAS_FLASH_ATTN
            and q_chunk.is_cuda
            and k_packed.is_cuda
        ):
            with autograd_profiler.record_function("local_flash_k8.flash_attn"):
                q_flash = q_chunk.unsqueeze(1).contiguous()
                k_flash = k_packed.permute(0, 2, 1, 3).contiguous()
                v_flash = v_packed.permute(0, 2, 1, 3).contiguous()
                out = flash_attn_func(
                    q_flash,
                    k_flash,
                    v_flash,
                    dropout_p=dropout_p,
                    softmax_scale=None,
                    causal=False,
                )
            return out.squeeze(1)

        # Fallback
        with autograd_profiler.record_function("local_flash_k8.sdpa"):
            scores = torch.matmul(
                q_chunk.unsqueeze(2), k_packed.transpose(-1, -2)
            ).squeeze(2)
            scores = scores * self.scale
            weights = F.softmax(scores, dim=-1)
            if dropout_p > 0.0:
                weights = F.dropout(weights, p=dropout_p, training=self.training)
            context = torch.matmul(weights.unsqueeze(2), v_packed).squeeze(2)
        return context.to(dtype=q_chunk.dtype)

    def forward(
        self,
        x: torch.Tensor,
        idx_k8: Optional[torch.Tensor] = None,
        rowptr: Optional[torch.Tensor] = None,
        col: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        idx = self._maybe_build_idx(x, idx_k8, rowptr, col, edge_index)

        if idx.shape[0] != x.shape[0]:
            raise ValueError(
                f"idx_k8 first dimension ({idx.shape[0]}) must equal number of nodes ({x.shape[0]})."
            )

        with autograd_profiler.record_function("local_flash_k8.qkv"):
            qkv = self.qkv(x)
        qkv = qkv.view(x.shape[0], 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)

        chunk = self.chunk_nodes or idx.shape[0]
        outputs = []
        for start in range(0, idx.shape[0], chunk):
            end = min(start + chunk, idx.shape[0])
            idx_chunk = idx[start:end]
            q_chunk = q[start:end]
            context = self._attention_chunk(q_chunk, k, v, idx_chunk)
            outputs.append(context)

        attn_out = torch.cat(outputs, dim=0)
        attn_out = attn_out.reshape(x.shape[0], self.n_heads * self.head_dim)

        with autograd_profiler.record_function("local_flash_k8.out_proj"):
            y = self.out_proj(attn_out)
        y = self.out_dropout(y)
        return y


__all__ = ["LocalFlashK8", "HAS_FLASH_ATTN"]
