"""
Triton kernel to gather and pack K=8 neighbours for attention tiles.

The kernel is optional: when Triton or CUDA is unavailable, a pure PyTorch
reference implementation is used instead.  Shapes:

    x:      [N, H, Dh]
    idx_k8: [B, 8]
    output: [B, H, 8, Dh]
"""
from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

K_FANOUT = 8
BLOCK_D = 64


def _reference_gather(x: torch.Tensor, idx_k8: torch.Tensor) -> torch.Tensor:
    idx_long = idx_k8.to(dtype=torch.long, device=x.device)
    gathered = x.index_select(0, idx_long.reshape(-1))
    gathered = gathered.reshape(idx_long.shape[0], K_FANOUT, x.shape[1], x.shape[2])
    return gathered.transpose(1, 2).contiguous()


def _validate_inputs(x: torch.Tensor, idx_k8: torch.Tensor) -> Tuple[int, int, int]:
    if x.dim() != 3:
        raise ValueError(f"`x` must have shape [N, H, Dh]; got {tuple(x.shape)}")
    if idx_k8.dim() != 2 or idx_k8.size(1) != K_FANOUT:
        raise ValueError(
            f"`idx_k8` must have shape [B, {K_FANOUT}]; got {tuple(idx_k8.shape)}"
        )
    if not torch.is_floating_point(x):
        raise TypeError("`x` must be a floating point tensor.")
    return x.shape


if HAS_TRITON:

    @triton.jit
    def _gather_pack_k8_kernel(
        x_ptr,
        idx_ptr,
        out_ptr,
        N,
        B,
        H,
        Dh,
        stride_x_node,
        stride_x_head,
        stride_x_feat,
        stride_idx_node,
        stride_idx_k,
        stride_out_node,
        stride_out_head,
        stride_out_k,
        stride_out_feat,
        NUM_CHUNKS: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        pid = tl.program_id(0)
        node_id = pid // H
        head_id = pid % H
        if node_id >= B:
            return

        idx_row = idx_ptr + node_id * stride_idx_node
        out_row = out_ptr + node_id * stride_out_node + head_id * stride_out_head
        feat_offsets = tl.arange(0, BLOCK_DMODEL)

        for k in tl.static_range(0, K_FANOUT):
            neighbour = tl.load(idx_row + k * stride_idx_k).to(tl.int32)
            neighbour = tl.max(neighbour, 0)
            neighbour = tl.minimum(neighbour, N - 1)

            x_base = x_ptr + neighbour * stride_x_node + head_id * stride_x_head
            out_base = out_row + k * stride_out_k

            for chunk in tl.static_range(0, NUM_CHUNKS):
                d_offsets = chunk * BLOCK_DMODEL + feat_offsets
                mask = d_offsets < Dh
                values = tl.load(
                    x_base + d_offsets * stride_x_feat,
                    mask=mask,
                    other=0.0,
                )
                tl.store(
                    out_base + d_offsets * stride_out_feat,
                    values,
                    mask=mask,
                )


def gather_pack_k8_triton(x: torch.Tensor, idx_k8: torch.Tensor) -> torch.Tensor:
    """
    Gather neighbour features for K=8 and return `[B, H, 8, Dh]`.

    Falls back to a PyTorch implementation when Triton or CUDA is unavailable.
    """
    N, H, Dh = _validate_inputs(x, idx_k8)
    B = idx_k8.shape[0]

    if (
        not HAS_TRITON
        or not x.is_cuda
        or not idx_k8.is_cuda
        or triton is None
        or tl is None
    ):
        return _reference_gather(x, idx_k8)

    idx_k8 = idx_k8.to(dtype=torch.int32, device=x.device).contiguous()
    x = x.contiguous()

    out = torch.empty(
        (B, H, K_FANOUT, Dh),
        dtype=x.dtype,
        device=x.device,
    )

    strides_x = x.stride()
    strides_idx = idx_k8.stride()
    strides_out = out.stride()

    num_chunks = (Dh + BLOCK_D - 1) // BLOCK_D
    grid = (B * H,)

    _gather_pack_k8_kernel[grid](
        x,
        idx_k8,
        out,
        N,
        B,
        H,
        Dh,
        strides_x[0],
        strides_x[1],
        strides_x[2],
        strides_idx[0],
        strides_idx[1],
        strides_out[0],
        strides_out[1],
        strides_out[2],
        strides_out[3],
        NUM_CHUNKS=num_chunks,
        BLOCK_DMODEL=BLOCK_D,
    )

    return out


__all__ = ["gather_pack_k8_triton", "HAS_TRITON"]
