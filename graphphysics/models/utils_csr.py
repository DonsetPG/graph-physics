"""
Utilities for working with sparse graph representations required by the
LocalFlashK8 attention layer.

This module focuses on converting PyG `edge_index` tensors to CSR
representations and constructing a fixed fanout neighbour index with
exactly eight entries per node.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch_sparse import SparseTensor


def edge_index_to_csr(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a `[2, E]` edge index tensor into CSR `(rowptr, col)` tensors."""

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            "edge_index must have shape [2, E]; got "
            f"{tuple(edge_index.shape)}"
        )

    row = edge_index[0].to(torch.int64, copy=True).cpu()
    col = edge_index[1].to(torch.int64, copy=True).cpu()

    sparse = SparseTensor(
        row=row,
        col=col,
        sparse_sizes=(num_nodes, num_nodes),
        is_sorted=False,
    )
    rowptr64, col64, _ = sparse.csr()

    rowptr = rowptr64.to(torch.int32)
    col = col64.to(torch.int32)

    return rowptr.to(device), col.to(device)


def build_fixed_fanout_k8(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    include_self: bool = True,
    sort_neighbors: bool = True,
    device: torch.device | None = None,
    K: int = 8,
) -> torch.Tensor:
    """
    Construct a fixed-size neighbour index `idx_k8` with exactly K (8) entries per node.

    Padding repeats the last available neighbour and fall
    back to self-loops for isolated nodes.
    """

    rowptr_cpu = rowptr.to(torch.int64, copy=True).cpu()
    col_cpu = col.to(torch.int64, copy=True).cpu()
    num_nodes = rowptr_cpu.numel() - 1

    idx_rows = []
    for node in range(num_nodes):
        start = rowptr_cpu[node].item()
        end = rowptr_cpu[node + 1].item()
        neighbours = col_cpu[start:end]

        if include_self:
            if not (neighbours == node).any():
                neighbours = torch.cat(
                    [neighbours, torch.tensor([node], dtype=torch.int64)]
                )

        if sort_neighbors and neighbours.numel() > 1:
            neighbours, _ = torch.sort(neighbours)

        if neighbours.numel() == 0:
            neighbours = torch.tensor([node], dtype=torch.int64)

        if neighbours.numel() >= K:
            trimmed = neighbours[:K]
        else:
            pad_value = neighbours[-1]
            padding = pad_value.repeat(K - neighbours.numel())
            trimmed = torch.cat([neighbours, padding])

        idx_rows.append(trimmed.to(torch.int32))

    idx_k8 = torch.stack(idx_rows, dim=0)
    return idx_k8.to(device=device, dtype=torch.int32)


__all__ = ["edge_index_to_csr", "build_fixed_fanout_k8"]
