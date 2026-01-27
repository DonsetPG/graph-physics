from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.transforms import KNNGraph


class UpSampler(nn.Module):
    def __init__(self, d_in: int, d_out: int, k: int = 6):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(d_in, d_out)

    @torch.compiler.disable
    def forward(
        self,
        x_coarse: torch.Tensor,  # [C, d_in]
        pos_coarse: torch.Tensor,  # [C, pos_dim]
        pos_fine: torch.Tensor,  # [N, pos_dim]
        batch_coarse: Optional[torch.Tensor] = None,
        batch_fine: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        interp = knn_interpolate(
            x=x_coarse,
            pos_x=pos_coarse,
            pos_y=pos_fine,
            batch_x=batch_coarse,
            batch_y=batch_fine,
            k=self.k,
        )  # [N, d_out]
        return self.lin(interp)


class DownSampler(nn.Module):

    def __init__(self, d_in: int, d_out: int, ratio: int = 0.25):

        super().__init__()

        self.ratio = ratio
        self.lin = nn.Linear(d_in, d_out)
        self.min_score = None
        self.nonlinearity = "softmax"

        self.select = SelectTopK(d_in, self.ratio, self.min_score, self.nonlinearity)
        self.remesher = KNNGraph(k=6, force_undirected=True)

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        attn: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        attn = x if attn is None else attn
        select_out = self.select(attn, batch)

        perm = select_out.node_index

        x_c = self.lin(x[perm])
        pos_c = pos[perm]
        batch_c = batch[perm]

        coarse_graph = Batch(
            x=x_c,
            batch=batch_c,
            pos=pos_c,
        )

        return self.remesher(coarse_graph)
