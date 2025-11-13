from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data
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
    def __init__(
        self,
        d_in: int,
        d_out: int,
        edge_dim: int,
        ratio: float = 0.25,
        k: int = 6,
    ):
        super().__init__()
        self.ratio = ratio
        self.lin = nn.Linear(d_in, d_out)
        self.min_score = None
        self.nonlinearity = "softmax"
        self.edge_dim = edge_dim
        self.select = SelectTopK(d_in, self.ratio, self.min_score, self.nonlinearity)
        self.remesher = KNNGraph(k=k, force_undirected=True)

    def _build_edge_features(self, graph: Data) -> torch.Tensor:
        senders, receivers = graph.edge_index
        rel_pos = graph.pos[senders] - graph.pos[receivers]
        rel_norm = torch.norm(rel_pos, p=2, dim=-1, keepdim=True)
        features = torch.cat([rel_pos, rel_norm], dim=-1)
        return features

    @torch.compiler.disable
    def forward(
        self,
        graph: Data,
        attn: Optional[torch.Tensor] = None,
    ) -> Data:

        scores = graph.x if attn is None else attn
        select_out = self.select(scores, None)
        perm = select_out.node_index

        x_c = self.lin(graph.x[perm])
        pos_c = graph.pos[perm]

        coarse_graph = Data(
            x=x_c,
            pos=pos_c,
        )
        coarse_graph = self.remesher(coarse_graph)
        coarse_graph.edge_attr = self._build_edge_features(coarse_graph)
        return coarse_graph
