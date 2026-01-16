from __future__ import annotations

from typing import Optional

from pytest import param
import torch
import torch.nn as nn
import math
from torch_geometric.data import Batch
from torch_geometric.nn import fps
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.transforms import Cartesian, Compose, Distance, KNNGraph
from torch_geometric.utils import subgraph
from loguru import logger

from graphphysics.utils.nodetype import NodeType
from graphphysics.utils.physical_loss import PhysicalLoss

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
        ratio: float = 0.5,
        k: int = 6,
        method: str = "fps",  # "topk", "random", "fps", "residu", "residu_prob"
        is_remeshing: bool = True,
        node_mask: str = "all",  # "normal", "all"
        sampling_temperature: float = 100.0,
    ):
        super().__init__()
        self.ratio = ratio
        self.method = method
        self.sampling_temperature = sampling_temperature
        self.is_remeshing = is_remeshing
        self.node_mask = node_mask

        self.lin = nn.Linear(d_in, d_out)
        self.edge_dim = edge_dim
        self.remesher = KNNGraph(k=k, force_undirected=True)

        if self.method == "topk":
            self.select = SelectTopK(
                d_in, ratio, min_score=None
            )
        else:
            self.select = None

    def _build_edge_features(self, graph: Batch) -> torch.Tensor:
        senders, receivers = graph.edge_index
        rel_pos = graph.pos[senders] - graph.pos[receivers]
        rel_norm = torch.norm(rel_pos, p=2, dim=-1, keepdim=True)
        features = torch.cat([rel_pos, rel_norm], dim=-1)
        return features

    def _topk_by_batch(
        self, scores: torch.Tensor, batch: Optional[torch.Tensor]
    ) -> torch.Tensor:
        scores = scores.view(-1)
        if batch is None:
            num_nodes = scores.size(0)
            k = int(self.ratio * num_nodes) if self.ratio < 1 else int(self.ratio)
            k = max(1, min(num_nodes, k))
            return torch.topk(scores, k=k, largest=True).indices

        perm = []
        for batch_idx in batch.unique(sorted=True):
            mask = batch == batch_idx
            idx = mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            num_nodes = idx.numel()
            k = int(self.ratio * num_nodes) if self.ratio < 1 else int(self.ratio)
            k = max(1, min(num_nodes, k))
            topk = torch.topk(scores[idx], k=k, largest=True).indices
            perm.append(idx[topk])

        if perm:
            return torch.cat(perm, dim=0)
        return torch.empty(0, dtype=torch.long, device=scores.device)

    def _prob_sample(self, scores: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
        scores = scores.view(-1)
        if scores.numel() == 0 or k <= 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        if k >= scores.numel():
            return torch.arange(scores.numel(), device=scores.device)
        if temperature <= 0:
            raise ValueError("sampling_temperature must be > 0 for probabilistic sampling.")
        logits = scores.float() / temperature
        logits = logits - logits.max()
        probs = torch.softmax(logits, dim=0)
        if torch.isnan(probs).any() or probs.sum() <= 0:
            probs = torch.ones_like(scores, dtype=torch.float)
            probs = probs / probs.numel()
        return torch.multinomial(probs, k, replacement=False)

    def _prob_sample_by_batch(
        self, scores: torch.Tensor, batch: Optional[torch.Tensor], temperature: float
    ) -> torch.Tensor:
        scores = scores.view(-1)
        if batch is None:
            k = self._num_keep(scores.numel())
            return self._prob_sample(scores, k, temperature)

        perm = []
        for batch_idx in batch.unique(sorted=True):
            mask = batch == batch_idx
            idx = mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            k = self._num_keep(idx.numel())
            sampled = self._prob_sample(scores[idx], k, temperature)
            perm.append(idx[sampled])

        if perm:
            return torch.cat(perm, dim=0)
        return torch.empty(0, dtype=torch.long, device=scores.device)

    def _resolve_candidate_nodes(self, graph: Batch) -> torch.Tensor:
        num_nodes = graph.num_nodes
        device = graph.x.device
        mask_setting = self.node_mask
        if mask_setting is None:
            mask_setting = "all"
        mask_setting = str(mask_setting).lower()
        if mask_setting in ("all", "none"):
            return torch.arange(num_nodes, device=device)
        if mask_setting in ("normal", "normal_only"):
            if not hasattr(graph, "node_type") or graph.node_type is None:
                raise ValueError("node_mask='normal' requires graph.node_type.")
            candidate = torch.nonzero(
                graph.node_type == NodeType.NORMAL, as_tuple=False
            ).view(-1)
            if candidate.numel() == 0:
                logger.warning(
                    "node_mask='normal' matched 0 nodes; falling back to all nodes."
                )
                return torch.arange(num_nodes, device=device)
            return candidate
        raise ValueError(
            f"Unknown node_mask '{self.node_mask}'. Expected 'normal' or 'all'."
        )

    def _num_keep(self, num_nodes: int) -> int:
        k = int(self.ratio * num_nodes) if self.ratio < 1 else int(self.ratio)
        return max(1, min(num_nodes, k))

    def _sample_nodes(self, graph: Batch, attn: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = graph.x.device
        batch = getattr(graph, "batch", None)
        candidate = self._resolve_candidate_nodes(graph)
        batch_candidate = batch[candidate] if batch is not None else None
        if candidate.numel() == 0:
            return candidate

        if self.method == "random":
            num_keep = self._num_keep(candidate.numel())
            perm = torch.randperm(candidate.numel(), device=device)[:num_keep]
            return candidate[perm]

        if self.method == "fps":
            perm = fps(graph.pos[candidate], batch=batch_candidate, ratio=self.ratio)
            return candidate[perm]

        if self.method == "topk":
            # If external scores are provided (e.g., physical residuals), use them directly.
            if attn is not None:
                scores = attn.squeeze(-1)[candidate]
                perm = self._topk_by_batch(scores, batch_candidate)
                return candidate[perm]
            scores = graph.x[candidate]
            perm = self.select(scores, batch=batch_candidate).node_index
            return candidate[perm]

        if self.method == "residu":
            scores = attn
            if scores is None:
                raise ValueError("No residual scores provided for 'residu' downsampling.")
            scores = scores.squeeze(-1)[candidate]
            perm = self._topk_by_batch(scores, batch_candidate)
            return candidate[perm]

        if self.method == "residu_prob":
            scores = attn
            if scores is None:
                raise ValueError("No residual scores provided for 'residu_prob' downsampling.")
            scores = scores.squeeze(-1)[candidate]
            perm = self._prob_sample_by_batch(
                scores, batch_candidate, self.sampling_temperature
            )
            return candidate[perm]

        raise ValueError(f"Unknown method: {self.method}")

    @torch.compiler.disable
    def forward(self, graph: Batch, attn: Optional[torch.Tensor] = None) -> Batch:
        perm = self._sample_nodes(graph, attn)
        x_c = self.lin(graph.x[perm])
        pos_c = graph.pos[perm]
        batch = getattr(graph, "batch", None)
        batch_c = batch[perm] if batch is not None else None
        coarse = Batch(x=x_c, pos=pos_c, batch=batch_c)
        if self.is_remeshing:
            coarse = self.remesher(coarse)
        else:
            edge_index_c, _ = subgraph(
                perm,
                graph.edge_index,
                edge_attr=None,  # keep None and recompute below
                relabel_nodes=True,
                num_nodes=graph.num_nodes,
            )
            coarse.edge_index = edge_index_c
        coarse.edge_attr = self._build_edge_features(coarse)
        coarse.pool_perm = perm
        coarse.node_type = graph.node_type[perm] if hasattr(graph, "node_type") else ValueError("Graph must have 'node_type' attribute for downsampling.")
        return coarse
