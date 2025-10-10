from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from graphphysics.models.layers import RMSNorm, build_gated_mlp


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = RMSNorm(d_model)
        self.ffn = build_gated_mlp(
            in_size=d_model, hidden_size=d_model, out_size=d_model
        )

    def forward(self, x, key_padding_mask=None):
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class _Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_EncoderBlock(d_model, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x, key_padding_mask=None):
        for lyr in self.layers:
            x = lyr(x, key_padding_mask=key_padding_mask)
        return x


def _make_undirected(edge_index: torch.Tensor, assume_undirected: bool) -> torch.Tensor:
    """
    Return edges with reverse directions added if not assumed undirected. No dedup (fast).
    """
    if assume_undirected:
        return edge_index.long()
    e = edge_index.long()
    rev = torch.stack([e[1], e[0]], dim=0)
    return torch.cat([e, rev], dim=1)


def _sorted_by_src(
    edge_index: torch.Tensor, num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort edges by src to get CSR-like structure.
    Returns (src_sorted, dst_sorted, row_ptr) where row_ptr has shape [N+1].
    """
    e = edge_index.long()
    src, dst = e[0], e[1]
    order = torch.argsort(src)
    src_s = src[order]
    dst_s = dst[order]

    counts = torch.bincount(src_s, minlength=num_nodes)  # [N]
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=e.device)
    row_ptr[1:] = counts.cumsum(0)
    return src_s, dst_s, row_ptr


class SpatialMTP1Hop(nn.Module):
    """
    Fast 1-hop Spatial MTP with ring attention:
      - Vectorized star packing
      - Single CSR-like sort by source per batch
      - Different neighbor inputs (node_encoder outputs), while centers use H
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 1,
        assume_undirected: bool = True,
        max_neighbors: Optional[int] = None,
    ):
        super().__init__()
        self.enc = _Encoder(d_model, num_heads, num_layers)
        self.in_ln = RMSNorm(d_model)
        self.assume_undirected = assume_undirected
        self.max_neighbors = max_neighbors

    @torch.no_grad()
    def _cap_neighbors(
        self, dst_s: torch.Tensor, row_ptr: torch.Tensor, centers: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally cap number of neighbors per selected center to max_neighbors.
        Returns (dst_concat, counts) where dst_concat is the concatenated neighbor list
        and counts[b] is the number kept for centers[b].
        """
        device = dst_s.device
        B = centers.numel()
        starts = row_ptr[centers]
        ends = row_ptr[centers + 1]
        counts = ends - starts  # [B]
        if self.max_neighbors is None:
            # Concatenate without sampling
            total = int(counts.sum().item())
            out = dst_s.new_empty(total)
            offset = 0
            for b in range(B):
                s, e = int(starts[b].item()), int(ends[b].item())
                if e > s:
                    out[offset : offset + (e - s)] = dst_s[s:e]
                    offset += e - s
            return out, counts

        # With cap: sample uniformly without replacement per row
        k = self.max_neighbors
        kept_counts = torch.clamp(counts, max=k)
        total = int(kept_counts.sum().item())
        out = dst_s.new_empty(total)
        offset = 0
        for b in range(B):
            s, e = int(starts[b].item()), int(ends[b].item())
            c = int(counts[b].item())
            kk = min(k, c)
            if kk > 0:
                if c == kk:
                    out[offset : offset + kk] = dst_s[s:e]
                else:
                    perm = torch.randperm(c, device=device)[:kk]
                    out[offset : offset + kk] = dst_s[s + perm]
                offset += kk
        return out, kept_counts

    def forward(
        self,
        H: torch.Tensor,  # [N, d_model] (centers use this)
        edge_index: torch.Tensor,  # [2, M]
        centers: torch.Tensor,  # [B]
        out_head: nn.Module,  # shared output head: d_model -> y_dim
        target: torch.Tensor,  # [N, y_dim]
        reduction: str = "mean_per_center",
        row_ptr: torch.Tensor | None = None,
        dst_sorted: torch.Tensor | None = None,
        H_neigh: (
            torch.Tensor | None
        ) = None,  # [N, d_neigh], raw neighbor inputs (node_encoder outputs)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        device = H.device
        N, d = H.shape
        centers = centers.to(device=device, dtype=torch.long)

        if centers.numel() == 0:
            zero = H.new_tensor(0.0)
            return zero, {
                "sp_mtp/centers": H.new_tensor(0),
                "sp_mtp/pairs": H.new_tensor(0),
            }

        # CSR-like adjacency: use cached if provided, else build once
        if (row_ptr is not None) and (dst_sorted is not None):
            row_ptr = row_ptr.to(device)
            dst_s = dst_sorted.to(device)
        else:
            e = _make_undirected(
                edge_index.to(device), assume_undirected=self.assume_undirected
            )
            _, dst_s, row_ptr = _sorted_by_src(e, num_nodes=N)

        # Gather neighbors for the chosen centers (optionally cap)
        dst_concat, counts = self._cap_neighbors(dst_s, row_ptr, centers)  # [T], [B]
        B = centers.numel()
        deg = counts  # [B]
        total_neighbors = int(deg.sum().item())
        if total_neighbors == 0:
            zero = H.new_tensor(0.0)
            return zero, {
                "sp_mtp/centers": torch.tensor(float(B), device=device),
                "sp_mtp/pairs": H.new_tensor(0),
            }

        # Build padded star index matrix without loops
        max_deg = int(deg.max().item())
        L = 1 + max_deg
        idx_mat = torch.full((B, L), -1, dtype=torch.long, device=device)
        idx_mat[:, 0] = centers

        pos = torch.arange(max_deg, device=device).unsqueeze(0)  # [1, max_deg]
        mask = pos < deg.unsqueeze(1)  # [B, max_deg]

        pref = torch.zeros_like(deg)
        pref[1:] = deg.cumsum(0)[:-1]
        idx_in_concat = pref.unsqueeze(1) + pos  # [B, max_deg]

        nbrs_full = torch.empty(B, max_deg, dtype=torch.long, device=device)
        nbrs_full[mask] = dst_concat[idx_in_concat[mask]]
        idx_mat[:, 1:][mask] = nbrs_full[mask]

        key_padding_mask = idx_mat.eq(-1)

        # Prepare inputs: centers from H, neighbors from H_neigh (if provided), both mapped to d_model
        safe_idx = idx_mat.clamp_min(0)  # replace -1 with 0 (won't be used)

        # centers
        X = torch.zeros(B, L, d, device=device, dtype=H.dtype)
        X[:, 0] = H[safe_idx[:, 0]]

        # neighbors
        if H_neigh is None:
            neigh_src = H
        else:
            neigh_src = H_neigh
        if max_deg > 0:
            X[:, 1:][mask] = neigh_src[nbrs_full[mask]]

        # apply layer norm and zero out paddings
        X = self.in_ln(X)
        X[key_padding_mask] = 0.0

        # Ring attention within each star = full attention within sequence
        Z = self.enc(X, key_padding_mask=key_padding_mask)  # [B, L, d]

        # Neighbor positions
        owners = torch.repeat_interleave(torch.arange(B, device=device), deg)  # [T]
        targets = idx_mat[:, 1:][mask]  # [T]
        Z_frontier = Z[:, 1:][mask]  # [T, d]

        y_hat = out_head(Z_frontier)  # [T, y_dim]
        y_true = target[targets]  # [T, y_dim]
        err = (y_hat - y_true).pow(2).mean(dim=-1)  # [T]

        if reduction == "mean":
            aux_loss = err.mean()
        else:
            loss_sum = torch.zeros(B, device=device, dtype=err.dtype).scatter_add_(
                0, owners, err
            )
            cnt = deg.to(err.dtype).clamp_min(1.0)
            aux_loss = (loss_sum / cnt).mean()

        stats = {
            "sp_mtp/centers": torch.tensor(float(B), device=device),
            "sp_mtp/pairs": torch.tensor(float(total_neighbors), device=device),
            "sp_mtp/mean_pair_loss": err.mean().detach(),
            "sp_mtp/max_deg": torch.tensor(float(max_deg), device=device),
        }
        return aux_loss, stats
