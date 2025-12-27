"""gnn_ep.py

Entropy production (EP) estimation utilities for *trained* (stochastic) GNNs.

This module adapts the Maximum-Entropy / DV variational EP estimator implemented
in `ep_estimators.py` + `observables.py` to the GNN setting by:
  1) sampling stochastic layer-to-layer transitions (Gaussian noise each layer)
  2) building *antisymmetric* cross-correlation observables from projected states
  3) estimating EP per layer and summing to obtain a *global EP budget*.

We expose four "orders" of observables:
  - "node"  : per-node state = projected node features
  - "edge"  : per-edge state = concat(projected src, projected dst)
  - "1hop"  : per-node state = concat(node, mean_1hop_neighbors)
  - "2hop"  : per-node state = concat(node, mean_1hop_neighbors, mean_2hop_neighbors)

Each order yields a (lower bound) EP estimate using `observables.CrossCorrelations1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

import ep_estimators
import observables

Order = Literal["node", "edge", "1hop", "2hop"]


@dataclass
class EPEstimationConfig:
    # Sampling
    num_trajectories: int = 32
    max_nodes_per_traj: int = 1024
    max_edges_per_traj: int = 2048
    seed: int = 0

    # State compression
    proj_dim: int = 8  # project hidden features to this dim before building observables
    standardize: bool = True  # z-score X0/X1 coordinates before EP estimation

    # Observable family
    observable: Literal["antisym", "non_antisym"] = "antisym"

    # Estimator
    estimator: Literal["MaxEnt", "MTUR"] = "MaxEnt"
    optimizer: str = "Adam"  # used only when estimator == "MaxEnt"
    max_iter: int = 500
    verbose: int = 0
    clip_objective: bool = True  # cap objective at log(nsamples) (as in original method)
    # passed to optimizer constructor in optimizers.py
    optimizer_kwargs: Optional[dict] = None

    # Dataset split (to reduce overfitting of the variational bound)
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    # Storage: when None, keep samples on the same device used for estimation.
    # Set True to offload samples to CPU (lower GPU memory, slower).
    offload_to_cpu: Optional[bool] = None

    # Performance: reuse the same stochastic trajectories for all layers.
    # This is much faster but changes the exact sampling compared to per-layer trajectories.
    share_trajectories_across_layers: bool = True


def _row_normalized_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Return sparse row-normalized adjacency P where (P @ X)[v] = mean_{u in N(v)} X[u].

    Assumes `edge_index` uses PyG convention (src=edge_index[0], dst=edge_index[1]).
    We build P with indices (dst, src).
    """
    edge_index = edge_index.to(device=device)
    src, dst = edge_index[0], edge_index[1]
    # degree = in-degree w.r.t dst
    deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg = torch.clamp(deg, min=1.0)
    val = (1.0 / deg[dst]).to(torch.float32)
    P = torch.sparse_coo_tensor(
        torch.stack([dst, src], dim=0),
        val,
        (num_nodes, num_nodes),
        device=device,
        dtype=torch.float32,
    ).coalesce()
    return P


def _make_projection(in_dim: int, proj_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Gaussian random projection.

    If ``in_dim >= proj_dim``, we QR-orthonormalize columns for stability.
    If ``in_dim < proj_dim``, true column-orthonormality is impossible; we instead
    return a column-normalized Gaussian matrix to keep a consistent output shape
    ``(in_dim, proj_dim)`` across layers/states.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    W = torch.randn(in_dim, proj_dim, generator=g, device=device, dtype=torch.float32)
    if in_dim >= proj_dim:
        # Orthonormalize columns for stability (QR)
        Q, _ = torch.linalg.qr(W, mode="reduced")
        return Q
    # in_dim < proj_dim: keep exact proj_dim output by normalizing columns.
    return torch.nn.functional.normalize(W, p=2, dim=0)


def _build_projection_pairs(
    states: List[torch.Tensor],
    cfg: EPEstimationConfig,
    device: torch.device,
) -> List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """Build per-layer projection matrices for (h_t, h_{t+1}) pairs.

    If ``cfg.proj_dim <= 0`` we skip projection and return ``(None, None)`` for each layer.
    When consecutive states have the same dimensionality, we reuse the same projection
    matrix so X0/X1 live in the same coordinate system.
    """
    proj_dim = int(getattr(cfg, "proj_dim", 0) or 0)
    n_layers = len(states) - 1
    if proj_dim <= 0:
        return [(None, None) for _ in range(n_layers)]

    proj_pairs: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = []
    for layer_idx in range(n_layers):
        d0 = int(states[layer_idx].shape[1])
        d1 = int(states[layer_idx + 1].shape[1])
        seed_base = int(getattr(cfg, "seed", 0)) + 10_000 * layer_idx
        if d0 == d1:
            W = _make_projection(d0, proj_dim, seed=seed_base + 1, device=device)
            proj_pairs.append((W, W))
        else:
            W0 = _make_projection(d0, proj_dim, seed=seed_base + 1, device=device)
            W1 = _make_projection(d1, proj_dim, seed=seed_base + 2, device=device)
            proj_pairs.append((W0, W1))
    return proj_pairs


def _standardize_pair(X0: torch.Tensor, X1: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Z-score standardize coordinates using stats from concatenated (X0,X1)."""
    X = torch.cat([X0, X1], dim=0)
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    return (X0 - mean) / std, (X1 - mean) / std


def _concat(*tensors: torch.Tensor) -> torch.Tensor:
    return torch.cat(list(tensors), dim=-1)


_ORDER_ID: Dict[str, int] = {"node": 0, "edge": 1, "1hop": 2, "2hop": 3}


def _collect_X0_X1_for_layer(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    layer_idx: int,
    order: Order,
    proj_mats: List[torch.Tensor],
    P: torch.Tensor,
    cfg: EPEstimationConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect transition samples (X0,X1) for one layer and one order."""
    order_id = _ORDER_ID[str(order)]
    rng = np.random.default_rng(cfg.seed + 1000 * layer_idx + 10 * order_id)

    X0_list: List[torch.Tensor] = []
    X1_list: List[torch.Tensor] = []
    store_on_cpu = cfg.offload_to_cpu
    if store_on_cpu is None:
        store_on_cpu = device.type == "cpu"

    model_was_training = model.training
    model.eval()

    # Precompute edge sampling buffers once.
    if order == "edge":
        src_all, dst_all = edge_index[0], edge_index[1]
        mask = src_all != dst_all
        src_all = src_all[mask]
        dst_all = dst_all[mask]
        if src_all.numel() == 0:
            raise ValueError("No non-self-loop edges available for edge-order EP")

    with torch.inference_mode():
        for t in range(cfg.num_trajectories):
            # Change torch RNG each trajectory for independent Gaussian noise draws
            torch.manual_seed(cfg.seed + 10_000 * layer_idx + 17 * t)

            # We assume model.forward supports return_states=True and stochastic=True
            _, states = model(x, edge_index, return_states=True, stochastic=True)
            h0 = states[layer_idx]
            h1 = states[layer_idx + 1]

            # Use raw features directly (no random projection).
            z0 = h0
            z1 = h1

            # Sample *entities* once per trajectory and apply the same indices to
            # both z0 and z1 so samples are properly paired.
            if order in ("node", "1hop", "2hop"):
                n = int(z0.size(0))
                if cfg.max_nodes_per_traj is not None and cfg.max_nodes_per_traj < n:
                    idx = rng.choice(n, size=cfg.max_nodes_per_traj, replace=False)
                else:
                    idx = np.arange(n)
                idx_t = torch.as_tensor(idx, device=z0.device, dtype=torch.long)

                if order == "node":
                    s0 = z0[idx_t]
                    s1 = z1[idx_t]
                elif order == "1hop":
                    m10 = torch.sparse.mm(P, z0)
                    m11 = torch.sparse.mm(P, z1)
                    s0 = _concat(z0[idx_t], m10[idx_t])
                    s1 = _concat(z1[idx_t], m11[idx_t])
                else:  # "2hop"
                    m10 = torch.sparse.mm(P, z0)
                    m11 = torch.sparse.mm(P, z1)
                    m20 = torch.sparse.mm(P, m10)
                    m21 = torch.sparse.mm(P, m11)
                    s0 = _concat(z0[idx_t], m10[idx_t], m20[idx_t])
                    s1 = _concat(z1[idx_t], m11[idx_t], m21[idx_t])

            elif order == "edge":
                m = int(src_all.numel())
                if cfg.max_edges_per_traj is not None and cfg.max_edges_per_traj < m:
                    eidx = rng.choice(m, size=cfg.max_edges_per_traj, replace=False)
                else:
                    eidx = np.arange(m)
                eidx_t = torch.as_tensor(eidx, device=z0.device, dtype=torch.long)
                s0 = _concat(z0[src_all[eidx_t]], z0[dst_all[eidx_t]])
                s1 = _concat(z1[src_all[eidx_t]], z1[dst_all[eidx_t]])
            else:
                raise ValueError(f"Unknown order: {order}")

            if s0.shape != s1.shape:
                raise RuntimeError(f"Shape mismatch for order={order}: {s0.shape} vs {s1.shape}")

            if store_on_cpu:
                X0_list.append(s0.detach().cpu())
                X1_list.append(s1.detach().cpu())
            else:
                X0_list.append(s0.detach())
                X1_list.append(s1.detach())

    if model_was_training:
        model.train()

    X0 = torch.cat(X0_list, dim=0)
    X1 = torch.cat(X1_list, dim=0)
    if cfg.standardize:
        X0, X1 = _standardize_pair(X0, X1)
    return X0, X1


def _estimate_ep_from_X0_X1(X0: torch.Tensor, X1: torch.Tensor, cfg: EPEstimationConfig) -> float:
    """Run the EP estimator (MaxEnt or MTUR) on a transition sample set."""
    if getattr(cfg, "observable", "antisym") == "non_antisym":
        data = observables.CrossCorrelations2(X0=X0, X1=X1)
    else:
        data = observables.CrossCorrelations1(X0=X0, X1=X1)
    trn, val, tst = data.split_train_val_test(
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        shuffle=True,
    )

    if cfg.estimator == "MTUR":
        ep, _ = ep_estimators.get_EP_MTUR(trn)
        return float(ep)

    if cfg.estimator == "MaxEnt":
        ep, _ = ep_estimators.get_EP_Estimate(
            trn,
            validation=val,
            test=tst,
            verbose=cfg.verbose,
            max_iter=cfg.max_iter,
            optimizer=cfg.optimizer,
            optimizer_kwargs=cfg.optimizer_kwargs,
            clip_objective=cfg.clip_objective,
        )
        return float(ep)

    raise ValueError(f"Unknown estimator: {cfg.estimator}")


def _embedding_variance(H: torch.Tensor) -> torch.Tensor:
    """Compute 𝒱(H) = (1/(2n)) * Σ_v ||h_v - mean(H)||^2 as a scalar tensor."""
    if H.ndim != 2:
        raise ValueError(f"Expected H with shape (num_nodes, d), got {tuple(H.shape)}")
    mean = H.mean(dim=0, keepdim=True)
    centered = H - mean
    return 0.5 * centered.pow(2).sum(dim=1).mean()


def _estimate_expected_V_per_layer(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    n_layers: int,
    cfg: EPEstimationConfig,
) -> List[float]:
    """Estimate V(t) = E[𝒱(H_t)] for t=1..n_layers using stochastic trajectories."""
    if cfg.num_trajectories <= 0:
        raise ValueError("cfg.num_trajectories must be positive to estimate expected V per layer")
    if n_layers <= 0:
        return []

    V_sums = np.zeros((n_layers,), dtype=np.float64)

    model_was_training = model.training
    model.eval()

    with torch.inference_mode():
        for t in range(cfg.num_trajectories):
            torch.manual_seed(cfg.seed + 77_777 * t)
            _, states = model(x, edge_index, return_states=True, stochastic=True)
            if len(states) != n_layers + 1:
                raise RuntimeError(f"Unexpected number of states: got {len(states)} expected {n_layers + 1}")
            for li in range(n_layers):
                V_val = _embedding_variance(states[li + 1]).detach()
                V_sums[li] += float(V_val.to(device="cpu").item())

    if model_was_training:
        model.train()

    return (V_sums / float(cfg.num_trajectories)).tolist()


def estimate_gnn_ep_all_orders(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    cfg: Optional[EPEstimationConfig] = None,
    orders: Tuple[Order, ...] = ("node", "edge", "1hop", "2hop"),
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, object]]:
    """Estimate global EP budget for a trained stochastic GNN.

    Returns a nested dict:
      results[order] = {
         "per_layer": [ep_0, ep_1, ...],
         "total": sum(per_layer),
         "n_layers": L,
         "proj_dim": cfg.proj_dim,
      }
    """
    if cfg is None:
        cfg = EPEstimationConfig()
    if device is None:
        device = x.device

    x = x.to(device)
    edge_index = edge_index.to(device)

    # Basic sanity: EP estimation requires stochasticity.
    noise_std = getattr(model, "noise_std", None)
    if noise_std is not None and float(noise_std) <= 0.0:
        # Deterministic dynamics: EP is 0 by construction. Still return V stats.
        model_was_training = model.training
        model.eval()
        with torch.inference_mode():
            _, states = model(x, edge_index, return_states=True, stochastic=False)
        if model_was_training:
            model.train()

        n_states = len(states)
        n_layers = n_states - 1
        if n_layers <= 0:
            raise ValueError("Model did not return enough states to define transitions")

        V0 = float(_embedding_variance(x).detach().to(device="cpu").item())
        V_per_layer = [_embedding_variance(states[i + 1]).detach().to(device="cpu").item() for i in range(n_layers)]
        V_final = float(V_per_layer[-1]) if len(V_per_layer) else float("nan")

        zero_stats = {
            "per_layer": [0.0] * n_layers,
            "total": 0.0,
            "n_layers": n_layers,
            "proj_dim": cfg.proj_dim,
            "V0": V0,
            "V_per_layer": V_per_layer,
            "V_final": V_final,
            "num_trajectories": cfg.num_trajectories,
            "max_nodes_per_traj": cfg.max_nodes_per_traj,
            "max_edges_per_traj": cfg.max_edges_per_traj,
            "estimator": cfg.estimator,
        }
        return {str(order): dict(zero_stats) for order in orders}

    # Build row-normalized adjacency once (for 1hop/2hop orders)
    num_nodes = int(x.size(0))
    P = _row_normalized_adj(edge_index=edge_index, num_nodes=num_nodes, device=device)

    # Run a deterministic forward pass to get per-state dimensions (supports dim changes)
    model_was_training = model.training
    model.eval()
    with torch.inference_mode():
        _, states = model(x, edge_index, return_states=True, stochastic=False)
    if model_was_training:
        model.train()

    n_states = len(states)
    n_layers = n_states - 1
    if n_layers <= 0:
        raise ValueError("Model did not return enough states to define transitions")

    proj_pairs = _build_projection_pairs(states=states, cfg=cfg, device=device)

    # V(0): variance of raw graph inputs before any message passing.
    V0 = float(_embedding_variance(x).detach().to(device="cpu").item())

    orders = tuple(str(order) for order in orders)
    if not orders:
        return {}

    store_on_cpu = cfg.offload_to_cpu
    if store_on_cpu is None:
        store_on_cpu = device.type == "cpu"

    need_1hop = any(order in ("1hop", "2hop") for order in orders)
    need_2hop = "2hop" in orders
    need_edge = "edge" in orders

    if need_edge:
        src_all, dst_all = edge_index[0], edge_index[1]
        mask = src_all != dst_all
        src_all = src_all[mask]
        dst_all = dst_all[mask]
        if src_all.numel() == 0:
            raise ValueError("No non-self-loop edges available for edge-order EP")
    else:
        src_all = dst_all = None

    results: Dict[str, Dict[str, object]] = {}
    per_layer_stats: Dict[str, List[float]] = {order: [] for order in orders}

    # Collect samples for all orders, reusing each stochastic forward pass.
    model_was_training = model.training
    model.eval()

    if cfg.share_trajectories_across_layers:
        V_sums = np.zeros((n_layers,), dtype=np.float64)
        X0_lists: Dict[str, List[List[torch.Tensor]]] = {
            order: [[] for _ in range(n_layers)] for order in orders
        }
        X1_lists: Dict[str, List[List[torch.Tensor]]] = {
            order: [[] for _ in range(n_layers)] for order in orders
        }
        rngs = {
            (li, order): np.random.default_rng(cfg.seed + 1000 * li + 10 * _ORDER_ID[order])
            for li in range(n_layers)
            for order in orders
        }

        with torch.inference_mode():
            for t in range(cfg.num_trajectories):
                torch.manual_seed(cfg.seed + 17 * t)
                _, states = model(x, edge_index, return_states=True, stochastic=True)
                if len(states) != n_layers + 1:
                    raise RuntimeError(
                        f"Unexpected number of states: got {len(states)} expected {n_layers + 1}"
                    )
                for layer_idx in range(n_layers):
                    h0 = states[layer_idx]
                    h1 = states[layer_idx + 1]
                    proj0, proj1 = proj_pairs[layer_idx]
                    z0 = h0 if proj0 is None else h0 @ proj0
                    z1 = h1 if proj1 is None else h1 @ proj1

                    if need_1hop:
                        m10 = torch.sparse.mm(P, z0)
                        m11 = torch.sparse.mm(P, z1)
                        if need_2hop:
                            m20 = torch.sparse.mm(P, m10)
                            m21 = torch.sparse.mm(P, m11)

                    for order in orders:
                        rng = rngs[(layer_idx, order)]
                        if order in ("node", "1hop", "2hop"):
                            n = int(z0.size(0))
                            if cfg.max_nodes_per_traj is not None and cfg.max_nodes_per_traj < n:
                                idx = rng.choice(n, size=cfg.max_nodes_per_traj, replace=False)
                            else:
                                idx = np.arange(n)
                            idx_t = torch.as_tensor(idx, device=z0.device, dtype=torch.long)

                            if order == "node":
                                s0 = z0[idx_t]
                                s1 = z1[idx_t]
                            elif order == "1hop":
                                s0 = _concat(z0[idx_t], m10[idx_t])
                                s1 = _concat(z1[idx_t], m11[idx_t])
                            else:  # "2hop"
                                s0 = _concat(z0[idx_t], m10[idx_t], m20[idx_t])
                                s1 = _concat(z1[idx_t], m11[idx_t], m21[idx_t])
                        elif order == "edge":
                            m = int(src_all.numel())
                            if cfg.max_edges_per_traj is not None and cfg.max_edges_per_traj < m:
                                eidx = rng.choice(m, size=cfg.max_edges_per_traj, replace=False)
                            else:
                                eidx = np.arange(m)
                            eidx_t = torch.as_tensor(eidx, device=z0.device, dtype=torch.long)
                            s0 = _concat(z0[src_all[eidx_t]], z0[dst_all[eidx_t]])
                            s1 = _concat(z1[src_all[eidx_t]], z1[dst_all[eidx_t]])
                        else:
                            raise ValueError(f"Unknown order: {order}")

                        if s0.shape != s1.shape:
                            raise RuntimeError(
                                f"Shape mismatch for order={order}: {s0.shape} vs {s1.shape}"
                            )

                        if store_on_cpu:
                            X0_lists[order][layer_idx].append(s0.detach().cpu())
                            X1_lists[order][layer_idx].append(s1.detach().cpu())
                        else:
                            X0_lists[order][layer_idx].append(s0.detach())
                            X1_lists[order][layer_idx].append(s1.detach())

                    V_val = _embedding_variance(states[layer_idx + 1]).detach()
                    V_sums[layer_idx] += float(V_val.to(device="cpu").item())

        V_per_layer = (V_sums / float(cfg.num_trajectories)).tolist()
        V_final = float(V_per_layer[-1]) if len(V_per_layer) else float("nan")

        for order in orders:
            for layer_idx in range(n_layers):
                X0 = torch.cat(X0_lists[order][layer_idx], dim=0)
                X1 = torch.cat(X1_lists[order][layer_idx], dim=0)
                if cfg.standardize:
                    X0, X1 = _standardize_pair(X0, X1)
                ep = _estimate_ep_from_X0_X1(X0, X1, cfg)
                per_layer_stats[order].append(ep)

    else:
        # Estimate expected embedding variance per layer once (independent of observable order).
        V_per_layer = _estimate_expected_V_per_layer(
            model=model,
            x=x,
            edge_index=edge_index,
            n_layers=n_layers,
            cfg=cfg,
        )
        V_final = float(V_per_layer[-1]) if len(V_per_layer) else float("nan")

        with torch.inference_mode():
            for layer_idx in range(n_layers):
                rngs = {
                    order: np.random.default_rng(cfg.seed + 1000 * layer_idx + 10 * _ORDER_ID[order])
                    for order in orders
                }
                X0_lists: Dict[str, List[torch.Tensor]] = {order: [] for order in orders}
                X1_lists: Dict[str, List[torch.Tensor]] = {order: [] for order in orders}

                for t in range(cfg.num_trajectories):
                    torch.manual_seed(cfg.seed + 10_000 * layer_idx + 17 * t)
                    _, stoch_states = model(x, edge_index, return_states=True, stochastic=True)
                    h0 = stoch_states[layer_idx]
                    h1 = stoch_states[layer_idx + 1]
                    proj0, proj1 = proj_pairs[layer_idx]
                    z0 = h0 if proj0 is None else h0 @ proj0
                    z1 = h1 if proj1 is None else h1 @ proj1

                    if need_1hop:
                        m10 = torch.sparse.mm(P, z0)
                        m11 = torch.sparse.mm(P, z1)
                        if need_2hop:
                            m20 = torch.sparse.mm(P, m10)
                            m21 = torch.sparse.mm(P, m11)

                    for order in orders:
                        rng = rngs[order]
                        if order in ("node", "1hop", "2hop"):
                            n = int(z0.size(0))
                            if cfg.max_nodes_per_traj is not None and cfg.max_nodes_per_traj < n:
                                idx = rng.choice(n, size=cfg.max_nodes_per_traj, replace=False)
                            else:
                                idx = np.arange(n)
                            idx_t = torch.as_tensor(idx, device=z0.device, dtype=torch.long)

                            if order == "node":
                                s0 = z0[idx_t]
                                s1 = z1[idx_t]
                            elif order == "1hop":
                                s0 = _concat(z0[idx_t], m10[idx_t])
                                s1 = _concat(z1[idx_t], m11[idx_t])
                            else:  # "2hop"
                                s0 = _concat(z0[idx_t], m10[idx_t], m20[idx_t])
                                s1 = _concat(z1[idx_t], m11[idx_t], m21[idx_t])
                        elif order == "edge":
                            m = int(src_all.numel())
                            if cfg.max_edges_per_traj is not None and cfg.max_edges_per_traj < m:
                                eidx = rng.choice(m, size=cfg.max_edges_per_traj, replace=False)
                            else:
                                eidx = np.arange(m)
                            eidx_t = torch.as_tensor(eidx, device=z0.device, dtype=torch.long)
                            s0 = _concat(z0[src_all[eidx_t]], z0[dst_all[eidx_t]])
                            s1 = _concat(z1[src_all[eidx_t]], z1[dst_all[eidx_t]])
                        else:
                            raise ValueError(f"Unknown order: {order}")

                        if s0.shape != s1.shape:
                            raise RuntimeError(
                                f"Shape mismatch for order={order}: {s0.shape} vs {s1.shape}"
                            )

                        if store_on_cpu:
                            X0_lists[order].append(s0.detach().cpu())
                            X1_lists[order].append(s1.detach().cpu())
                        else:
                            X0_lists[order].append(s0.detach())
                            X1_lists[order].append(s1.detach())

                for order in orders:
                    X0 = torch.cat(X0_lists[order], dim=0)
                    X1 = torch.cat(X1_lists[order], dim=0)
                    if cfg.standardize:
                        X0, X1 = _standardize_pair(X0, X1)
                    ep = _estimate_ep_from_X0_X1(X0, X1, cfg)
                    per_layer_stats[order].append(ep)

    if model_was_training:
        model.train()

    for order in orders:
        per_layer = per_layer_stats[order]
        results[order] = {
            "per_layer": per_layer,
            "total": float(np.sum(per_layer)),
            "n_layers": n_layers,
            "proj_dim": cfg.proj_dim,
            "V0": V0,
            "V_per_layer": V_per_layer,
            "V_final": V_final,
            "num_trajectories": cfg.num_trajectories,
            "max_nodes_per_traj": cfg.max_nodes_per_traj,
            "max_edges_per_traj": cfg.max_edges_per_traj,
            "estimator": cfg.estimator,
        }

    return results
