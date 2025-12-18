"""graph_ep.py

Layer-wise entropy production (EP) + oversmoothing diagnostics for graph-physics models.

This is an adaptation of the previous GNN EP utilities to the `graphphysics` library:

- Works on *trained* models after `trainer.fit(...)`.
- Estimates **entropy production per message-passing / transformer layer**
  by sampling *stochastic* layer-to-layer transitions (Gaussian noise injected
  before each layer).
- Estimates **oversmoothing per layer** using the embedding variance functional

    V(H) = (1/(2n)) * sum_v ||h_v - mean(H)||^2

The code is model-agnostic as long as we can obtain a list of per-layer node
states. Preferred path is a model forward that supports:

    forward(graph, return_states=True, stochastic=True) -> (out, states)

If the model does not support `return_states`, we fall back to forward hooks and
try to infer the list of "layer" modules from common attributes (e.g.
`processor_list`).

This module only depends on torch + torch_geometric, and on the EP estimators
defined in this same subpackage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from . import ep_estimators, observables

Order = Literal["node", "edge", "1hop", "2hop"]


@dataclass
class EPEstimationConfig:
    # Sampling
    num_trajectories: int = 32
    max_nodes_per_traj: Optional[int] = 1024
    max_edges_per_traj: Optional[int] = 2048
    seed: int = 0

    # Noise injected *before each layer* to make layer-to-layer transitions stochastic.
    # If the model already exposes `noise_std`, we will temporarily set it to this value
    # during estimation. Otherwise, we will inject noise via forward-pre-hooks.
    noise_std: float = 1e-2

    # State compression
    proj_dim: int = 8  # project hidden features to this dim before building observables
    standardize: bool = True  # z-score X0/X1 coordinates before EP estimation

    # Estimator
    estimator: Literal["MaxEnt", "MTUR"] = "MaxEnt"
    optimizer: str = "Adam"  # used only when estimator == "MaxEnt"
    max_iter: int = 500
    verbose: int = 0
    optimizer_kwargs: Optional[dict] = None

    # Dataset split (to reduce overfitting of the variational bound)
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    # Hook fallback: where to find the list/ModuleList of per-layer blocks.
    # Examples: "processor_list", "model.blocks", "encoder.layers".
    layer_modules_attr: Optional[str] = None


def _row_normalized_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Return sparse row-normalized adjacency P where (P @ X)[v] = mean_{u in N(v)} X[u].

    Assumes `edge_index` uses PyG convention (src=edge_index[0], dst=edge_index[1]).
    We build P with indices (dst, src).
    """
    edge_index = edge_index.to(device=device)
    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    deg = torch.clamp(deg, min=1.0)
    val = (1.0 / deg[dst]).to(torch.float32)
    P = torch.sparse_coo_tensor(
        torch.stack([dst, src], dim=0),
        val,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=torch.float32,
    )
    return P.coalesce()


def _make_projection(in_dim: int, proj_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Random orthonormal projection matrix (in_dim, proj_dim)."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    Z = torch.randn(in_dim, proj_dim, generator=g, device=device, dtype=torch.float32)
    # QR gives orthonormal columns
    Q, _ = torch.linalg.qr(Z, mode="reduced")
    return Q[:, :proj_dim].contiguous()


def _standardize_pair(X0: torch.Tensor, X1: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Z-score standardize two paired sample matrices using the joint mean/std."""
    mean = 0.5 * (X0.mean(dim=0, keepdim=True) + X1.mean(dim=0, keepdim=True))
    var = 0.5 * ((X0 - mean).pow(2).mean(dim=0, keepdim=True) + (X1 - mean).pow(2).mean(dim=0, keepdim=True))
    std = torch.sqrt(var + eps)
    return (X0 - mean) / std, (X1 - mean) / std


def _concat(*tensors: torch.Tensor) -> torch.Tensor:
    return torch.cat(list(tensors), dim=-1)


def _as_node_features(H: torch.Tensor) -> torch.Tensor:
    """Coerce a captured state tensor into shape (num_nodes, d).

    - Graph processors usually yield (N, d).
    - Some transformer implementations (e.g. Transolver-like) yield (B, N, d).
      For diagnostics we support B==1 by squeezing the batch dim.
    """
    if H.ndim == 2:
        return H
    if H.ndim == 3 and H.size(0) == 1:
        return H.squeeze(0)
    raise ValueError(f"Expected a 2D (N,d) or 3D (1,N,d) tensor for a layer state, got shape {tuple(H.shape)}")


_ORDER_ID: Dict[str, int] = {"node": 0, "edge": 1, "1hop": 2, "2hop": 3}


def _get_attr_path(obj: object, path: str) -> object:
    cur: object = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(f"Object of type {type(cur).__name__} has no attribute '{part}' while resolving '{path}'")
        cur = getattr(cur, part)
    return cur


def _guess_layer_modules(model: nn.Module) -> List[nn.Module]:
    """Infer which submodules correspond to per-layer blocks.

    We try, in order:
      1) `model.processor_list` (graphphysics processors)
      2) Any direct ModuleList attribute on `model`
      3) Any nested ModuleList found via `named_modules()` (useful for wrappers like TransolverProcessor)

    If the heuristic is wrong for a given architecture, set `cfg.layer_modules_attr`
    to an explicit attribute path (e.g. "processor_list" or "model.blocks").
    """
    # (1) Common case in graphphysics
    if hasattr(model, "processor_list") and isinstance(getattr(model, "processor_list"), nn.ModuleList):
        return list(getattr(model, "processor_list"))

    # (2) Heuristic: pick the largest direct ModuleList attribute
    direct: List[Tuple[str, nn.ModuleList]] = []
    for name, value in vars(model).items():
        if isinstance(value, nn.ModuleList) and len(value) > 0:
            direct.append((name, value))
    if direct:
        direct.sort(key=lambda kv: len(kv[1]), reverse=True)
        return list(direct[0][1])

    # (3) Nested ModuleLists (best-effort)
    nested: List[Tuple[str, nn.ModuleList]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            nested.append((name, mod))

    if nested:
        def _score(item: Tuple[str, nn.ModuleList]) -> float:
            name, ml = item
            lname = name.lower()
            score = 0.0
            if lname == "processor_list" or lname.endswith(".processor_list") or lname.endswith("processor_list"):
                score += 1000.0
            if "processor" in lname:
                score += 200.0
            if "block" in lname:
                score += 150.0
            if "layer" in lname:
                score += 120.0
            if "encoder" in lname:
                score += 80.0
            if "transformer" in lname:
                score += 80.0
            # prefer longer lists
            score += 10.0 * float(len(ml))
            return score

        nested.sort(key=_score, reverse=True)
        return list(nested[0][1])

    raise ValueError(
        "Could not infer which submodules correspond to per-layer blocks. "
        "Either implement forward(..., return_states=True, ...) on the model, or "
        "set cfg.layer_modules_attr to an attribute path containing a ModuleList/list of blocks."
    )



@torch.no_grad()
def _collect_states_with_hooks(
    model: nn.Module,
    graph: Data,
    *,
    layers: List[nn.Module],
    stochastic: bool,
    noise_std: float,
) -> List[torch.Tensor]:
    """Fallback: collect per-layer states via forward hooks on a list of layer modules."""

    states: List[torch.Tensor] = []

    # --- Pre-hooks: inject noise into the first tensor argument (typically x) ---
    def _make_pre_hook(capture_input: bool):
        def _pre_hook(_module: nn.Module, inputs: Tuple[object, ...]):
            if not inputs:
                return None
            x0 = inputs[0]
            if isinstance(x0, torch.Tensor):
                if capture_input:
                    states.append(x0.detach())
                if stochastic and noise_std > 0.0:
                    x0 = x0 + torch.randn_like(x0) * float(noise_std)
                    return (x0,) + inputs[1:]
            return None
        return _pre_hook

    # --- Forward hook: capture output as state (first element if tuple) ---
    def _fwd_hook(_module: nn.Module, _inputs: Tuple[object, ...], output: object):
        y = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(y, torch.Tensor):
            states.append(y.detach())

    handles: List[torch.utils.hooks.RemovableHandle] = []
    try:
        for li, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(_make_pre_hook(capture_input=(li == 0))))
            handles.append(layer.register_forward_hook(_fwd_hook))

        _ = model(graph)  # output ignored; hooks populate `states`
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    if len(states) != len(layers) + 1:
        raise RuntimeError(
            f"Hook-based state capture failed: expected {len(layers)+1} states (including input), got {len(states)}. "
            "You may need to set cfg.layer_modules_attr to the correct layer list."
        )
    return states


@torch.no_grad()
def _get_states(
    model: nn.Module,
    graph: Data,
    *,
    cfg: EPEstimationConfig,
    stochastic: bool,
) -> List[torch.Tensor]:
    """Get per-layer node states.

    Preferred: call the model with return_states=True.
    Fallback: use forward hooks.
    """
    # Try native support
    try:
        out = model(graph, return_states=True, stochastic=stochastic)
        if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], (list, tuple)):
            states = list(out[1])
            return [_as_node_features(s) for s in states]
    except TypeError:
        pass

    # Hook fallback
    if cfg.layer_modules_attr is not None:
        layers_obj = _get_attr_path(model, cfg.layer_modules_attr)
        if isinstance(layers_obj, nn.ModuleList):
            layers = list(layers_obj)
        elif isinstance(layers_obj, (list, tuple)) and all(isinstance(m, nn.Module) for m in layers_obj):
            layers = list(layers_obj)
        else:
            raise ValueError(
                f"cfg.layer_modules_attr='{cfg.layer_modules_attr}' did not resolve to a ModuleList or list of Modules; got {type(layers_obj)}"
            )
    else:
        layers = _guess_layer_modules(model)

    states = _collect_states_with_hooks(
        model=model,
        graph=graph,
        layers=layers,
        stochastic=stochastic,
        noise_std=cfg.noise_std,
    )
    return [_as_node_features(s) for s in states]


def _collect_X0_X1_for_layer(
    model: nn.Module,
    graph: Data,
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

    x = graph.x
    edge_index = graph.edge_index

    # Precompute edge sampling buffers once.
    if order == "edge":
        src_all, dst_all = edge_index[0], edge_index[1]
        mask = src_all != dst_all
        src_all = src_all[mask]
        dst_all = dst_all[mask]
        if src_all.numel() == 0:
            raise ValueError("No non-self-loop edges available for edge-order EP")

    model_was_training = model.training
    model.eval()

    # Temporarily set model.noise_std if present (native stochastic forward path).
    old_noise_std: Optional[float] = None
    if hasattr(model, "noise_std"):
        try:
            old_noise_std = float(getattr(model, "noise_std"))
            setattr(model, "noise_std", float(cfg.noise_std))
        except Exception:
            old_noise_std = None

    try:
        for t in range(cfg.num_trajectories):
            torch.manual_seed(cfg.seed + 10_000 * layer_idx + 17 * t)

            states = _get_states(model, graph, cfg=cfg, stochastic=True)
            if layer_idx + 1 >= len(states):
                raise RuntimeError(f"Requested layer_idx={layer_idx} but only got {len(states)} states")

            h0 = states[layer_idx]
            h1 = states[layer_idx + 1]

            # project to cfg.proj_dim (layer-dependent in_dim allowed)
            z0 = h0.to(device=device) @ proj_mats[layer_idx]
            z1 = h1.to(device=device) @ proj_mats[layer_idx + 1]

            # Sample *entities* once per trajectory and apply the same indices to
            # both z0 and z1 so samples are properly paired.
            if order in ("node", "1hop", "2hop"):
                n = int(z0.size(0))
                if cfg.max_nodes_per_traj is not None and cfg.max_nodes_per_traj < n:
                    idx = rng.choice(n, size=int(cfg.max_nodes_per_traj), replace=False)
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
                    eidx = rng.choice(m, size=int(cfg.max_edges_per_traj), replace=False)
                else:
                    eidx = np.arange(m)
                eidx_t = torch.as_tensor(eidx, device=z0.device, dtype=torch.long)
                s0 = _concat(z0[src_all[eidx_t]], z0[dst_all[eidx_t]])
                s1 = _concat(z1[src_all[eidx_t]], z1[dst_all[eidx_t]])
            else:
                raise ValueError(f"Unknown order: {order}")

            if s0.shape != s1.shape:
                raise RuntimeError(f"Shape mismatch for order={order}: {s0.shape} vs {s1.shape}")

            X0_list.append(s0.detach().cpu())
            X1_list.append(s1.detach().cpu())
    finally:
        if old_noise_std is not None and hasattr(model, "noise_std"):
            try:
                setattr(model, "noise_std", old_noise_std)
            except Exception:
                pass
        if model_was_training:
            model.train()

    X0 = torch.cat(X0_list, dim=0)
    X1 = torch.cat(X1_list, dim=0)
    if cfg.standardize:
        X0, X1 = _standardize_pair(X0, X1)
    return X0, X1


def _estimate_ep_from_X0_X1(X0: torch.Tensor, X1: torch.Tensor, cfg: EPEstimationConfig) -> float:
    """Run the EP estimator (MaxEnt or MTUR) on a transition sample set."""
    data = observables.CrossCorrelations1(X0=X0, X1=X1)
    trn, val, tst = data.split_train_val_test(
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        shuffle=True,
    )

    if cfg.estimator == "MTUR":
        ep, _ = ep_estimators.get_EP_MTUR(trn)
        return float(ep)

    # MaxEnt (DV variational bound)
    ep, _ = ep_estimators.get_EP_Estimate(
        trn,
        validation=val,
        test=tst,
        optimizer=cfg.optimizer,
        max_iter=cfg.max_iter,
        verbose=cfg.verbose,
        optimizer_kwargs=cfg.optimizer_kwargs,
    )
    return float(ep)


def _embedding_variance(H: torch.Tensor) -> torch.Tensor:
    """Compute V(H) = (1/(2n)) * sum_v ||h_v - mean(H)||^2 as a scalar tensor."""
    H = _as_node_features(H)
    mean = H.mean(dim=0, keepdim=True)
    centered = H - mean
    return 0.5 * centered.pow(2).sum(dim=1).mean()


def _estimate_expected_V_per_layer(
    model: nn.Module,
    graph: Data,
    n_layers: int,
    cfg: EPEstimationConfig,
) -> List[float]:
    """Estimate V(t) = E[V(H_t)] for t=1..n_layers using stochastic trajectories."""
    if cfg.num_trajectories <= 0:
        raise ValueError("cfg.num_trajectories must be positive to estimate expected V per layer")
    if n_layers <= 0:
        return []

    V_sums = np.zeros((n_layers,), dtype=np.float64)

    model_was_training = model.training
    model.eval()

    old_noise_std: Optional[float] = None
    if hasattr(model, "noise_std"):
        try:
            old_noise_std = float(getattr(model, "noise_std"))
            setattr(model, "noise_std", float(cfg.noise_std))
        except Exception:
            old_noise_std = None

    try:
        for t in range(cfg.num_trajectories):
            torch.manual_seed(cfg.seed + 77_777 * t)
            states = _get_states(model, graph, cfg=cfg, stochastic=True)
            if len(states) != n_layers + 1:
                raise RuntimeError(f"Unexpected number of states: got {len(states)} expected {n_layers + 1}")
            for li in range(n_layers):
                V_val = _embedding_variance(states[li + 1]).detach()
                V_sums[li] += float(V_val.to(device="cpu").item())
    finally:
        if old_noise_std is not None and hasattr(model, "noise_std"):
            try:
                setattr(model, "noise_std", old_noise_std)
            except Exception:
                pass
        if model_was_training:
            model.train()

    return (V_sums / float(cfg.num_trajectories)).tolist()


def estimate_graph_ep_all_orders(
    model: nn.Module,
    graph: Union[Data, "torch_geometric.data.Batch"],
    cfg: Optional[EPEstimationConfig] = None,
    orders: Tuple[Order, ...] = ("node", "edge", "1hop", "2hop"),
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, object]]:
    """Estimate layerwise EP + oversmoothing for a trained (stochastic) graph model.

    Returns a nested dict:
      results[order] = {
         "per_layer": [ep_0, ep_1, ...],
         "total": sum(per_layer),
         "n_layers": L,
         "proj_dim": cfg.proj_dim,
         "V_input": V(graph.x),
         "V_per_layer": [E[V(H_1)], ..., E[V(H_L)]],
         "V_final": V_per_layer[-1],
         "noise_std": cfg.noise_std,
         ...
      }
    """
    if cfg is None:
        cfg = EPEstimationConfig()

    # Handle Batch by treating it as one big disconnected graph (PyG uses disjoint union).
    # For diagnostics you may want to pass a single-graph Data instead.
    if device is None:
        # Prefer model's device if it has parameters
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = graph.x.device  # type: ignore[assignment]

    graph = graph.to(device)

    x = graph.x
    edge_index = graph.edge_index

    # Compute deterministic states once to get number of layers + per-state dims
    model_was_training = model.training
    model.eval()

    old_noise_std: Optional[float] = None
    if hasattr(model, "noise_std"):
        try:
            old_noise_std = float(getattr(model, "noise_std"))
            setattr(model, "noise_std", 0.0)  # deterministic
        except Exception:
            old_noise_std = None

    try:
        states_det = _get_states(model, graph, cfg=cfg, stochastic=False)
    finally:
        if old_noise_std is not None and hasattr(model, "noise_std"):
            try:
                setattr(model, "noise_std", old_noise_std)
            except Exception:
                pass
        if model_was_training:
            model.train()

    n_states = len(states_det)
    n_layers = n_states - 1
    if n_layers <= 0:
        raise ValueError("Model did not return enough states to define transitions")

    # V_input: variance of normalized node inputs.
    V_input = float(_embedding_variance(x).detach().cpu().item())

    # Deterministic embedding variance per layer (no injected noise).
    V_det_per_layer = [
        float(_embedding_variance(states_det[li + 1]).detach().cpu().item()) for li in range(n_layers)
    ]
    V_det_final = float(V_det_per_layer[-1]) if len(V_det_per_layer) else float("nan")


    # Expected embedding variance per layer under stochastic trajectories.
    V_per_layer = _estimate_expected_V_per_layer(model=model, graph=graph, n_layers=n_layers, cfg=cfg)
    V_final = float(V_per_layer[-1]) if len(V_per_layer) else float("nan")

    # Projection matrices (one per state index)
    proj_mats: List[torch.Tensor] = []
    for i, h in enumerate(states_det):
        in_dim = int(h.size(-1))
        proj_mats.append(_make_projection(in_dim, cfg.proj_dim, seed=cfg.seed + 123 * i, device=device))

    P = _row_normalized_adj(edge_index=edge_index, num_nodes=int(x.size(0)), device=device)

    results: Dict[str, Dict[str, object]] = {}
    for order in orders:
        per_layer: List[float] = []
        for li in range(n_layers):
            X0, X1 = _collect_X0_X1_for_layer(
                model=model,
                graph=graph,
                layer_idx=li,
                order=order,
                proj_mats=proj_mats,
                P=P,
                cfg=cfg,
                device=device,
            )
            ep_li = _estimate_ep_from_X0_X1(X0, X1, cfg)
            per_layer.append(float(ep_li))

        results[str(order)] = {
            "per_layer": per_layer,
            "total": float(np.sum(per_layer)),
            "n_layers": n_layers,
            "proj_dim": cfg.proj_dim,
            "V_input": V_input,
            "V_per_layer": V_per_layer,
            "V_det_per_layer": V_det_per_layer,
            "V_det_final": V_det_final,
            "V_final": V_final,
            "num_trajectories": cfg.num_trajectories,
            "max_nodes_per_traj": cfg.max_nodes_per_traj,
            "max_edges_per_traj": cfg.max_edges_per_traj,
            "estimator": cfg.estimator,
            "optimizer": cfg.optimizer,
            "max_iter": cfg.max_iter,
            "noise_std": float(cfg.noise_std),
        }

    return results