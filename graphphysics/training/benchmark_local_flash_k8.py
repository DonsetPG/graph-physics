"""Microbenchmark for LocalFlashK8 attention layer."""
from __future__ import annotations

import argparse
import time

import torch

from graphphysics.models.local_flash_attn_k8 import LocalFlashK8


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LocalFlashK8")
    parser.add_argument("--num-nodes", type=int, default=32768)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--use-flash", action="store_true", help="Enable FlashAttention path (requires GPU)")
    return parser.parse_args()


def _get_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[name]


def _timed(module: LocalFlashK8, x: torch.Tensor, idx: torch.Tensor, repeat: int, warmup: int) -> float:
    device = x.device
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            module(x, idx_k8=idx)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        durations: list[float] = []
        for _ in range(repeat):
            start = time.perf_counter()
            module(x, idx_k8=idx)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
    return min(durations)


def main() -> None:
    args = _build_args()
    device = torch.device(args.device)
    dtype = _get_dtype(args.dtype)

    num_nodes = args.num_nodes
    d_model = args.d_model
    n_heads = args.n_heads
    head_dim = args.head_dim

    torch.manual_seed(0)
    x = torch.randn(num_nodes, d_model, device=device, dtype=dtype)
    idx = torch.randint(0, num_nodes, (num_nodes, 8), dtype=torch.int32, device=device)

    fast = LocalFlashK8(
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        dropout=0.0,
        chunk_nodes=None,
        use_triton=True,
        use_flash_attn=args.use_flash,
    ).to(device=device, dtype=dtype)

    baseline = LocalFlashK8(
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        dropout=0.0,
        chunk_nodes=None,
        use_triton=False,
        use_flash_attn=False,
    ).to(device=device, dtype=dtype)

    t_fast = _timed(fast, x, idx, args.repeat, args.warmup)
    t_base = _timed(baseline, x, idx, args.repeat, args.warmup)

    nodes_per_sec_fast = num_nodes / t_fast
    nodes_per_sec_base = num_nodes / t_base

    print("=== LocalFlashK8 Benchmark ===")
    print(f"device={device.type}, dtype={dtype}, nodes={num_nodes}, d_model={d_model}, heads={n_heads}")
    print(f"fast path (triton={fast.use_triton}, flash={fast.use_flash_attn}): {nodes_per_sec_fast:,.0f} nodes/s (time {t_fast:.4f}s)")
    print(f"baseline (triton={baseline.use_triton}): {nodes_per_sec_base:,.0f} nodes/s (time {t_base:.4f}s)")
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"peak memory: {peak_mem:.1f} MiB")


if __name__ == "__main__":  # pragma: no cover
    main()
