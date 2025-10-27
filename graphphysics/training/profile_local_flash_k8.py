"""Profiler harness for LocalFlashK8."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from graphphysics.models.local_flash_attn_k8 import LocalFlashK8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile LocalFlashK8")
    parser.add_argument("--num-nodes", type=int, default=24576)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--trace", type=str, default="profile_local_flash_k8.json")
    return parser.parse_args()


def _dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[name]


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    dtype = _dtype_from_string(args.dtype)

    num_nodes = args.num_nodes
    d_model = args.d_model
    n_heads = args.n_heads
    head_dim = args.head_dim

    torch.manual_seed(42)
    x = torch.randn(num_nodes, d_model, device=device, dtype=dtype)
    idx = torch.randint(0, num_nodes, (num_nodes, 8), dtype=torch.int32, device=device)

    layer = LocalFlashK8(
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        dropout=0.0,
        chunk_nodes=None,
    ).to(device=device, dtype=dtype)
    layer.eval()

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_path = Path(args.trace)
    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        with torch.no_grad():
            total_steps = args.warmup + args.steps
            for step in range(total_steps):
                layer(x, idx_k8=idx)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                prof.step()

    prof.export_chrome_trace(str(trace_path))
    print(f"Trace written to {trace_path.resolve()}")
    if device.type == "cuda":
        sort_by = "self_cuda_time_total"
    else:
        sort_by = "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=20))


if __name__ == "__main__":  # pragma: no cover
    main()
