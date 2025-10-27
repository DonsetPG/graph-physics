import pytest
import torch

from graphphysics.kernels.triton_gather_pack_k8 import (
    HAS_TRITON,
    gather_pack_k8_triton,
)


def _manual_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    idx_long = idx.to(torch.long, device=x.device)
    gathered = x.index_select(0, idx_long.view(-1))
    gathered = gathered.view(idx.size(0), idx.size(1), x.size(1), x.size(2))
    return gathered.transpose(1, 2).contiguous()


@pytest.mark.parametrize("include_self", [True, False])
def test_gather_pack_matches_reference_cpu(include_self: bool):
    torch.manual_seed(0)
    N, H, Dh = 12, 4, 16
    x = torch.randn(N, H, Dh)
    idx = torch.randint(0, N, (5, 8), dtype=torch.int32)
    if include_self:
        idx[:, 0] = torch.arange(idx.size(0)) % N

    out = gather_pack_k8_triton(x, idx)
    ref = _manual_gather(x, idx)
    assert torch.allclose(out, ref)


@pytest.mark.skipif(not (HAS_TRITON and torch.cuda.is_available()), reason="CUDA Triton required")
def test_gather_pack_cuda_matches_reference():
    torch.manual_seed(1)
    device = torch.device("cuda")
    N, H, Dh = 64, 3, 32
    x = torch.randn(N, H, Dh, device=device, dtype=torch.float16)
    idx = torch.randint(0, N, (10, 8), dtype=torch.int32, device=device)

    out = gather_pack_k8_triton(x, idx)
    ref = _manual_gather(x.cpu(), idx.cpu()).to(device=device, dtype=x.dtype)
    assert torch.allclose(out.cpu().to(torch.float32), ref.cpu().to(torch.float32), atol=1e-3)
