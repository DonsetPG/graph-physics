import torch
from torch_geometric.data import Data

from graphphysics.models.local_flash_attn_k8 import LocalFlashK8
from graphphysics.models.processors import EncodeLocalFlashDecode
from graphphysics.models.utils_csr import build_fixed_fanout_k8, edge_index_to_csr


def _manual_pack(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    idx_long = idx.to(torch.long)
    gathered = x.index_select(0, idx_long.view(-1))
    return gathered.view(idx.size(0), idx.size(1), x.size(1), x.size(2)).transpose(1, 2)


def _manual_attention(x: torch.Tensor, idx: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    q = x.view(x.size(0), n_heads, head_dim)
    k = q
    v = q
    k_pack = _manual_pack(k, idx)
    v_pack = _manual_pack(v, idx)
    scale = head_dim**-0.5

    scores = torch.matmul(q.unsqueeze(2), k_pack.transpose(-1, -2)).squeeze(2)
    scores = scores * scale
    weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(weights.unsqueeze(2), v_pack).squeeze(2)
    return context.view(x.size(0), n_heads * head_dim)


def _build_ring_edge_index(num_nodes: int) -> torch.Tensor:
    src = torch.arange(num_nodes, dtype=torch.long)
    dst = (src + 1) % num_nodes
    edges = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edges


def test_local_flash_matches_manual_attention():
    num_nodes = 6
    d_model = 32
    n_heads = 4
    head_dim = d_model // n_heads

    edge_index = _build_ring_edge_index(num_nodes)
    rowptr, col = edge_index_to_csr(edge_index, num_nodes)
    idx = build_fixed_fanout_k8(rowptr, col)

    layer = LocalFlashK8(
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        dropout=0.0,
        use_triton=False,
        use_flash_attn=False,
    )

    # Force Q=K=V=X and output projection identity.
    layer.qkv.weight.data.zero_()
    for i in range(d_model):
        layer.qkv.weight.data[i, i] = 1.0
        layer.qkv.weight.data[d_model + i, i] = 1.0
        layer.qkv.weight.data[2 * d_model + i, i] = 1.0
    layer.out_proj.weight.data.copy_(torch.eye(d_model))
    layer.out_proj.bias.data.zero_()

    x = torch.randn(num_nodes, d_model)

    out = layer(x, idx_k8=idx)
    ref = _manual_attention(x, idx, n_heads, head_dim)
    assert torch.allclose(out, ref, atol=1e-6)


def test_local_flash_backward():
    num_nodes = 5
    d_model = 16
    n_heads = 4
    head_dim = d_model // n_heads
    edge_index = _build_ring_edge_index(num_nodes)
    rowptr, col = edge_index_to_csr(edge_index, num_nodes)
    idx = build_fixed_fanout_k8(rowptr, col)

    layer = LocalFlashK8(
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        dropout=0.0,
        use_triton=False,
        use_flash_attn=False,
    )

    x = torch.randn(num_nodes, d_model, requires_grad=True)
    out = layer(x, idx_k8=idx)
    loss = out.pow(2).sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_local_flash_builds_idx_from_edge_index_and_chunking():
    num_nodes = 7
    d_model = 24
    n_heads = 3
    head_dim = d_model // n_heads
    edge_index = _build_ring_edge_index(num_nodes)

    layer_full = LocalFlashK8(
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        dropout=0.0,
        use_triton=False,
        use_flash_attn=False,
    )

    layer_chunked = LocalFlashK8(
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        dropout=0.0,
        chunk_nodes=3,
        use_triton=False,
        use_flash_attn=False,
    )

    x = torch.randn(num_nodes, d_model)

    out_full = layer_full(x, edge_index=edge_index)
    out_chunked = layer_chunked(x, edge_index=edge_index)
    assert torch.allclose(out_full, out_chunked, atol=1e-5)


def test_encode_local_flash_decode_forward():
    num_nodes = 10
    hidden = 24
    heads = 3
    edge_index = _build_ring_edge_index(num_nodes)
    rowptr, col = edge_index_to_csr(edge_index, num_nodes)
    idx = build_fixed_fanout_k8(rowptr, col)

    model = EncodeLocalFlashDecode(
        message_passing_num=2,
        node_input_size=hidden,
        output_size=hidden,
        hidden_size=hidden,
        num_heads=heads,
        only_processor=True,
        use_triton=False,
        use_flash_attn=False,
    )

    graph = Data(x=torch.randn(num_nodes, hidden), edge_index=edge_index)
    graph.idx_k8 = idx
    graph.rowptr = rowptr
    graph.col = col

    out = model(graph)
    assert out.shape == (num_nodes, hidden)
