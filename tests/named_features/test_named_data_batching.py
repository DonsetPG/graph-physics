from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from torch_geometric.loader import DataLoader

from named_features import NamedData, make_x_layout


def make_batch_samples():
    layout = make_x_layout(["a", "b"], {"a": 1, "b": 2})
    samples = []
    for nodes in [3, 5]:
        x = torch.randn(nodes, 3)
        samples.append(
            NamedData(
                x=x, edge_index=torch.empty((2, 0), dtype=torch.long), x_layout=layout
            )
        )
    return samples


def test_batching_preserves_layout():
    loader = DataLoader(make_batch_samples(), batch_size=2)
    batch = next(iter(loader))
    assert isinstance(batch, NamedData)
    assert batch.x_layout.names() == ["a", "b"]
    assert batch.x_sel("a").shape[0] == sum(
        sample.num_nodes for sample in make_batch_samples()
    )
