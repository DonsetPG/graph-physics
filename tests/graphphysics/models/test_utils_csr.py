from pathlib import Path

import torch

from graphphysics.models.utils_csr import build_fixed_fanout_k8, edge_index_to_csr
from tools.build_fixed_fanout_k8 import main as cli_main


def test_edge_index_to_csr_basic():
    edge_index = torch.tensor(
        [[0, 0, 1, 3], [1, 2, 2, 0]],
        dtype=torch.long,
    )
    rowptr, col = edge_index_to_csr(edge_index, num_nodes=4)

    assert rowptr.dtype == torch.int32
    assert col.dtype == torch.int32
    assert rowptr.tolist() == [0, 2, 3, 3, 4]
    assert col.tolist() == [1, 2, 2, 0]


def test_build_fixed_fanout_k8_padding_and_truncation():
    rowptr = torch.tensor([0, 3, 5, 6], dtype=torch.int32)
    col = torch.tensor([2, 1, 3, 0, 2, 1], dtype=torch.int32)

    idx_k8 = build_fixed_fanout_k8(rowptr, col, include_self=True, sort_neighbors=True)

    assert idx_k8.shape == (3, 8)
    assert idx_k8.dtype == torch.int32

    expected = torch.tensor(
        [
            [0, 1, 2, 3, 3, 3, 3, 3],
            [0, 1, 2, 2, 2, 2, 2, 2],
            [1, 2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(idx_k8, expected)


def test_build_fixed_fanout_k8_no_sort():
    rowptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    col = torch.tensor([2, 3, 1, 0], dtype=torch.int32)

    idx_k8 = build_fixed_fanout_k8(
        rowptr, col, include_self=False, sort_neighbors=False
    )
    assert idx_k8[0, :3].tolist() == [2, 3, 3]
    assert idx_k8[1, :3].tolist() == [1, 0, 0]


def test_cli_build_fixed_fanout_k8(tmp_path: Path):
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    edge_path = tmp_path / "edge_index.pt"
    torch.save(edge_index, edge_path)

    output = tmp_path / "idx.pt"

    cli_main(
        [
            "--edge-index",
            str(edge_path),
            "--output",
            str(output),
            "--save-csr",
        ]
    )

    idx = torch.load(output)
    assert idx.shape == (3, 8)

    rowptr_path = output.with_name(output.name + ".rowptr.pt")
    col_path = output.with_name(output.name + ".col.pt")
    assert rowptr_path.exists()
    assert col_path.exists()
