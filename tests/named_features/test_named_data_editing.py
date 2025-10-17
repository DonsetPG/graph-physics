from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from named_features import NamedData, make_x_layout


def make_data() -> NamedData:
    layout = make_x_layout(["a", "b", "c"], {"a": 1, "b": 2, "c": 1})
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    return NamedData(x=x, x_layout=layout)


def test_rename():
    data = make_data()
    data.x_rename({"b": "beta"})
    assert "beta" in data.x_names()
    assert torch.allclose(data.x_sel("beta"), data.x_sel("beta"))


def test_reorder_out_of_place():
    data = make_data()
    clone = data.x_reorder(["c", "a", "b"], inplace=False)
    assert clone.x_names() == ["c", "a", "b"]
    assert torch.allclose(clone.x_sel("a"), data.x_sel("a"))
    assert torch.allclose(clone.x_sel("b"), data.x_sel("b"))


def test_drop():
    data = make_data()
    dropped = data.x_drop(["b"])
    expected = torch.cat([data.x_sel("a"), data.x_sel("c")], dim=-1)
    assert torch.allclose(dropped, expected)
