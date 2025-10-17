from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from named_features import NamedData, make_x_layout


def build_named_data() -> NamedData:
    layout = make_x_layout(["a", "b"], {"a": 3, "b": 2})
    x = torch.arange(10, dtype=torch.float32).reshape(1, 5).repeat(4, 1)
    return NamedData(x=x, x_layout=layout)


def test_single_selection_returns_view():
    data = build_named_data()
    view = data.x_sel("a")
    assert view.storage().data_ptr() == data.x.storage().data_ptr()
    view.zero_()
    assert torch.all(data.x[..., :3] == 0)


def test_multi_selection_concatenates():
    data = build_named_data()
    tensor = data.x_sel(["a", "b"])
    assert tensor.shape[-1] == 5
    assert torch.allclose(tensor, data.x)


def test_empty_selection_raises():
    data = build_named_data()
    try:
        data.x_sel([])
    except Exception as exc:  # pragma: no branch - expect FeatureNotFoundError
        assert "x_sel requires" in str(exc)
    else:  # pragma: no cover - sanity
        raise AssertionError("Expected exception not raised")
