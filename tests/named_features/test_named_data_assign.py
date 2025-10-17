from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from named_features import FeatureAssignmentError, NamedData, make_x_layout


def make_data() -> NamedData:
    layout = make_x_layout(["a", "b"], {"a": 2, "b": 1})
    x = torch.zeros(3, 3, dtype=torch.float32, requires_grad=True)
    return NamedData(x=x, x_layout=layout)


def test_assignment_inplace():
    data = make_data()
    value = torch.ones_like(data.x_sel("a"))
    data.x_assign({"a": value})
    assert torch.allclose(data.x_sel("a"), value)


def test_assignment_out_of_place():
    data = make_data()
    value = torch.full_like(data.x_sel("b"), 2.0)
    clone = data.x_assign({"b": value}, inplace=False)
    assert torch.allclose(clone.x_sel("b"), value)
    assert torch.allclose(data.x_sel("b"), torch.zeros_like(value))


def test_assignment_validation():
    data = make_data()
    with torch.no_grad():
        wrong_shape = torch.zeros(3, 1, 2)
    with pytest.raises(Exception):  # pragma: no cover - expecting error
        data.x_assign({"a": wrong_shape})


def test_assignment_errors_are_informative():
    data = make_data()
    with pytest.raises(FeatureAssignmentError):
        data.x_assign({"a": torch.zeros(3, 1)})
