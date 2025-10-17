from __future__ import annotations

import pytest

from named_features import (
    FeatureNotFoundError,
    XFeatureLayout,
    make_x_layout,
    x_layout_from_meta_and_spec,
)


def test_layout_construction_and_introspection():
    layout = XFeatureLayout([("a", 3), ("b", 2), ("c", 1)])
    assert layout.names() == ["a", "b", "c"]
    assert layout.sizes() == {"a": 3, "b": 2, "c": 1}
    assert layout.slc("b") == slice(3, 5)
    assert layout.feature_dim() == 6


def test_layout_rename_and_reorder():
    layout = XFeatureLayout([("a", 3), ("b", 2), ("c", 1)])
    renamed = layout.rename({"b": "beta"})
    assert renamed.names() == ["a", "beta", "c"]
    reordered = renamed.reorder(["c", "a", "beta"])
    assert reordered.names() == ["c", "a", "beta"]
    assert reordered.slc("c") == slice(0, 1)
    assert reordered.slc("a") == slice(1, 4)
    assert reordered.slc("beta") == slice(4, 6)


def test_duplicate_name_raises():
    with pytest.raises(Exception):
        XFeatureLayout([("a", 1), ("a", 2)])


def test_invalid_size_raises():
    with pytest.raises(ValueError):
        XFeatureLayout([("a", 0)])


def test_reorder_missing_name():
    layout = XFeatureLayout([("a", 1), ("b", 2)])
    with pytest.raises(FeatureNotFoundError):
        layout.reorder(["a"])
    with pytest.raises(FeatureNotFoundError):
        layout.reorder(["a", "b", "c"])


def test_layout_serialisation_round_trip():
    layout = XFeatureLayout([("a", 2), ("b", 3)])
    restored = XFeatureLayout.from_dict(layout.to_dict(), order=["b", "a"])
    assert restored.names() == ["b", "a"]
    assert restored.sizes() == {"b": 3, "a": 2}


def test_layout_from_meta_and_overrides():
    meta = {
        "features": {
            "mesh_pos": {"shape": [10, 3]},
            "Vitesse": {"shape": [10, 3]},
            "wall_mask": {"shape": [10, 1]},
        }
    }
    layout = x_layout_from_meta_and_spec(
        meta,
        ["mesh_pos", "Vitesse", "wall_mask", "custom"],
        overrides={"custom": 7},
    )
    assert layout.sizes()["custom"] == 7
    with pytest.raises(FeatureNotFoundError):
        x_layout_from_meta_and_spec(meta, ["mesh_pos", "unknown"])
