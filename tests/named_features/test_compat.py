from __future__ import annotations

from named_features import LegacyIndexAdapter, XFeatureLayout, old_indices_from_layout


def test_old_indices_from_layout():
    layout = XFeatureLayout([("a", 2), ("b", 3), ("type", 1)])
    indices = old_indices_from_layout(layout, ["b"], node_type_name="type")
    assert indices == {
        "feature_index_start": 0,
        "feature_index_end": 6,
        "output_index_start": 2,
        "output_index_end": 5,
        "node_type_index": 5,
    }


def test_legacy_index_adapter():
    layout = XFeatureLayout([("vel", 3), ("pressure", 1), ("type", 1)])
    adapter = LegacyIndexAdapter(layout, ["vel"], node_type_name="type")

    assert adapter.feature_window("vel") == (0, 3)
    assert adapter.as_dict() == {
        "feature_index_start": 0,
        "feature_index_end": 5,
        "output_index_start": 0,
        "output_index_end": 3,
        "node_type_index": 4,
    }

    mismatches = adapter.mismatches(
        {
            "feature_index_start": 0,
            "feature_index_end": 5,
            "output_index_start": 1,
            "output_index_end": 4,
            "node_type_index": 2,
        }
    )
    assert mismatches == {
        "output_index_start": (1, 0),
        "output_index_end": (4, 3),
        "node_type_index": (2, 4),
    }
