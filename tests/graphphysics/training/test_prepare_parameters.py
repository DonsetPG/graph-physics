import json
from copy import deepcopy

import pytest

pytest.importorskip("torch")

from graphphysics.training.parse_parameters import prepare_parameters


def test_prepare_parameters_semantic(tmp_path):
    meta_path = tmp_path / "meta.json"
    meta = {
        "features": {
            "mesh_pos": {"shape": [1, -1, 2]},
            "velocity": {"shape": [1, -1, 2]},
            "node_type": {"shape": [1, -1, 1]},
        }
    }
    meta_path.write_text(json.dumps(meta))

    parameters = {
        "dataset": {
            "extension": "h5",
            "meta_path": "meta.json",
            "targets": ["velocity"],
        },
        "model": {"node_input_size": 5},
        "features": {"node": ["mesh_pos", "velocity", "node_type"]},
        "sizes": {"node_type": 1},
        "targets": ["velocity"],
    }

    prepared = prepare_parameters(
        deepcopy(parameters), config_dir=str(tmp_path), named_features_mode="semantic"
    )
    layout = prepared["named_features"]["x_layout"]

    assert layout.names() == ["mesh_pos", "velocity", "node_type"]
    assert prepared["index"] == {
        "feature_index_start": 0,
        "feature_index_end": 4,
        "output_index_start": 2,
        "output_index_end": 4,
        "node_type_index": 4,
    }
    assert prepared["named_features"]["targets"] == ["velocity"]


def test_prepare_parameters_legacy_fallback():
    parameters = {
        "dataset": {"extension": "h5", "targets": ["velocity"]},
        "model": {"node_input_size": 3},
        "index": {
            "feature_index_start": 0,
            "feature_index_end": 3,
            "output_index_start": 1,
            "output_index_end": 3,
            "node_type_index": 0,
        },
    }

    prepared = prepare_parameters(deepcopy(parameters))
    named = prepared["named_features"]

    assert "x_layout" not in named or named["x_layout"] is None
    assert named["targets"] == ["velocity"]
    assert named.get("node_type") is None
    assert prepared["index"] == parameters["index"]


def test_prepare_parameters_warn_on_index_mismatch(tmp_path, caplog):
    meta_path = tmp_path / "meta.json"
    meta = {
        "features": {
            "mesh_pos": {"shape": [1, -1, 2]},
            "velocity": {"shape": [1, -1, 2]},
            "node_type": {"shape": [1, -1, 1]},
        }
    }
    meta_path.write_text(json.dumps(meta))

    parameters = {
        "dataset": {
            "extension": "h5",
            "meta_path": "meta.json",
            "targets": ["velocity"],
        },
        "model": {"node_input_size": 5},
        "features": {"node": ["mesh_pos", "velocity", "node_type"]},
        "targets": ["velocity"],
        "index": {
            "feature_index_start": 0,
            "feature_index_end": 5,
            "output_index_start": 1,
            "output_index_end": 5,
            "node_type_index": 0,
        },
    }

    with caplog.at_level("WARNING"):
        prepared = prepare_parameters(
            deepcopy(parameters), config_dir=str(tmp_path), named_features_mode="auto"
        )

    assert "Legacy index configuration disagrees" in caplog.text
    assert prepared["index"]["feature_index_end"] == 4
    assert prepared["named_features"]["legacy_indices"]["feature_index_end"] == 4
    assert prepared["index"]["output_index_start"] == 2
    assert prepared["named_features"]["legacy_indices"]["output_index_start"] == 2


def test_prepare_parameters_semantic_requires_config():
    parameters = {
        "dataset": {"extension": "h5"},
        "model": {"node_input_size": 3},
    }

    with pytest.raises(ValueError):
        prepare_parameters(deepcopy(parameters), named_features_mode="semantic")


def test_prepare_parameters_legacy_skips_warning(tmp_path, caplog):
    meta_path = tmp_path / "meta.json"
    meta = {
        "features": {
            "mesh_pos": {"shape": [1, -1, 2]},
            "velocity": {"shape": [1, -1, 2]},
            "node_type": {"shape": [1, -1, 1]},
        }
    }
    meta_path.write_text(json.dumps(meta))

    parameters = {
        "dataset": {
            "extension": "h5",
            "meta_path": "meta.json",
            "targets": ["velocity"],
        },
        "model": {"node_input_size": 5},
        "features": {"node": ["mesh_pos", "velocity", "node_type"]},
        "targets": ["velocity"],
        "index": {
            "feature_index_start": 0,
            "feature_index_end": 5,
            "output_index_start": 1,
            "output_index_end": 5,
            "node_type_index": 0,
        },
    }

    with caplog.at_level("WARNING"):
        prepare_parameters(
            deepcopy(parameters),
            config_dir=str(tmp_path),
            named_features_mode="legacy",
        )

    assert "Legacy index configuration disagrees" not in caplog.text
