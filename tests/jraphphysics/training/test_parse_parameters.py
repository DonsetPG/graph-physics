from unittest.mock import MagicMock, patch

import pytest

from graphphysics.utils.nodetype import NodeType
from jraphphysics.training.parse_parameters import (
    get_dataset,
    get_gradient_method,
    get_loss,
    get_model,
    get_preprocessing,
    get_simulator,
)


@pytest.fixture
def base_param():
    return {
        "transformations": {
            "preprocessing": {
                "noise": 0.1,
                "noise_index_start": [0],
                "noise_index_end": [2],
            }
        },
        "index": {
            "feature_index_start": 0,
            "feature_index_end": 3,
            "output_index_start": 0,
            "output_index_end": 2,
            "node_type_index": 3,
        },
        "model": {
            "type": "transformer",
            "message_passing_num": 2,
            "node_input_size": 5,
            "edge_input_size": 0,
            "output_size": 2,
            "hidden_size": 32,
            "num_heads": 4,
        },
        "dataset": {
            "extension": "xdmf",
            "xdmf_folder": "tests/mock_xdmf",
            "meta_path": "tests/mock_h5/meta10.json",
            "targets": ["velocity_x", "velocity_y"],
            "khop": 1,
        },
    }


def test_get_preprocessing_adds_noise(base_param):
    graph = MagicMock()
    key = MagicMock()

    preprocess_fn = MagicMock(return_value=("graph_with_noise", "next_key"))
    with patch("jraphphysics.training.parse_parameters.build_preprocessing") as mock_build:
        mock_build.return_value = preprocess_fn
        preprocessing = get_preprocessing(base_param)
        out_graph, out_key = preprocessing(graph, key)

    assert out_graph == "graph_with_noise"
    assert out_key == "next_key"
    preprocess_fn.assert_called_once_with(graph, key=key)
    called_kwargs = mock_build.call_args.kwargs
    assert called_kwargs["noise_parameters"] is not None


def test_get_preprocessing_remove_noise(base_param):
    graph = MagicMock()
    key = MagicMock()

    preprocess_fn = MagicMock(return_value=(graph, key))
    with patch("jraphphysics.training.parse_parameters.build_preprocessing") as mock_build:
        mock_build.return_value = preprocess_fn
        preprocessing = get_preprocessing(base_param, remove_noise=True)
        out_graph, out_key = preprocessing(graph, key)

    assert out_graph is graph
    assert out_key is key
    called_kwargs = mock_build.call_args.kwargs
    assert called_kwargs["noise_parameters"] is None


def test_get_model_transformer(base_param):
    with patch("jraphphysics.training.parse_parameters.EncodeTransformDecode") as mock_model:
        get_model(base_param, rngs=MagicMock())
        called = mock_model.call_args.kwargs

    assert called["node_input_size"] == base_param["model"]["node_input_size"] + NodeType.SIZE
    assert called["output_size"] == base_param["model"]["output_size"]


def test_get_model_epd(base_param):
    base_param["model"]["type"] = "epd"
    base_param["model"]["edge_input_size"] = 4
    with patch("jraphphysics.training.parse_parameters.EncodeProcessDecode") as mock_model:
        get_model(base_param, rngs=MagicMock())
        called = mock_model.call_args.kwargs

    assert called["edge_input_size"] == 4
    assert called["node_input_size"] == base_param["model"]["node_input_size"] + NodeType.SIZE


def test_get_model_transolver(base_param):
    base_param["model"]["type"] = "transolver"
    with patch("jraphphysics.training.parse_parameters.TransolverProcessor") as mock_model:
        get_model(base_param, rngs=MagicMock())
        called = mock_model.call_args.kwargs
    assert called["node_input_size"] == base_param["model"]["node_input_size"] + NodeType.SIZE


def test_get_model_invalid(base_param):
    base_param["model"]["type"] = "invalid"
    with pytest.raises(ValueError, match="not supported"):
        get_model(base_param, rngs=MagicMock())


def test_get_simulator(base_param):
    with patch("jraphphysics.training.parse_parameters.Simulator") as mock_simulator:
        model = MagicMock()
        get_simulator(base_param, model=model, rngs=MagicMock())
        called = mock_simulator.call_args.kwargs

    assert called["node_input_size"] == base_param["model"]["node_input_size"] + NodeType.SIZE
    assert called["model"] is model
    assert called["node_type_index"] == base_param["index"]["node_type_index"]


def test_get_dataset_xdmf(base_param):
    with patch("jraphphysics.training.parse_parameters.XDMFDataset") as mock_dataset:
        get_dataset(base_param, preprocessing=MagicMock(), switch_to_val=True)
        called = mock_dataset.call_args.kwargs

    assert called["xdmf_folder"] == base_param["dataset"]["xdmf_folder"]
    assert called["targets"] == base_param["dataset"]["targets"]
    assert called["switch_to_val"] is True


def test_get_dataset_h5(base_param):
    base_param["dataset"] = {
        "extension": "h5",
        "h5_path": "tests/mock_h5/write_mock.h5",
        "meta_path": "tests/mock_h5/meta.json",
        "targets": ["velocity"],
        "khop": 1,
    }
    with patch("jraphphysics.training.parse_parameters.H5Dataset") as mock_dataset:
        get_dataset(base_param, preprocessing=None, switch_to_val=True)
        called = mock_dataset.call_args.kwargs

    assert called["h5_path"] == base_param["dataset"]["h5_path"]
    assert called["targets"] == base_param["dataset"]["targets"]
    assert called["switch_to_val"] is True


def test_get_dataset_invalid_extension(base_param):
    base_param["dataset"]["extension"] = "invalid_ext"
    with pytest.raises(ValueError, match="not supported"):
        get_dataset(base_param, preprocessing=MagicMock())


def test_get_loss_default(base_param):
    loss, name = get_loss(base_param)
    assert loss is not None
    assert name == "L2LOSS"


def test_get_loss_multi(base_param):
    base_param["loss"] = {
        "type": ["l2loss", "divergencel2loss"],
        "weights": [0.5, 0.5],
        "gradient_method": "finite_diff",
    }
    loss, names = get_loss(base_param)
    assert loss is not None
    assert len(names) == 2


def test_get_gradient_method(base_param):
    base_param["loss"] = {"type": ["l2loss"], "gradient_method": "finite_diff"}
    assert get_gradient_method(base_param) == "finite_diff"
