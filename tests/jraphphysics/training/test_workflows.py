from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import jraph
import numpy as np

from jraphphysics.training.workflows import (
    SimpleTrainer,
    evaluate,
    graph_from_dataset_item,
    load_checkpoint,
    rollout,
    save_checkpoint,
    save_rollout_predictions,
)


@dataclass
class FakeGraph:
    nodes: dict

    def _replace(self, **kwargs):
        payload = {"nodes": self.nodes}
        payload.update(kwargs)
        return FakeGraph(**payload)


class DummySimulator:
    def __init__(self):
        self.output_size = 2
        self.inputs = []

    def __call__(self, graph, is_training=True):
        features = np.asarray(graph.nodes["features"], dtype=np.float32)
        self.inputs.append(features.copy())
        target = features[:, :2]
        network_output = target + 1.0
        if is_training:
            return network_output, target, None
        return network_output, target, network_output


class DummyLogger:
    def __init__(self):
        self.records = []

    def log(self, payload, step=None):
        self.records.append((step, dict(payload)))


@dataclass
class FakeTorchData:
    x: np.ndarray
    pos: np.ndarray
    edge_index: np.ndarray
    y: np.ndarray
    edge_attr: np.ndarray | None = None


@dataclass
class FakeGlobalsGraph:
    nodes: dict
    globals: dict | None = None

    def _replace(self, **kwargs):
        payload = {"nodes": self.nodes, "globals": self.globals}
        payload.update(kwargs)
        return FakeGlobalsGraph(**payload)


def test_save_and_load_checkpoint_round_trip(tmp_path):
    simulator = {"weights": [1, 2, 3]}
    ckpt_path = tmp_path / "ckpt.pkl"

    save_checkpoint(str(ckpt_path), simulator=simulator, epoch=4, metadata={"a": 1})
    payload = load_checkpoint(str(ckpt_path))

    assert payload["epoch"] == 4
    assert payload["metadata"]["a"] == 1
    assert payload["simulator"] == simulator


def test_graph_from_torch_item_conversion():
    item = FakeTorchData(
        x=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        edge_index=np.array([[0], [0]], dtype=np.int32),
        y=np.array([[0.5, 0.25]], dtype=np.float32),
        edge_attr=np.array([[1.0]], dtype=np.float32),
    )

    graph = graph_from_dataset_item(item)
    assert graph.nodes["features"].shape == (1, 3)
    assert graph.globals["target_features"].shape == (1, 2)


def test_rollout_autoregressive_updates_next_input():
    simulator = DummySimulator()
    graph1 = FakeGraph(nodes={"features": np.array([[1.0, 2.0, 9.0]], dtype=np.float32)})
    graph2 = FakeGraph(nodes={"features": np.array([[0.0, 0.0, 9.0]], dtype=np.float32)})

    dataset = [{}, {}]

    from unittest.mock import patch

    with patch(
        "jraphphysics.training.workflows.graph_from_dataset_item",
        side_effect=[graph1, graph2],
    ):
        predictions = rollout(
            simulator=simulator,
            dataset=dataset,
            autoregressive_feature_slice=(0, 2),
        )

    assert len(predictions) == 2
    assert np.allclose(simulator.inputs[1][:, :2], np.asarray(predictions[0]))


def test_evaluate_returns_mean_loss():
    simulator = DummySimulator()
    graph = FakeGraph(nodes={"features": np.array([[0.0, 0.0, 1.0]], dtype=np.float32)})
    dataset = [{}, {}]

    from unittest.mock import patch

    with patch(
        "jraphphysics.training.workflows.graph_from_dataset_item",
        side_effect=[graph, graph],
    ):
        loss = evaluate(simulator=simulator, dataset=dataset, max_samples=2)

    assert np.isclose(loss, 1.0)


def test_simple_trainer_fit_tracks_loss_history():
    simulator = DummySimulator()
    trainer = SimpleTrainer(simulator=simulator, learning_rate=1e-3)
    graph = FakeGraph(nodes={"features": np.array([[0.0, 0.0, 1.0]], dtype=np.float32)})
    dataset = [{}, {}]

    from unittest.mock import patch

    with patch(
        "jraphphysics.training.workflows.graph_from_dataset_item",
        side_effect=[graph, graph, graph, graph],
    ):
        history = trainer.fit(dataset=dataset, num_epochs=2, max_train_samples=2)

    assert len(history["train_loss"]) == 2


def test_save_rollout_predictions_writes_npz(tmp_path):
    output = tmp_path / "predictions.npz"
    predictions = [np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]

    save_rollout_predictions(str(output), predictions)
    loaded = np.load(Path(output), allow_pickle=True)

    assert "predictions" in loaded
    assert loaded["predictions"].shape == (2, 2, 2)


def test_simple_trainer_logs_graphphysics_metric_names():
    simulator = DummySimulator()
    logger = DummyLogger()
    trainer = SimpleTrainer(
        simulator=simulator,
        learning_rate=1e-3,
        loss_name="L2LOSS",
        logger=logger,
    )
    graph = FakeGlobalsGraph(
        nodes={"features": np.array([[0.0, 0.0, 1.0]], dtype=np.float32)},
        globals={"target_features": np.array([[0.0, 0.0]], dtype=np.float32)},
    )
    dataset = [{}, {}]

    from unittest.mock import patch

    with patch(
        "jraphphysics.training.workflows.graph_from_dataset_item",
        return_value=graph,
    ):
        history = trainer.fit(
            dataset=dataset,
            num_epochs=1,
            val_dataset=dataset,
            max_train_samples=1,
            max_val_samples=1,
        )

    all_keys = {
        key
        for _, payload in logger.records
        for key in payload.keys()
    }
    assert "train_L2LOSS" in all_keys
    assert "val_loss" in all_keys
    assert "val_all_rollout_rmse" in all_keys
    assert "val_1step_rmse" in all_keys
    assert len(history["val_all_rollout_rmse"]) == 1
    assert len(history["val_1step_rmse"]) == 1


def test_simple_trainer_optimizer_update_uses_model_and_grads_signature():
    simulator = DummySimulator()
    trainer = SimpleTrainer(simulator=simulator, learning_rate=1e-3)
    graph = FakeGraph(nodes={"features": np.array([[0.0, 0.0, 1.0]], dtype=np.float32)})

    class FakeNNX:
        @staticmethod
        def value_and_grad(fn):
            def _wrapped(model):
                _ = fn, model
                return np.array(1.0, dtype=np.float32), {"dummy": 1}

            return _wrapped

    class RecordingOptimizer:
        def __init__(self):
            self.calls = []

        def update(self, model, grads):
            self.calls.append((model, grads))

    trainer._nnx = FakeNNX()
    trainer._optimizer = RecordingOptimizer()
    trainer._use_optimizer = True

    _, metrics = trainer.train_step(graph)

    assert len(trainer._optimizer.calls) == 1
    assert trainer._optimizer.calls[0][0] is simulator
    assert "train_loss" in metrics


def test_simple_trainer_batches_multiple_graphs():
    simulator = DummySimulator()
    trainer = SimpleTrainer(simulator=simulator, learning_rate=1e-3)

    graph = jraph.GraphsTuple(
        nodes={
            "features": jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32),
            "pos": jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32),
        },
        edges=None,
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([0], dtype=jnp.int32),
        n_node=jnp.array([1], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
        globals={},
    )
    dataset = [{}, {}, {}]

    from unittest.mock import patch

    with patch(
        "jraphphysics.training.workflows.graph_from_dataset_item",
        side_effect=[graph, graph, graph],
    ):
        history = trainer.fit(dataset=dataset, num_epochs=1, batch_size=2, max_train_samples=3)

    assert len(history["train_loss"]) == 1
