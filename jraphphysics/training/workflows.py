from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import jraph
from loguru import logger

from graphphysics.utils.nodetype import NodeType
from jraphphysics.training.parse_parameters import GraphPreprocessing
from jraphphysics.utils.loss import MultiLoss
from jraphphysics.utils.jax_graph import meshdata_to_graph


def _to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_numpy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_numpy(v) for v in value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def graph_from_dataset_item(item: Mapping[str, Any]) -> jraph.GraphsTuple:
    if isinstance(item, tuple):
        item = item[0]

    if hasattr(item, "x") and hasattr(item, "pos"):
        edge_index = getattr(item, "edge_index", None)
        if edge_index is None:
            raise ValueError("Torch graph item must provide 'edge_index' for conversion.")

        node_features = _to_numpy(item.x)
        if node_features.ndim == 1:
            node_features = node_features[:, None]

        node_pos = _to_numpy(item.pos)
        edge_index = _to_numpy(edge_index)
        senders = edge_index[0]
        receivers = edge_index[1]

        edge_attr = getattr(item, "edge_attr", None)
        edge_attr = _to_numpy(edge_attr) if edge_attr is not None else None

        target = getattr(item, "y", None)
        globals_dict: Dict[str, Any] = {}
        if target is not None:
            target = _to_numpy(target)
            if target.ndim == 1:
                target = target[:, None]
            globals_dict["target_features"] = jnp.asarray(target, dtype=jnp.float32)

        return jraph.GraphsTuple(
            nodes={
                "features": jnp.asarray(node_features, dtype=jnp.float32),
                "pos": jnp.asarray(node_pos, dtype=jnp.float32),
            },
            edges=(
                jnp.asarray(edge_attr, dtype=jnp.float32)
                if edge_attr is not None
                else None
            ),
            senders=jnp.asarray(senders, dtype=jnp.int32),
            receivers=jnp.asarray(receivers, dtype=jnp.int32),
            n_node=jnp.asarray([node_features.shape[0]], dtype=jnp.int32),
            n_edge=jnp.asarray([senders.shape[0]], dtype=jnp.int32),
            globals=globals_dict,
        )

    return meshdata_to_graph(
        points=_to_numpy(item["points"]),
        cells=_to_numpy(item["cells"]),
        point_data=_to_numpy(item.get("point_data")),
        time=item.get("time", 1.0),
        target=_to_numpy(item.get("target_data")),
    )


def _apply_preprocessing(
    graph: jraph.GraphsTuple,
    preprocessing: Optional[GraphPreprocessing],
    key: Optional[Any],
) -> Tuple[jraph.GraphsTuple, Optional[Any]]:
    if preprocessing is None:
        return graph, key
    transformed = preprocessing(graph, key)
    if isinstance(transformed, tuple) and len(transformed) == 2:
        return transformed
    return transformed, key


def mse_loss(simulator: Any, graph: jraph.GraphsTuple, is_training: bool = True) -> Any:
    network_output, target_delta_normalized, _ = simulator(graph, is_training=is_training)
    return jnp.mean((network_output - target_delta_normalized) ** 2)


def supervised_loss(
    simulator: Any,
    graph: jraph.GraphsTuple,
    loss_fn: Any | None = None,
    masks: list[NodeType] | None = None,
    gradient_method: str | None = None,
    is_training: bool = True,
) -> jnp.ndarray:
    network_output, target_delta_normalized, _ = simulator(graph, is_training=is_training)

    if loss_fn is None:
        return jnp.mean((network_output - target_delta_normalized) ** 2)

    node_type = graph.nodes["features"][:, simulator.node_type_index].astype(jnp.int32)
    masks = masks or [NodeType.NORMAL, NodeType.OUTFLOW]

    network_output_physical = simulator._build_outputs(graph, network_output)
    target_physical = simulator._build_outputs(graph, target_delta_normalized)

    if isinstance(loss_fn, MultiLoss):
        loss_value = loss_fn(
            graph=graph,
            target=target_delta_normalized,
            network_output=network_output,
            node_type=node_type,
            masks=masks,
            network_output_physical=network_output_physical,
            target_physical=target_physical,
            gradient_method=gradient_method,
        )
        return loss_value

    return loss_fn(
        graph=graph,
        target=target_delta_normalized,
        network_output=network_output,
        node_type=node_type,
        masks=masks,
        network_output_physical=network_output_physical,
        target_physical=target_physical,
        gradient_method=gradient_method,
    )


def evaluate(
    simulator: Any,
    dataset: Sequence[Mapping[str, Any]],
    preprocessing: Optional[GraphPreprocessing] = None,
    max_samples: Optional[int] = None,
    loss_fn: Any | None = None,
    masks: list[NodeType] | None = None,
    gradient_method: str | None = None,
    key: Optional[Any] = None,
) -> float:
    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    if limit == 0:
        return 0.0

    losses = []
    for idx in range(limit):
        graph = graph_from_dataset_item(dataset[idx])
        graph, key = _apply_preprocessing(graph, preprocessing, key)
        losses.append(
            float(
                supervised_loss(
                    simulator=simulator,
                    graph=graph,
                    loss_fn=loss_fn,
                    masks=masks,
                    gradient_method=gradient_method,
                    is_training=False,
                )
            )
        )
    return float(np.mean(losses))


def _slice_assign(features: Any, start: int, end: int, values: Any) -> Any:
    width = end - start
    values = np.asarray(values)[:, :width]

    if hasattr(features, "at"):
        return features.at[:, start:end].set(values)

    updated = np.array(features, copy=True)
    updated[:, start:end] = values
    return updated


def rollout(
    simulator: Any,
    dataset: Sequence[Mapping[str, Any]],
    preprocessing: Optional[GraphPreprocessing] = None,
    start_index: int = 0,
    num_steps: Optional[int] = None,
    autoregressive_feature_slice: Optional[Tuple[int, int]] = None,
    key: Optional[Any] = None,
) -> list[Any]:
    if start_index < 0:
        raise ValueError("start_index must be >= 0.")

    max_end = len(dataset)
    if num_steps is None:
        end_index = max_end
    else:
        end_index = min(max_end, start_index + max(num_steps, 0))

    predictions: list[Any] = []
    prev_prediction = None

    for idx in range(start_index, end_index):
        graph = graph_from_dataset_item(dataset[idx])

        if autoregressive_feature_slice is not None and prev_prediction is not None:
            start, end = autoregressive_feature_slice
            updated_features = _slice_assign(
                graph.nodes["features"], start, end, prev_prediction
            )
            nodes = dict(graph.nodes)
            nodes["features"] = updated_features
            graph = graph._replace(nodes=nodes)

        graph, key = _apply_preprocessing(graph, preprocessing, key)
        _, _, output = simulator(graph, is_training=False)
        if output is None:
            output = graph.nodes["features"][:, : simulator.output_size]
        predictions.append(output)
        prev_prediction = output

    return predictions


def save_rollout_predictions(path: str, predictions: Sequence[Any]) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = [np.asarray(p) for p in predictions]
    if len(arrays) == 0:
        payload = np.empty((0,), dtype=np.float32)
    elif all(a.shape == arrays[0].shape for a in arrays):
        payload = np.stack(arrays, axis=0)
    else:
        payload = np.array(arrays, dtype=object)

    np.savez_compressed(output_path, predictions=payload)
    return str(output_path)


def save_checkpoint(
    path: str,
    simulator: Any,
    epoch: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "simulator": simulator,
        "epoch": int(epoch),
        "metadata": metadata or {},
    }
    with checkpoint_path.open("wb") as fp:
        pickle.dump(payload, fp)
    return str(checkpoint_path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint_path = Path(path)
    with checkpoint_path.open("rb") as fp:
        return pickle.load(fp)


@dataclass
class SimpleTrainer:
    simulator: Any
    learning_rate: float = 1e-3
    warmup_steps: int = 0
    total_steps: Optional[int] = None
    loss_fn: Any | None = None
    loss_name: Any | None = None
    masks: list[NodeType] | None = None
    gradient_method: str | None = None
    logger: Any | None = None
    progress_log_interval: int = 50
    _optimizer: Optional[Any] = field(default=None, init=False)
    _nnx: Optional[Any] = field(default=None, init=False)
    _use_optimizer: bool = field(default=False, init=False)
    _global_step: int = field(default=0, init=False)
    _optimizer_error_logged: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        try:
            from flax import nnx
            import optax
        except Exception:
            return

        self._nnx = nnx
        learning_rate: Any = self.learning_rate
        if self.warmup_steps > 0:
            decay_steps = max(int(self.total_steps or 1), self.warmup_steps + 1)
            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.learning_rate,
                warmup_steps=self.warmup_steps,
                decay_steps=decay_steps,
                end_value=self.learning_rate * 1e-3,
            )
        tx = optax.adam(learning_rate)

        for kwargs in ({}, {"wrt": nnx.Param}):
            try:
                self._optimizer = nnx.Optimizer(self.simulator, tx, **kwargs)
                self._use_optimizer = True
                return
            except TypeError:
                continue
            except Exception:
                return

    def _log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self.logger is None or len(metrics) == 0:
            return

        payload = {k: float(v) for k, v in metrics.items()}
        try:
            if hasattr(self.logger, "log"):
                if step is None:
                    self.logger.log(payload)
                else:
                    self.logger.log(payload, step=step)
            elif callable(self.logger):
                if step is None:
                    self.logger(payload)
                else:
                    self.logger(payload, step=step)
        except Exception:
            # Logging should never break training.
            return

    def _single_loss_metric_name(self) -> str:
        if isinstance(self.loss_name, str):
            return self.loss_name
        if self.loss_name is None:
            return "loss"
        return str(self.loss_name)

    def _loss_mask_summary(self, graph: jraph.GraphsTuple) -> Tuple[int, int]:
        try:
            node_type = np.asarray(
                graph.nodes["features"][:, self.simulator.node_type_index]
            ).astype(np.int32)
            masks = self.masks or [NodeType.NORMAL, NodeType.OUTFLOW]
            selected = np.zeros_like(node_type, dtype=bool)
            for node_t in masks:
                selected |= node_type == int(node_t)
            return int(selected.sum()), int(node_type.shape[0])
        except Exception:
            return 0, 0

    def _raise_non_finite_loss(
        self,
        split: str,
        epoch_idx: int,
        step_idx: int,
        loss_value: float,
        graph: jraph.GraphsTuple,
    ) -> None:
        node_features = np.asarray(graph.nodes["features"])
        node_finite = np.isfinite(node_features)
        node_type = None
        if node_features.ndim == 2 and self.simulator.node_type_index < node_features.shape[1]:
            node_type = node_features[:, self.simulator.node_type_index]

        target = None
        if graph.globals is not None and isinstance(graph.globals, dict):
            target = graph.globals.get("target_features")
        target_finite_ratio = None
        if target is not None:
            target_np = np.asarray(target)
            target_finite_ratio = float(np.mean(np.isfinite(target_np)))

        active_nodes, total_nodes = self._loss_mask_summary(graph)
        msg = (
            f"Non-finite {split} loss at epoch={epoch_idx + 1}, step={step_idx + 1}, "
            f"global_step={self._global_step + 1}, loss={loss_value}. "
            f"finite(node_features)={float(np.mean(node_finite)):.6f}, "
            f"active_loss_nodes={active_nodes}/{total_nodes}"
        )
        if node_type is not None:
            uniq = np.unique(node_type.astype(np.int32))
            msg += f", unique_node_types={uniq.tolist()[:16]}"
        if target_finite_ratio is not None:
            msg += f", finite(target)={target_finite_ratio:.6f}"
        raise FloatingPointError(msg)

    def _compute_multiloss_components(
        self, graph: jraph.GraphsTuple
    ) -> Dict[str, float]:
        if not isinstance(self.loss_fn, MultiLoss):
            return {}

        network_output, target_delta_normalized, _ = self.simulator(graph, is_training=True)
        node_type = graph.nodes["features"][:, self.simulator.node_type_index].astype(jnp.int32)
        masks = self.masks or [NodeType.NORMAL, NodeType.OUTFLOW]

        network_output_physical = self.simulator._build_outputs(graph, network_output)
        target_physical = self.simulator._build_outputs(graph, target_delta_normalized)
        _, train_losses = self.loss_fn(
            graph=graph,
            target=target_delta_normalized,
            network_output=network_output,
            node_type=node_type,
            masks=masks,
            network_output_physical=network_output_physical,
            target_physical=target_physical,
            gradient_method=self.gradient_method,
            return_all_losses=True,
        )

        names = self.loss_name if isinstance(self.loss_name, (list, tuple)) else []
        metrics: Dict[str, float] = {}
        for idx, train_loss in enumerate(train_losses):
            label = names[idx] if idx < len(names) else f"loss_{idx}"
            metrics[f"train_loss - {label}"] = float(train_loss)
        return metrics

    def _compute_rollout_rmse_metrics(
        self,
        dataset: Sequence[Mapping[str, Any]],
        preprocessing: Optional[GraphPreprocessing],
        max_samples: Optional[int],
        key: Optional[Any],
    ) -> Dict[str, float]:
        limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        if limit == 0:
            return {}

        predictions = rollout(
            simulator=self.simulator,
            dataset=dataset,
            preprocessing=preprocessing,
            num_steps=limit,
            key=key,
        )
        if len(predictions) == 0:
            return {}

        squared_errors: list[np.ndarray] = []
        first_step_rmse = None

        for idx in range(len(predictions)):
            graph = graph_from_dataset_item(dataset[idx])
            graph, key = _apply_preprocessing(graph, preprocessing, key)

            target = None
            if graph.globals is not None and isinstance(graph.globals, dict):
                target = graph.globals.get("target_features")
            if target is None:
                target = graph.nodes["features"][:, : self.simulator.output_size]

            pred = np.asarray(predictions[idx], dtype=np.float32)
            tgt = np.asarray(target, dtype=np.float32)
            err = (pred - tgt) ** 2
            squared_errors.append(err)

            if idx == 0:
                first_step_rmse = float(np.sqrt(np.mean(err)))

        all_rollout_rmse = float(
            np.sqrt(np.mean(np.concatenate([e.reshape(-1) for e in squared_errors], axis=0)))
        )

        metrics = {"val_all_rollout_rmse": all_rollout_rmse}
        if first_step_rmse is not None:
            metrics["val_1step_rmse"] = first_step_rmse
        return metrics

    def train_step(self, graph: jraph.GraphsTuple) -> tuple[float, Dict[str, float]]:
        metrics: Dict[str, float] = {}
        if isinstance(self.loss_fn, MultiLoss):
            metrics.update(self._compute_multiloss_components(graph))

        if self._use_optimizer and self._nnx is not None and self._optimizer is not None:
            try:
                def _loss(model: Any) -> Any:
                    return supervised_loss(
                        simulator=model,
                        graph=graph,
                        loss_fn=self.loss_fn,
                        masks=self.masks,
                        gradient_method=self.gradient_method,
                        is_training=True,
                    )

                grad_fn = self._nnx.value_and_grad(_loss)
                loss, grads = grad_fn(self.simulator)
                self._optimizer.update(grads)
                loss_value = float(loss)
                if isinstance(self.loss_fn, MultiLoss):
                    metrics["train_multiloss"] = loss_value
                else:
                    metrics[f"train_{self._single_loss_metric_name()}"] = loss_value
                return loss_value, metrics
            except Exception as exc:
                self._use_optimizer = False
                if not self._optimizer_error_logged:
                    logger.warning(
                        "Disabling optimizer updates after value_and_grad failure: {}",
                        exc,
                    )
                    self._optimizer_error_logged = True

        loss_value = float(
            supervised_loss(
                simulator=self.simulator,
                graph=graph,
                loss_fn=self.loss_fn,
                masks=self.masks,
                gradient_method=self.gradient_method,
                is_training=True,
            )
        )
        if isinstance(self.loss_fn, MultiLoss):
            metrics["train_multiloss"] = loss_value
        else:
            metrics[f"train_{self._single_loss_metric_name()}"] = loss_value
        return loss_value, metrics

    def fit(
        self,
        dataset: Sequence[Mapping[str, Any]],
        num_epochs: int,
        preprocessing: Optional[GraphPreprocessing] = None,
        val_dataset: Optional[Sequence[Mapping[str, Any]]] = None,
        val_preprocessing: Optional[GraphPreprocessing] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        key: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if num_epochs < 1:
            raise ValueError("num_epochs must be >= 1.")

        train_limit = len(dataset) if max_train_samples is None else min(
            len(dataset), max_train_samples
        )
        if train_limit == 0:
            raise ValueError("Training dataset is empty.")

        history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_all_rollout_rmse": [],
            "val_1step_rmse": [],
        }

        val_limit = (
            len(val_dataset)
            if (val_dataset is not None and max_val_samples is None)
            else (
                min(len(val_dataset), max_val_samples)
                if val_dataset is not None
                else 0
            )
        )
        logger.info(
            "Starting JAX training: epochs={}, train_steps_per_epoch={}, val_steps={}",
            num_epochs,
            train_limit,
            val_limit,
        )

        for epoch_idx in range(num_epochs):
            epoch_start = time.time()
            logger.info("Epoch {}/{} started", epoch_idx + 1, num_epochs)
            epoch_losses = []
            epoch_metric_values: Dict[str, list[float]] = {}
            for idx in range(train_limit):
                graph = graph_from_dataset_item(dataset[idx])
                graph, key = _apply_preprocessing(graph, preprocessing, key)
                if idx == 0:
                    active_nodes, total_nodes = self._loss_mask_summary(graph)
                    if total_nodes > 0 and active_nodes == 0:
                        logger.warning(
                            "No nodes match default loss mask on first batch (active_loss_nodes=0/{}). "
                            "Check 'index.node_type_index' and node type values in your dataset.",
                            total_nodes,
                        )
                step_loss, step_metrics = self.train_step(graph)
                if not np.isfinite(step_loss):
                    self._raise_non_finite_loss(
                        split="train",
                        epoch_idx=epoch_idx,
                        step_idx=idx,
                        loss_value=step_loss,
                        graph=graph,
                    )
                epoch_losses.append(step_loss)
                for name, value in step_metrics.items():
                    epoch_metric_values.setdefault(name, []).append(float(value))
                self._global_step += 1
                self._log_metrics(step_metrics, step=self._global_step)
                if (
                    self.progress_log_interval > 0
                    and ((idx + 1) % self.progress_log_interval == 0 or idx + 1 == train_limit)
                ):
                    logger.info(
                        "Epoch {}/{} step {}/{} train_loss={:.6g}",
                        epoch_idx + 1,
                        num_epochs,
                        idx + 1,
                        train_limit,
                        step_loss,
                    )

            epoch_loss = float(np.mean(epoch_losses))
            history["train_loss"].append(epoch_loss)
            self._log_metrics({"train_loss": epoch_loss}, step=self._global_step)
            for name, values in epoch_metric_values.items():
                mean_value = float(np.mean(values))
                self._log_metrics({name: mean_value}, step=self._global_step)

            if val_dataset is not None:
                val_loss = evaluate(
                    simulator=self.simulator,
                    dataset=val_dataset,
                    preprocessing=val_preprocessing,
                    max_samples=max_val_samples,
                    loss_fn=self.loss_fn,
                    masks=self.masks,
                    gradient_method=self.gradient_method,
                    key=key,
                )
                if not np.isfinite(val_loss):
                    first_graph = graph_from_dataset_item(val_dataset[0])
                    first_graph, _ = _apply_preprocessing(first_graph, val_preprocessing, key)
                    self._raise_non_finite_loss(
                        split="val",
                        epoch_idx=epoch_idx,
                        step_idx=0,
                        loss_value=val_loss,
                        graph=first_graph,
                    )
                history["val_loss"].append(val_loss)
                self._log_metrics({"val_loss": val_loss}, step=self._global_step)

                val_rollout_metrics = self._compute_rollout_rmse_metrics(
                    dataset=val_dataset,
                    preprocessing=val_preprocessing,
                    max_samples=max_val_samples,
                    key=key,
                )
                for metric_name, metric_value in val_rollout_metrics.items():
                    history.setdefault(metric_name, []).append(metric_value)
                self._log_metrics(val_rollout_metrics, step=self._global_step)
                logger.info(
                    "Epoch {}/{} done in {:.2f}s | train_loss={:.6g} val_loss={:.6g}",
                    epoch_idx + 1,
                    num_epochs,
                    time.time() - epoch_start,
                    epoch_loss,
                    val_loss,
                )
            else:
                logger.info(
                    "Epoch {}/{} done in {:.2f}s | train_loss={:.6g}",
                    epoch_idx + 1,
                    num_epochs,
                    time.time() - epoch_start,
                    epoch_loss,
                )

        return history
