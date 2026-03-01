from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jraph
from flax import nnx

from graphphysics.utils.nodetype import NodeType
from jraphphysics.dataset.h5_dataset import H5Dataset
from jraphphysics.dataset.preprocessing import build_preprocessing
from jraphphysics.dataset.xdmf_dataset import XDMFDataset
from jraphphysics.models.layers import set_use_silu_activation
from jraphphysics.models.processors import (
    EncodeProcessDecode,
    EncodeTransformDecode,
    TransolverProcessor,
)
from jraphphysics.models.simulator import Simulator
from jraphphysics.utils.loss import L2Loss, LossType, MultiLoss


GraphPreprocessing = Callable[
    [jraph.GraphsTuple, Optional[Any]],
    Union[jraph.GraphsTuple, Tuple[jraph.GraphsTuple, Any]],
]
_UNSET = object()


def _default_rngs(seed: int = 0) -> nnx.Rngs:
    return nnx.Rngs(params=seed, dropout=seed + 1)


def get_preprocessing(
    param: Dict[str, Any],
    remove_noise: bool = False,
    use_edge_feature: bool = True,
    extra_node_features: Optional[Callable] = None,
    extra_edge_features: Optional[Callable] = None,
) -> GraphPreprocessing:
    preprocessing_params = param.get("transformations", {}).get("preprocessing", {})
    noise_scale = preprocessing_params.get("noise", 0)
    noise_parameters = None
    if noise_scale != 0 and not remove_noise:
        noise_parameters = {
            "noise_index_start": preprocessing_params.get("noise_index_start"),
            "noise_index_end": preprocessing_params.get("noise_index_end"),
            "noise_scale": noise_scale,
            "node_type_index": param["index"]["node_type_index"],
        }

    world_pos_parameters = None
    world_pos_params = param.get("transformations", {}).get("world_pos_parameters", {})
    if world_pos_params.get("use", False):
        world_pos_parameters = {
            "world_pos_index_start": world_pos_params.get("world_pos_index_start"),
            "world_pos_index_end": world_pos_params.get("world_pos_index_end"),
            "node_type_index": param["index"]["node_type_index"],
        }

    preprocessing = build_preprocessing(
        noise_parameters=noise_parameters,
        world_pos_parameters=world_pos_parameters,
        add_edges_features=use_edge_feature,
        extra_node_features=extra_node_features,
        extra_edge_features=extra_edge_features,
    )

    def _wrapped(
        graph: jraph.GraphsTuple,
        key: Any = _UNSET,
    ):
        key_provided = key is not _UNSET
        if key is _UNSET:
            key = None
        out_graph, out_key = preprocessing(graph, key=key)
        if key_provided:
            return out_graph, out_key
        return out_graph

    return _wrapped


def get_model(
    param: Dict[str, Any],
    rngs: Optional[nnx.Rngs] = None,
):
    model_type = param.get("model", {}).get("type", "")
    model_params = param.get("model", {})
    training_params = param.get("training", {})
    set_use_silu_activation(model_params.get("use_silu_activation", False))

    model_rngs = rngs or _default_rngs()
    base_kwargs = dict(
        message_passing_num=model_params["message_passing_num"],
        node_input_size=model_params["node_input_size"] + NodeType.SIZE,
        output_size=model_params["output_size"],
        hidden_size=model_params.get("hidden_size", 128),
        use_rope_embeddings=model_params.get("use_rope_embeddings", False),
        use_gated_attention=model_params.get("use_gated_attention", False),
        rope_pos_dimension=model_params.get("rope_pos_dimension", 3),
        rope_base=model_params.get("rope_base", 10000.0),
        use_temporal_block=training_params.get("use_temporal_block", False),
        rngs=model_rngs,
    )

    if model_type == "epd":
        return EncodeProcessDecode(
            edge_input_size=model_params.get("edge_input_size", 0),
            use_gated_mlp=model_params.get("use_gated_mlp", False),
            **base_kwargs,
        )
    if model_type == "transformer":
        return EncodeTransformDecode(
            num_heads=model_params.get("num_heads", 4),
            **base_kwargs,
        )
    if model_type == "transolver":
        return TransolverProcessor(
            num_heads=model_params.get("num_heads", 4),
            dropout=model_params.get("dropout", 0.0),
            mlp_ratio=model_params.get("mlp_ratio", 1),
            slice_num=model_params.get("slice_num", 32),
            ref=model_params.get("ref", 8),
            unified_pos=model_params.get("unified_pos", False),
            **base_kwargs,
        )
    raise ValueError(f"Model type '{model_type}' not supported.")


def get_simulator(
    param: Dict[str, Any],
    model: Any,
    rngs: Optional[nnx.Rngs] = None,
) -> Simulator:
    simulator_rngs = rngs or _default_rngs()

    return Simulator(
        node_input_size=param["model"]["node_input_size"] + NodeType.SIZE,
        edge_input_size=param["model"].get("edge_input_size", 0),
        output_size=param["model"]["output_size"],
        feature_index_start=param["index"]["feature_index_start"],
        feature_index_end=param["index"]["feature_index_end"],
        output_index_start=param["index"]["output_index_start"],
        output_index_end=param["index"]["output_index_end"],
        node_type_index=param["index"]["node_type_index"],
        model=model,
        rngs=simulator_rngs,
    )


def get_dataset(
    param: Dict[str, Any],
    preprocessing: Optional[Callable] = None,
    use_edge_feature: bool = True,
    use_previous_data: bool = False,
    switch_to_val: bool = False,
) -> Any:
    dataset_params = param.get("dataset", {})
    targets = dataset_params.get("targets", [])
    if len(targets) == 0:
        raise ValueError("Please provide a list of target properties to predict.")

    extension = dataset_params.get("extension", "")
    khop = dataset_params.get("khop", 1)

    if extension == "xdmf":
        return XDMFDataset(
            xdmf_folder=dataset_params["xdmf_folder"],
            meta_path=dataset_params["meta_path"],
            targets=targets,
            preprocessing=preprocessing,
            khop=khop,
            add_edge_features=use_edge_feature,
            use_previous_data=use_previous_data,
            switch_to_val=switch_to_val,
        )

    if extension == "h5":
        return H5Dataset(
            h5_path=dataset_params["h5_path"],
            meta_path=dataset_params["meta_path"],
            targets=targets,
            preprocessing=preprocessing,
            masking_ratio=None,
            khop=khop,
            add_edge_features=use_edge_feature,
            use_previous_data=use_previous_data,
            switch_to_val=switch_to_val,
        )

    raise ValueError(
        f"Dataset extension '{extension}' not supported in jraphphysics."
    )


def get_torch_preprocessing(
    param: Dict[str, Any],
    remove_noise: bool = False,
    use_edge_feature: bool = True,
) -> Callable:
    import torch
    from graphphysics.training.parse_parameters import (
        get_preprocessing as get_graphphysics_preprocessing,
    )

    return get_graphphysics_preprocessing(
        param=param,
        device=torch.device("cpu"),
        use_edge_feature=use_edge_feature,
        remove_noise=remove_noise,
    )


def get_num_workers(param: Dict[str, Any], default_num_workers: int) -> int:
    extension = param.get("dataset", {}).get("extension", "")
    if extension in {"xdmf", "h5"}:
        return default_num_workers
    raise ValueError(
        f"Dataset extension '{extension}' not supported in jraphphysics."
    )


def get_loss(param: Dict[str, Any], **kwargs):
    try:
        _ = param["loss"]
    except KeyError:
        return L2Loss(**kwargs), LossType.L2LOSS.name

    loss_types = param["loss"]["type"]
    if len(loss_types) > 1:
        losses = [LossType[t.upper()].value(**kwargs) for t in loss_types]
        weights = param["loss"]["weights"]
        names = [LossType[t.upper()].name for t in loss_types]
        return MultiLoss(losses=losses, weights=weights), names

    loss = LossType[loss_types[0].upper()]
    return loss.value(**kwargs), loss.name


def get_gradient_method(param: Dict[str, Any], **kwargs) -> Optional[str]:
    del kwargs
    try:
        return param["loss"]["gradient_method"]
    except KeyError:
        return None
