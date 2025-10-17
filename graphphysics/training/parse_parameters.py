from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from loguru import logger
from torch_geometric.data import Data

from graphphysics.dataset.h5_dataset import H5Dataset
from graphphysics.dataset.preprocessing import build_preprocessing
from graphphysics.dataset.xdmf_dataset import XDMFDataset
from graphphysics.models.processors import (
    EncodeProcessDecode,
    EncodeTransformDecode,
    TransolverProcessor,
)
from graphphysics.models.simulator import Simulator
from graphphysics.utils.loss import LossType, MultiLoss
from graphphysics.utils.nodetype import NodeType
from named_features import (
    LegacyIndexAdapter,
    XFeatureLayout,
    x_layout_from_meta_and_spec,
)


def _resolve_path(path: str, config_dir: Optional[str]) -> str:
    if not path:
        return path
    if config_dir and not os.path.isabs(path):
        return os.path.normpath(os.path.join(config_dir, path))
    return path


def _load_meta(meta_path: Optional[str]) -> Mapping[str, Any]:
    if not meta_path:
        return {}
    with open(meta_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _normalise_targets(
    param: Dict[str, Any], dataset_cfg: Mapping[str, Any]
) -> List[str]:
    direct_targets = param.get("targets")
    if isinstance(direct_targets, Sequence) and not isinstance(
        direct_targets, (str, bytes)
    ):
        targets = [str(name) for name in direct_targets]
        if isinstance(dataset_cfg, dict):
            dataset_cfg.setdefault("targets", targets)
        param["targets"] = targets
        return targets

    dataset_targets = (
        dataset_cfg.get("targets") if isinstance(dataset_cfg, Mapping) else None
    )
    if isinstance(dataset_targets, Sequence) and not isinstance(
        dataset_targets, (str, bytes)
    ):
        targets = [str(name) for name in dataset_targets]
        param["targets"] = targets
        return targets

    return []


_VALID_MODES = {"auto", "semantic", "legacy"}


def _resolve_mode(named_section: Dict[str, Any], mode_override: Optional[str]) -> str:
    if mode_override:
        mode = mode_override.lower()
        named_section["mode"] = mode
    else:
        mode = str(named_section.get("mode", "auto")).lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unsupported named feature mode '{mode}'. Expected one of {_VALID_MODES}."
        )
    return mode


def prepare_parameters(
    param: Dict[str, Any],
    config_dir: Optional[str] = None,
    *,
    named_features_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Augment raw parameters with named feature layouts and legacy indices.

    Parameters
    ----------
    param:
        Configuration dictionary loaded from JSON.
    config_dir:
        Optional directory of the configuration file; used to resolve relative
        metadata paths.
    named_features_mode:
        Optional override for the named feature preparation mode. When provided,
        this value is normalised to ``{"auto", "semantic", "legacy"}`` and stored
        under ``param["named_features"]["mode"]``.
    """

    dataset_cfg = param.get("dataset", {})
    features_cfg = param.get("features")
    sizes_cfg = param.get("sizes")
    named_section = param.get("named_features")
    if not isinstance(named_section, dict):
        named_section = {}
        param["named_features"] = named_section

    layout: Optional[XFeatureLayout] = None

    mode = _resolve_mode(named_section, named_features_mode)

    if mode == "semantic" and not (
        isinstance(features_cfg, Mapping) and "node" in features_cfg
    ):
        raise ValueError(
            "Named feature mode 'semantic' requires a 'features[\"node\"]' configuration."
        )

    if isinstance(features_cfg, Mapping) and "node" in features_cfg:
        node_order = features_cfg["node"]
        if (
            not isinstance(node_order, Sequence)
            or isinstance(node_order, (str, bytes))
            or not node_order
        ):
            raise ValueError(
                "Named feature configuration requires 'features[\"node\"]' to be a non-empty sequence of strings."
            )
        node_order = [str(name) for name in node_order]

        overrides: Dict[str, int] = {}
        if isinstance(sizes_cfg, Mapping):
            overrides = {str(name): int(size) for name, size in sizes_cfg.items()}

        meta_path = _resolve_path(dataset_cfg.get("meta_path"), config_dir)
        meta = _load_meta(meta_path) if meta_path else {}

        layout = x_layout_from_meta_and_spec(meta, node_order, overrides)
        named_section["x_layout"] = layout
        named_section["node_features"] = node_order
        named_section["sizes"] = layout.sizes()

        node_type_name = None
        node_type_cfg = features_cfg.get("node_type")
        if isinstance(node_type_cfg, str):
            node_type_name = node_type_cfg
        elif "node_type" in node_order:
            node_type_name = "node_type"
        if node_type_name is not None and node_type_name not in layout.sizes():
            logger.warning(
                "Configured node type feature '%s' not present in layout; ignoring for compatibility indices.",
                node_type_name,
            )
            node_type_name = None
        named_section["node_type"] = node_type_name

        targets = _normalise_targets(param, dataset_cfg)
        if not targets:
            raise ValueError(
                "Named feature configuration requires 'targets' (either top-level or dataset.targets)."
            )
        named_section["targets"] = targets

        node_type_for_indices = node_type_name if node_type_name else None
        if node_type_for_indices and node_type_for_indices not in layout.sizes():
            node_type_for_indices = None

        adapter = LegacyIndexAdapter(
            layout,
            targets,
            node_type_name=node_type_for_indices,
        )
        named_section["legacy_adapter"] = adapter
        derived_indices = adapter.as_dict()

        legacy_index_cfg = param.get("index")
        if isinstance(legacy_index_cfg, Mapping) and mode != "legacy":
            mismatches = adapter.mismatches(legacy_index_cfg)
            if mismatches:
                mismatch_strings = ", ".join(
                    f"{key}: configured={configured}, derived={derived}"
                    for key, (configured, derived) in sorted(mismatches.items())
                )
                warning_msg = (
                    "Legacy index configuration disagrees with derived layout values "
                    f"({mismatch_strings}); using derived indices."
                )
                logger.warning(warning_msg)
                logging.getLogger(__name__).warning(warning_msg)
        param["index"] = derived_indices
        named_section["legacy_indices"] = derived_indices
        return param

    index_cfg = param.get("index")
    if isinstance(index_cfg, Mapping):
        targets = _normalise_targets(param, dataset_cfg)
        if targets:
            named_section.setdefault("targets", targets)

    return param


def get_preprocessing(
    param: Dict[str, Any],
    device: torch.device,
    use_edge_feature: bool = True,
    remove_noise: bool = False,
    extra_node_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
    extra_edge_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None,
):
    """
    Constructs the preprocessing function based on provided parameters.

    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.
        device (torch.device): The device to perform computations on.
        use_edge_feature (bool, optional): Whether to add edge features. Defaults to True.
        extra_node_features (Optional[Union[Callable[[Data], Data], List[Callable[[Data], Data]]]], optional):
            Additional functions to compute extra node features. Defaults to None.
        extra_edge_features (Optional[Union[Callable[[Data], Data], List[Callable[[Data], Data]]]], optional):
            Additional functions to compute extra edge features. Defaults to None.

    Returns:
        Callable[[Data], Data]: A function that preprocesses a Data object.
    """
    preprocessing_params = param.get("transformations", {}).get("preprocessing", {})
    noise_scale = preprocessing_params.get("noise", 0)
    noise_parameters = None

    named_section = param.get("named_features", {})
    layout: Optional[XFeatureLayout] = named_section.get("x_layout")

    if not remove_noise:
        if "noise_features" in preprocessing_params:
            noise_parameters = {
                "noise_features": preprocessing_params["noise_features"],
                "noise_scale": preprocessing_params.get("noise_scale", noise_scale),
                "node_type_feature": preprocessing_params.get(
                    "node_type_feature", named_section.get("node_type")
                ),
            }
        elif noise_scale != 0:
            noise_parameters = {
                "noise_index_start": preprocessing_params.get("noise_index_start"),
                "noise_index_end": preprocessing_params.get("noise_index_end"),
                "noise_scale": noise_scale,
                "node_type_index": param["index"]["node_type_index"],
            }

    world_pos_params = param.get("transformations", {}).get("world_pos_parameters", {})
    world_pos_parameters = None
    if world_pos_params.get("use", False):
        if (
            "world_pos_feature" in world_pos_params
            or "displacement_feature" in world_pos_params
        ):
            node_type_feature = world_pos_params.get(
                "node_type_feature", named_section.get("node_type")
            )
            world_pos_parameters = {
                "use": True,
                "world_pos_feature": world_pos_params.get("world_pos_feature"),
                "target_feature": world_pos_params.get("target_feature"),
                "displacement_feature": world_pos_params.get("displacement_feature"),
                "node_type_feature": node_type_feature,
                "radius": world_pos_params.get("radius", 0.03),
            }
        else:
            world_pos_parameters = {
                "use": True,
                "world_pos_index_start": world_pos_params.get("world_pos_index_start"),
                "world_pos_index_end": world_pos_params.get("world_pos_index_end"),
                "node_type_index": param["index"]["node_type_index"],
                "radius": world_pos_params.get("radius", 0.03),
            }

    return build_preprocessing(
        noise_parameters=noise_parameters,
        world_pos_parameters=world_pos_parameters,
        add_edges_features=use_edge_feature,
        extra_node_features=extra_node_features,
        extra_edge_features=extra_edge_features,
        x_layout=layout,
    )


def get_model(param: Dict[str, Any], only_processor: bool = False):
    """
    Constructs the model based on provided parameters.

    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.
        only_processor (bool, optional): Whether to use only the processor part of the model. Defaults to False.

    Returns:
        nn.Module: The constructed model.

    Raises:
        ValueError: If the model type specified in param is not supported.
    """
    model_type = param.get("model", {}).get("type", "")
    node_input_size = param["model"]["node_input_size"] + NodeType.SIZE

    if model_type == "epd":
        return EncodeProcessDecode(
            message_passing_num=param["model"]["message_passing_num"],
            node_input_size=node_input_size,
            edge_input_size=param["model"]["edge_input_size"],
            output_size=param["model"]["output_size"],
            hidden_size=param["model"]["hidden_size"],
            only_processor=only_processor,
        )
    elif model_type == "transformer":
        return EncodeTransformDecode(
            message_passing_num=param["model"]["message_passing_num"],
            node_input_size=node_input_size,
            output_size=param["model"]["output_size"],
            hidden_size=param["model"]["hidden_size"],
            num_heads=param["model"]["num_heads"],
            only_processor=only_processor,
        )
    elif model_type == "transolver":
        return TransolverProcessor(
            message_passing_num=param["model"]["message_passing_num"],
            node_input_size=node_input_size,
            output_size=param["model"]["output_size"],
            hidden_size=param["model"]["hidden_size"],
            num_heads=param["model"]["num_heads"],
        )
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")


def get_simulator(param: Dict[str, Any], model, device: torch.device) -> Simulator:
    """
    Constructs the Simulator based on provided parameters.

    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.
        model: The model to be used within the simulator.
        device (torch.device): The device to perform computations on.

    Returns:
        Simulator: The constructed Simulator object.
    """
    node_input_size = param["model"]["node_input_size"] + NodeType.SIZE

    named_section = param.get("named_features", {})
    layout: Optional[XFeatureLayout] = named_section.get("x_layout")
    node_type_name = named_section.get("node_type")
    targets = (
        named_section.get("targets") if isinstance(named_section, Mapping) else None
    )

    feature_start = param["index"]["feature_index_start"]
    feature_end = param["index"]["feature_index_end"]
    output_start = param["index"]["output_index_start"]
    output_end = param["index"]["output_index_end"]
    node_type_index = param["index"].get("node_type_index")

    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None

    if layout is not None:
        feature_names = [
            name
            for name in layout.names()
            if layout.slc(name).start >= feature_start
            and layout.slc(name).stop <= feature_end
        ]
        if not feature_names:
            feature_names = None

        if isinstance(targets, Sequence):
            target_names = [str(name) for name in targets]

        if node_type_name is None and node_type_index is not None:
            for name in layout.names():
                slc = layout.slc(name)
                if slc.start <= node_type_index < slc.stop:
                    node_type_name = name
                    break

    return Simulator(
        node_input_size=node_input_size,
        edge_input_size=param["model"]["edge_input_size"],
        output_size=param["model"]["output_size"],
        feature_index_start=feature_start,
        feature_index_end=feature_end,
        output_index_start=output_start,
        output_index_end=output_end,
        node_type_index=param["index"]["node_type_index"],
        model=model,
        device=device,
        x_layout=layout,
        feature_names=feature_names,
        target_names=target_names,
        node_type_name=node_type_name,
    )


def get_dataset(
    param: Dict[str, Any],
    preprocessing: Callable[[Data], Data],
    masking_ratio: Optional[float] = None,
    use_edge_feature: bool = True,
    use_previous_data: bool = False,
    switch_to_val: bool = False,
):
    """
    Constructs the dataset based on provided parameters.

    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.
        preprocessing (Callable[[Data], Data]): The preprocessing function to apply to the data.
        masking_ratio (Optional[float], optional): The ratio of data to mask. Defaults to None.
        use_edge_feature (bool, optional): Whether to add edge features. Defaults to True.
        use_previous_data (bool, optional): Whether to use previous data in the dataset. Defaults to False.

    Returns:
        Dataset: The constructed dataset.

    Raises:
        ValueError: If the dataset extension specified in param is not supported.
    """
    dataset_params = param.get("dataset", {})
    targets = dataset_params.get("targets", [])
    if len(targets) == 0:
        raise ValueError("Please provide a list of target properties to predict.")
    khop = dataset_params.get("khop", 1)
    new_edges_ratio = dataset_params.get("new_edges_ratio", 0)
    extension = dataset_params.get("extension", "")

    named_section = param.get("named_features", {})
    x_layout: Optional[XFeatureLayout] = named_section.get("x_layout")
    x_coords = (
        named_section.get("x_coords") if isinstance(named_section, Mapping) else None
    )

    world_pos_parameters = None
    if khop > 1:
        transformations = param.get("transformations", {})
        if "world_pos_parameters" in transformations:
            wpp = transformations["world_pos_parameters"]
            if wpp.get("use", False):
                world_pos_parameters = wpp

    if extension == "h5":
        return H5Dataset(
            h5_path=dataset_params["h5_path"],
            meta_path=dataset_params["meta_path"],
            targets=targets,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=use_edge_feature,
            use_previous_data=use_previous_data,
            switch_to_val=switch_to_val,
            world_pos_parameters=world_pos_parameters,
            x_layout=x_layout,
            x_coords=x_coords,
        )
    elif extension == "xdmf":
        return XDMFDataset(
            xdmf_folder=dataset_params["xdmf_folder"],
            meta_path=dataset_params["meta_path"],
            targets=targets,
            preprocessing=preprocessing,
            masking_ratio=masking_ratio,
            khop=khop,
            new_edges_ratio=new_edges_ratio,
            add_edge_features=use_edge_feature,
            use_previous_data=use_previous_data,
            switch_to_val=switch_to_val,
            x_layout=x_layout,
            x_coords=x_coords,
        )
    else:
        raise ValueError(f"Dataset extension '{extension}' not supported.")


def get_num_workers(param: Dict[str, Any], default_num_workers: int) -> int:
    """
    Determines the number of workers to use for DataLoader based on dataset extension.

    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.
        default_num_workers (int): The default number of workers specified.

    Returns:
        int: The adjusted number of workers.
    """
    dataset_params = param.get("dataset", {})
    extension = dataset_params.get("extension", "")
    if extension == "h5":
        return default_num_workers
    elif extension == "xdmf":
        return default_num_workers
    else:
        raise ValueError(f"Dataset extension '{extension}' not supported.")


def get_loss(param: Dict[str, Any], **kwargs):
    """
    Parse parameters for loss function. If several loss types are specified, a weighted loss is used.
    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.

    Returns:
        Loss: Initialised loss object.
        Union[str, List[str]]: loss name if single loss, list of loss name if MultiLoss.
    """
    try:
        _ = param["loss"]
    except KeyError:
        logger.info("No loss specified, fall back to default loss L2Loss")
        return LossType.L2LOSS.value(**kwargs), LossType.L2LOSS.name

    if len(param["loss"]["type"]) > 1:
        losses = [LossType[t.upper()].value(**kwargs) for t in param["loss"]["type"]]
        losses_names = [LossType[t.upper()].name for t in param["loss"]["type"]]
        weights = param["loss"]["weights"]
        return MultiLoss(losses, weights), losses_names
    else:
        loss = LossType[param["loss"]["type"][0].upper()]
        return loss.value(**kwargs), loss.name


def get_gradient_method(param: Dict[str, Any], **kwargs) -> str:
    """
    Parse parameters for gradient computation method. If not specified, returns None.
    Args:
        param (Dict[str, Any]): Dictionary containing configuration parameters.

    Returns:
        str: Name of gradient method.
    """
    try:
        gradient_method = param["loss"]["gradient_method"]
    except KeyError:
        logger.info("No gradient method specified.")
        gradient_method = None
    return gradient_method
