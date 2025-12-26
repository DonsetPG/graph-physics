import json
import os
import warnings
from typing import Iterable, Optional, Tuple

import torch
from absl import app, flags
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graphphysics.external.aneurysm import build_features
from graphphysics.training.lightning_module import LightningModule
from graphphysics.training.parse_parameters import (
    get_dataset,
    get_num_workers,
    get_preprocessing,
)

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "training_parameters_path", None, "Path to the training parameters JSON file"
)
flags.DEFINE_string(
    "model_path", None, "Path to a checkpoint (.ckpt) to load a trained model"
)
flags.DEFINE_string(
    "dataset_split",
    "train",
    "Dataset split to use (train or val). 'val' switches to test files like train.py.",
)
flags.DEFINE_integer("batch_size", 1, "Batch size (use 1 for full-graph estimates)")
flags.DEFINE_integer("num_workers", 2, "Number of DataLoader workers")
flags.DEFINE_integer("prefetch_factor", 2, "Number of batches to prefetch")
flags.DEFINE_integer(
    "max_batches",
    -1,
    "Maximum number of batches to scan (-1 means full dataset)",
)
flags.DEFINE_integer(
    "normalizer_batches",
    0,
    "If >0 and no checkpoint is provided, warm up normalizers on this many batches.",
)
flags.DEFINE_bool("no_edge_feature", False, "Whether to use edge features")
flags.DEFINE_bool("use_previous_data", True, "Whether to use previous data or not")
flags.DEFINE_integer(
    "previous_data_start", 5, "Index of the start of the previous data in the features"
)
flags.DEFINE_integer(
    "previous_data_end", 8, "Index of the end of the previous data in the features"
)
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_string(
    "plot_path",
    None,
    "Optional path to save a plot of per-graph K values (e.g., outputs/k_lipschitz.png).",
)


def _split_batch(batch) -> Tuple[Data, Optional[torch.Tensor]]:
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    return batch, None


def _build_dataloader(dataset, batch_size: int, num_workers: int, prefetch_factor: int, device):
    dataloader_kwargs = {
        "dataset": dataset,
        "shuffle": False,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "exclude_keys": ["tetra"],
    }

    if dataset.type == "h5":
        dataloader_kwargs["pin_memory"] = device.type == "cuda"

    if num_workers > 0:
        dataloader_kwargs.update(
            {
                "prefetch_factor": prefetch_factor,
                "persistent_workers": True,
            }
        )

    return DataLoader(**dataloader_kwargs)


def _warmup_normalizers(
    module: LightningModule,
    dataloader: Iterable,
    device: torch.device,
    max_batches: int,
) -> None:
    if max_batches <= 0:
        return

    logger.info(
        f"Warming up normalizers on {max_batches} batches (train mode, no grad)."
    )
    module.train()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            graph, _ = _split_batch(batch)
            graph = graph.to(device)
            module.model(graph)
    module.eval()


def _estimate_lipschitz(
    module: LightningModule,
    dataloader: Iterable,
    device: torch.device,
    max_batches: int,
):
    model = module.model
    loss_fn = module.loss
    loss_masks = module.loss_masks
    gradient_method = module.gradient_method
    is_multiloss = module.is_multiloss

    k_values = []
    skipped = 0

    module.eval()
    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        graph, selected_indices = _split_batch(batch)
        graph = graph.to(device)
        if selected_indices is not None:
            selected_indices = selected_indices.to(device)

        with torch.no_grad():
            network_output, target_delta_normalized, _ = model(graph)

        network_output_var = network_output.detach().requires_grad_(True)
        node_type = graph.x[:, model.node_type_index]

        if is_multiloss:
            network_output_physical = model.build_outputs(graph, network_output_var)
            target_physical = model.build_outputs(graph, target_delta_normalized)
            loss_val = loss_fn(
                graph=graph,
                target=target_delta_normalized,
                network_output=network_output_var,
                node_type=node_type,
                masks=loss_masks,
                selected_indexes=selected_indices,
                network_output_physical=network_output_physical,
                target_physical=target_physical,
                gradient_method=gradient_method,
            )
        else:
            loss_val = loss_fn(
                graph=graph,
                target=target_delta_normalized,
                network_output=network_output_var,
                node_type=node_type,
                masks=loss_masks,
                selected_indexes=selected_indices,
                gradient_method=gradient_method,
            )

        if torch.isnan(loss_val) or torch.isinf(loss_val):
            skipped += 1
            continue

        grad = torch.autograd.grad(
            loss_val, network_output_var, retain_graph=False, create_graph=False
        )[0]
        if grad is None:
            skipped += 1
            continue

        grad_norm = torch.linalg.vector_norm(grad).item()
        if not (grad_norm == grad_norm):  # NaN check without extra deps
            skipped += 1
            continue
        k_values.append(grad_norm)

    return k_values, skipped


def main(argv):
    del argv

    if not FLAGS.training_parameters_path:
        raise ValueError("The 'training_parameters_path' flag must be provided.")

    training_parameters_path = FLAGS.training_parameters_path
    logger.info(f"Opening training parameters from {training_parameters_path}")
    try:
        with open(training_parameters_path, "r") as fp:
            parameters = json.load(fp)
    except Exception as exc:
        logger.error(f"Error reading training parameters: {exc}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(FLAGS.seed)

    use_edge_feature = not FLAGS.no_edge_feature
    use_previous_data = FLAGS.use_previous_data

    train_preprocessing = get_preprocessing(
        param=parameters,
        device=device,
        use_edge_feature=use_edge_feature,
        extra_node_features=build_features,
    )

    use_val = FLAGS.dataset_split.lower() in {"val", "valid", "validation", "test"}
    dataset = get_dataset(
        param=parameters,
        preprocessing=train_preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=use_previous_data,
        switch_to_val=use_val,
    )

    num_workers = get_num_workers(param=parameters, default_num_workers=FLAGS.num_workers)
    dataloader = _build_dataloader(
        dataset=dataset,
        batch_size=FLAGS.batch_size,
        num_workers=num_workers,
        prefetch_factor=FLAGS.prefetch_factor,
        device=device,
    )

    num_steps = max(1, len(dataloader))
    prev_data_kwargs = {}
    if use_previous_data:
        prev_data_kwargs = {
            "use_previous_data": True,
            "previous_data_start": FLAGS.previous_data_start,
            "previous_data_end": FLAGS.previous_data_end,
        }

    if FLAGS.model_path and os.path.isfile(FLAGS.model_path):
        logger.info(f"Loading model from checkpoint: {FLAGS.model_path}")
        lightning_module = LightningModule.load_from_checkpoint(
            checkpoint_path=FLAGS.model_path,
            parameters=parameters,
            warmup=0,
            learning_rate=0.0,
            num_steps=num_steps,
            trajectory_length=dataset.trajectory_length,
            timestep=dataset.dt,
            **prev_data_kwargs,
        )
    else:
        if FLAGS.model_path:
            logger.warning(f"Checkpoint not found: {FLAGS.model_path}. Using fresh model.")
        lightning_module = LightningModule(
            parameters=parameters,
            learning_rate=0.0,
            num_steps=num_steps,
            warmup=0,
            trajectory_length=dataset.trajectory_length,
            timestep=dataset.dt,
            **prev_data_kwargs,
        )

    lightning_module = lightning_module.to(device)
    lightning_module.eval()

    if (not FLAGS.model_path) and FLAGS.normalizer_batches > 0:
        _warmup_normalizers(
            module=lightning_module,
            dataloader=dataloader,
            device=device,
            max_batches=FLAGS.normalizer_batches,
        )

    k_values, skipped = _estimate_lipschitz(
        module=lightning_module,
        dataloader=dataloader,
        device=device,
        max_batches=FLAGS.max_batches,
    )

    if not k_values:
        logger.error("No valid batches were processed. Check dataset/masks for empty graphs.")
        return

    k_tensor = torch.tensor(k_values, dtype=torch.float32)
    k_max = k_tensor.max().item()
    k_mean = k_tensor.mean().item()
    k_p95 = torch.quantile(k_tensor, 0.95).item()

    logger.success(
        "Estimated K-Lipschitz constant (L2) for the full-graph loss:\n"
        f"  K_max={k_max:.6g} | K_mean={k_mean:.6g} | K_p95={k_p95:.6g}\n"
        f"  graphs={len(k_values)} | skipped={skipped}"
    )

    if FLAGS.plot_path:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(os.path.dirname(FLAGS.plot_path), exist_ok=True)
            plt.figure(figsize=(10, 4))
            plt.plot(k_values, linewidth=1.25)
            plt.xlabel("Graph index")
            plt.ylabel("K estimate (||∂L/∂ŷ||₂)")
            plt.title("Per-graph K-Lipschitz estimates")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FLAGS.plot_path, dpi=150)
            plt.close()
            logger.success(f"Saved K plot to {FLAGS.plot_path}")
        except Exception as exc:
            logger.error(f"Failed to save plot: {exc}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)
