import json
import warnings

import torch
from absl import app, flags
from loguru import logger
from torch_geometric.loader import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from graphphysics.training.lightning_module import LightningModule
from graphphysics.training.parse_parameters import (
    get_dataset,
    get_num_workers,
    get_preprocessing,
)
from graphphysics.external.aneurysm import build_features

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

torch.set_float32_matmul_precision("high")

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 2, "Batch size")
flags.DEFINE_integer("num_workers", 2, "Number of DataLoader workers")
flags.DEFINE_integer("prefetch_factor", 2, "Number of batches to prefetch")
flags.DEFINE_string("model_path", None, "Path to the checkpoint (.ckpt) file")
flags.DEFINE_bool("no_edge_feature", False, "Whether to use edge features")
flags.DEFINE_string(
    "predict_parameters_path", None, "Path to the training parameters JSON file"
)


def main(argv):
    del argv

    # Check that the parameters path is provided
    if not FLAGS.predict_parameters_path:
        logger.error("The 'predict_parameters_path' flag must be provided.")
        return

    # Load training parameters from JSON file
    parameters_path = FLAGS.predict_parameters_path
    logger.info(f"Opening prediction parameters from {parameters_path}")
    try:
        with open(parameters_path, "r") as fp:
            parameters = json.load(fp)
    except Exception as e:
        logger.error(f"Error reading training parameters: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = FLAGS.batch_size
    num_workers = FLAGS.num_workers
    prefetch_factor = FLAGS.prefetch_factor
    model_path = FLAGS.model_path
    use_edge_feature = not FLAGS.no_edge_feature

    # Build preprocessing function
    preprocessing = get_preprocessing(
        param=parameters,
        device=device,
        use_edge_feature=use_edge_feature,
        extra_node_features=build_features,
    )

    # Get predict datasets
    predict_dataset = get_dataset(
        param=parameters,
        preprocessing=preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=True,
    )

    num_workers = get_num_workers(param=parameters, default_num_workers=num_workers)

    predict_dataloader_kwargs = {
        "dataset": predict_dataset,
        "shuffle": False,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    # Update arguments if num_workers > 0
    if num_workers > 0:
        predict_dataloader_kwargs.update(
            {
                "prefetch_factor": prefetch_factor,
                "persistent_workers": True,
            }
        )

    # Create DataLoader
    # TODO: let's test with only one traj for now
    predict_dataloader = DataLoader(**predict_dataloader_kwargs)

    # Load trained model

    logger.info(f"Loading model from checkpoint: {model_path}")
    lightning_module = LightningModule.load_from_checkpoint(
        checkpoint_path=model_path,
        parameters=parameters,
        trajectory_length=predict_dataset.trajectory_length,
    )

    csv_logger = CSVLogger("logs", name="prediction_log")
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      logger=csv_logger,
                      devices=1,
                      inference_mode=True)

    # Start prediction
    logger.success("Starting prediction")
    trainer.predict(model=lightning_module, dataloaders=predict_dataloader)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)
