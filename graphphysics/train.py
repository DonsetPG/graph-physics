import json
import os
import warnings

import torch
import wandb
from absl import app, flags
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch_geometric.loader import DataLoader

from graphphysics.training.callback import LogPyVistaPredictionsCallback
from graphphysics.training.lightning_module import LightningModule
from graphphysics.training.parse_parameters import (
    get_dataset,
    get_num_workers,
    get_preprocessing,
)
from graphphysics.utils.progressbar import ColabProgressBar

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

torch.set_float32_matmul_precision("high")

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "my_project", "Name of the WandB project")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate")
flags.DEFINE_integer("batch_size", 2, "Batch size")
flags.DEFINE_integer("warmup", 1000, "Learning rate warmup steps")
flags.DEFINE_integer("num_workers", 2, "Number of DataLoader workers")
flags.DEFINE_integer("prefetch_factor", 2, "Number of batches to prefetch")
flags.DEFINE_string("model_save_path", None, "Path to the checkpoint (.ckpt) file")
flags.DEFINE_bool("use_previous_data", False, "Whether to use previous data or not")
flags.DEFINE_integer(
    "previous_data_start", 4, "Index of the start of the previous data in the features"
)
flags.DEFINE_integer(
    "previous_data_end", 7, "Index of the end of the previous data in the features"
)
flags.DEFINE_bool("no_edge_feature", False, "Whether to use edge features")
flags.DEFINE_string(
    "training_parameters_path", None, "Path to the training parameters JSON file"
)


def main(argv):
    del argv

    # Check that the training parameters path is provided
    if not FLAGS.training_parameters_path:
        raise ValueError("The 'training_parameters_path' flag must be provided.")

    # Load training parameters from JSON file
    training_parameters_path = FLAGS.training_parameters_path
    logger.info(f"Opening training parameters from {training_parameters_path}")
    try:
        with open(training_parameters_path, "r") as fp:
            parameters = json.load(fp)
    except Exception as e:
        logger.error(f"Error reading training parameters: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb_project_name = FLAGS.project_name
    num_epochs = FLAGS.num_epochs
    initial_lr = FLAGS.init_lr
    batch_size = FLAGS.batch_size
    warmup = FLAGS.warmup
    num_workers = FLAGS.num_workers
    prefetch_factor = FLAGS.prefetch_factor
    model_save_path = FLAGS.model_save_path
    use_edge_feature = not FLAGS.no_edge_feature
    use_previous_data = FLAGS.use_previous_data
    previous_data_start = FLAGS.previous_data_start
    previous_data_end = FLAGS.previous_data_end

    # Build preprocessing function
    preprocessing = get_preprocessing(
        param=parameters,
        device=device,
        use_edge_feature=use_edge_feature,
        extra_node_features=None,
    )

    # Get training and validation datasets
    train_dataset = get_dataset(
        param=parameters,
        preprocessing=preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=use_previous_data,
    )

    val_dataset = get_dataset(
        param=parameters,
        preprocessing=preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=use_previous_data,
        switch_to_val=True,
    )

    num_workers = get_num_workers(param=parameters, default_num_workers=num_workers)

    train_dataloader_kwargs = {
        "dataset": train_dataset,
        "shuffle": True,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    valid_dataloader_kwargs = {
        "dataset": val_dataset,
        "shuffle": False,
        "batch_size": 1,
        "num_workers": num_workers,
    }

    # Update arguments if num_workers > 0
    if num_workers > 0:
        train_dataloader_kwargs.update(
            {
                "prefetch_factor": prefetch_factor,
                "persistent_workers": True,
            }
        )
        valid_dataloader_kwargs.update(
            {
                "prefetch_factor": prefetch_factor,
                "persistent_workers": True,
            }
        )

    # Create DataLoaders
    train_dataloader = DataLoader(**train_dataloader_kwargs)
    valid_dataloader = DataLoader(**valid_dataloader_kwargs)

    # Define or resume model
    num_steps = num_epochs * len(train_dataloader)

    prev_data_kwargs = {}
    if use_previous_data is True:
        prev_data_kwargs = {
            "use_previous_data": True,
            "previous_data_start": previous_data_start,
            "previous_data_end": previous_data_end,
        }

    if model_save_path and os.path.isfile(model_save_path):
        logger.info(f"Loading model from checkpoint: {model_save_path}")
        lightning_module = LightningModule.load_from_checkpoint(
            checkpoint_path=model_save_path,
            parameters=parameters,
            warmup=warmup,
            learning_rate=initial_lr,
            num_steps=num_steps,
            trajectory_length=train_dataset.trajectory_length,
            **prev_data_kwargs,
        )
    else:
        logger.info("Initializing new model")
        lightning_module = LightningModule(
            parameters=parameters,
            learning_rate=initial_lr,
            num_steps=num_steps,
            warmup=warmup,
            trajectory_length=train_dataset.trajectory_length,
            **prev_data_kwargs,
        )

    # Initialize WandbLogger
    wandb_run = wandb.init(project=wandb_project_name)
    wandb_logger = WandbLogger(experiment=wandb_run)
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    wandb_logger.experiment.config.update(
        {
            "architecture": parameters["model"]["type"],
            "#_layers": parameters["model"]["message_passing_num"],
            "#_neurons": parameters["model"]["hidden_size"],
            "#_hops": parameters["dataset"]["khop"],
            "max_lr": initial_lr,
            "batch_size": batch_size,
        }
    )

    # Configure Trainer
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[
            ColabProgressBar(),
            checkpoint_callback,
            LogPyVistaPredictionsCallback(dataset=val_dataset, indices=[1, 50, 100]),
            lr_monitor,
        ],
        log_every_n_steps=100,
    )

    # Start training
    logger.success("Starting training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)
