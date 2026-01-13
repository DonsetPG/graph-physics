import json
import os
import warnings

import torch
from absl import app, flags
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch_geometric.loader import DataLoader

import wandb
from graphphysics.external.aneurysm import build_features
# from graphphysics.training.callback import LogPyVistaPredictionsCallback
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
torch.multiprocessing.set_sharing_strategy("file_system")

FLAGS = flags.FLAGS
flags.DEFINE_string("project_name", "my_project", "Name of the WandB project")
flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate")
flags.DEFINE_integer("batch_size", 2, "Batch size")
flags.DEFINE_integer("warmup", 1000, "Learning rate warmup steps")
flags.DEFINE_integer("num_workers", 2, "Number of DataLoader workers")
flags.DEFINE_integer("prefetch_factor", 2, "Number of batches to prefetch")
flags.DEFINE_string(
    "model_save_name", None, "Name to save the checkpoint during training"
)
flags.DEFINE_string(
    "model_path", None, "Path to the checkpoint (.ckpt) to resume training from"
)
flags.DEFINE_bool(
    "resume_training", False, "Whether to resume an unfinished training or not"
)

flags.DEFINE_bool("use_previous_data", True, "Whether to use previous data or not")
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
flags.DEFINE_bool(
    "use_partitioning", False, "Whether to use graph partitioning"
)
flags.DEFINE_integer(
    "num_partitions", 1, "Number of partitions for graph partitioning"
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
    model_save_name = FLAGS.model_save_name
    model_path = FLAGS.model_path
    resume_training = FLAGS.resume_training
    use_edge_feature = not FLAGS.no_edge_feature
    use_previous_data = FLAGS.use_previous_data
    previous_data_start = FLAGS.previous_data_start
    previous_data_end = FLAGS.previous_data_end
    use_partitioning = FLAGS.use_partitioning
    num_partitions = FLAGS.num_partitions

    seed_everything(FLAGS.seed, workers=True)

    # Build preprocessing function
    train_preprocessing = get_preprocessing(
        param=parameters,
        device=device,
        use_edge_feature=use_edge_feature,
        extra_node_features=build_features,
    )

    # Get training and validation datasets
    train_dataset = get_dataset(
        param=parameters,
        preprocessing=train_preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=use_previous_data,
        use_partitioning=use_partitioning,
        num_partitions=num_partitions
    )

    val_preprocessing = get_preprocessing(
        param=parameters,
        device=device,
        use_edge_feature=use_edge_feature,
        remove_noise=True,
        extra_node_features=build_features,
    )

    val_dataset = get_dataset(
        param=parameters,
        preprocessing=val_preprocessing,
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
        "exclude_keys": ["tetra"],
    }

    valid_dataloader_kwargs = {
        "dataset": val_dataset,
        "shuffle": False,
        "batch_size": 1,
        "num_workers": num_workers,
    }

    if train_dataset.type == "h5":
        train_dataloader_kwargs["pin_memory"] = device.type == "cuda"
        valid_dataloader_kwargs["pin_memory"] = device.type == "cuda"

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

    if model_path and os.path.isfile(model_path):
        logger.info(f"Loading model from checkpoint: {model_path}")
        lightning_module = LightningModule.load_from_checkpoint(
            checkpoint_path=model_path,
            parameters=parameters,
            warmup=warmup,
            learning_rate=initial_lr,
            num_steps=num_steps,
            trajectory_length=train_dataset.trajectory_length,
            timestep=train_dataset.dt,
            **prev_data_kwargs,
        )
        logger.info(f"Resuming WandB run: {lightning_module.wandb_run_id}")
    else:
        logger.info("Initializing new model")
        lightning_module = LightningModule(
            parameters=parameters,
            learning_rate=initial_lr,
            num_steps=num_steps,
            warmup=warmup,
            trajectory_length=train_dataset.trajectory_length,
            timestep=train_dataset.dt,
            **prev_data_kwargs,
        )

    # Initialize WandbLogger
    if resume_training:
        wandb_run = wandb.init(
            project=wandb_project_name, id=lightning_module.wandb_run_id, resume="allow"
        )
    else:
        wandb_run = wandb.init(project=wandb_project_name)

    wandb_logger = WandbLogger(experiment=wandb_run)
    lightning_module.wandb_run_id = wandb_logger.experiment.id
    if model_save_name is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/", filename=model_save_name
        )
    else:
        checkpoint_callback = ModelCheckpoint(dirpath="checkpoints")
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
            LogPyVistaPredictionsCallback(dataset=val_dataset, indices=[1, 2, 3]),
            lr_monitor,
        ],
        log_every_n_steps=100,
        gradient_clip_val=1.0,
        # accumulate_grad_batches=num_partitions
    )

    # Resuming training from a checkpoint
    if model_path and os.path.isfile(model_path) and resume_training:
        logger.success("Resuming training")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
            ckpt_path=model_path,
        )
    else:
        logger.success("Starting training")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)
