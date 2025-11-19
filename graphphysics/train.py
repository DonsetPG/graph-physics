import json
import os
import warnings

import torch
import wandb
from absl import app, flags
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch_geometric.loader import DataLoader

from graphphysics.external.aneurysm import build_features
from graphphysics.training.callback import LogPyVistaPredictionsCallback
from graphphysics.training.lightning_module import LightningModule
from graphphysics.training.parse_parameters import (
    get_dataset,
    get_num_workers,
    get_preprocessing,
)
from graphphysics.utils.progressbar import ColabProgressBar

import socket
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy
from datetime import timedelta  # si tu ne l’as pas déjà


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

def print_dist_info(stage: str):
    """Affiche les variables de distribution pour debug (appel court)."""
    hostname = socket.gethostname()
    env = os.environ
    rank = int(env.get("RANK", env.get("SLURM_PROCID", -1)))
    local_rank = int(env.get("LOCAL_RANK", env.get("SLURM_LOCALID", -1)))
    node_rank = int(env.get("NODE_RANK", env.get("SLURM_NODEID", -1)))
    world_size = int(env.get("WORLD_SIZE", env.get("SLURM_NTASKS", -1)))
    initialized = dist.is_available() and dist.is_initialized()
    num_gpus = torch.cuda.device_count()
    curr = torch.cuda.current_device() if torch.cuda.is_available() and num_gpus > 0 else -1
    gpu_name = torch.cuda.get_device_name(curr) if curr != -1 else "CPU"
    print(
        f"[{stage}] host={hostname} rank={rank} local_rank={local_rank} node_rank={node_rank} "
        f"world_size={world_size} dist_init={initialized} gpus_on_node={num_gpus} "
        f"current_device={curr} device_name={gpu_name}",
        flush=True,
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
    preproc_device = torch.device("cpu")

    #wandb_project_name = FLAGS.project_name
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

    seed_everything(FLAGS.seed, workers=True)

    print_dist_info("startup")


    # Build preprocessing function
    train_preprocessing = get_preprocessing(
        param=parameters,
        #device=device,
        device=preproc_device,
        use_edge_feature=use_edge_feature,
        extra_node_features=build_features,
    )

    # Get training and validation datasets
    train_dataset = get_dataset(
        param=parameters,
        preprocessing=train_preprocessing,
        use_edge_feature=use_edge_feature,
        use_previous_data=use_previous_data,
    )

    val_preprocessing = get_preprocessing(
        param=parameters,
        #device=device,
        device=preproc_device,
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

    print("---- DEBUG ----")
    print("TRAIN Dataset type:", type(train_dataset))
    print("Number of XDMF files:", train_dataset.size_dataset)
    print("Trajectory length:", train_dataset.trajectory_length)
    print("Computed len(train_dataset):", len(train_dataset))
    print("----------------")
    print("VAL Dataset type:", type(val_dataset))
    print("Number of XDMF files:", val_dataset.size_dataset)
    print("Trajectory length:", val_dataset.trajectory_length)
    print("Computed len(val_dataset):", len(val_dataset))
    print("----------------")
    
    num_workers = get_num_workers(param=parameters, default_num_workers=num_workers)

    rank_env = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    if rank_env == 0:
        print("---- DEBUG (rank 0) ----")
        print("TRAIN Dataset type:", type(train_dataset))
        print("Number of XDMF files:", train_dataset.size_dataset)
        print("Trajectory length:", train_dataset.trajectory_length)
        print("Computed len(train_dataset):", len(train_dataset))
        print("----------------")
        print("VAL Dataset type:", type(val_dataset))
        print("Number of XDMF files:", val_dataset.size_dataset)
        print("Trajectory length:", val_dataset.trajectory_length)
        print("Computed len(val_dataset):", len(val_dataset))
        print("----------------")
    else:
        # Court message pour confirmer la présence des autres ranks
        print(f"[rank {rank_env}] len(train)={len(train_dataset)} len(val)={len(val_dataset)}", flush=True)

    train_dataloader_kwargs = {
        "dataset": train_dataset,
        "shuffle": True,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "exclude_keys": ["tetra"],
        "drop_last": True,
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
    '''
    if resume_training:
        wandb_run = wandb.init(
            project=wandb_project_name, id=lightning_module.wandb_run_id, resume="allow"
        )
    else:
        wandb_run = wandb.init(project=wandb_project_name)

    wandb_logger = WandbLogger(experiment=wandb_run)
    lightning_module.wandb_run_id = wandb_logger.experiment.id
    '''
    if model_save_name is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/", filename=model_save_name
        )
    else:
        checkpoint_callback = ModelCheckpoint(dirpath="checkpoints")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    '''
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
    '''
    '''
    # Configure Trainer
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=num_epochs,
        #logger=wandb_logger,
        callbacks=[
            ColabProgressBar(),
            checkpoint_callback,
            #LogPyVistaPredictionsCallback(dataset=val_dataset, indices=[1, 2, 3]),
            lr_monitor,
        ],
        log_every_n_steps=100,
        gradient_clip_val=1.0,
    )
    '''
    # === Trainer DDP ===
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if devices == 0:
        raise RuntimeError("Aucun GPU visible alors que DDP/gpu est demandé. Vérifie l'allocation SLURM.")

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,               # 1 GPU par process (PL mappe LOCAL_RANK -> CUDA)
        num_nodes=num_nodes,
        strategy=DDPStrategy(
            process_group_backend="nccl",   # explicite
            find_unused_parameters=False,   # évite des allreduces non appariés
            static_graph=True,              # exige même graphe/chemin à chaque itération
            timeout=timedelta(minutes=15),  # fail > deadlock
        ),
        precision="bf16-mixed",      # <-- AJOUT: H100 supporte BF16 nativement
        max_epochs=num_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            ColabProgressBar(),
            checkpoint_callback,
            lr_monitor,
        ],
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        # --- SMOKE TEST: borner provisoirement le # de batches ---
        #limit_train_batches=5,              # <--- TEMP pour diagnostiquer (remets 1.0 après)
        #limit_val_batches=2,                # <--- TEMP remettre a 2
    )
    print_dist_info("trainer_built")

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

    print_dist_info("fit_done")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    app.run(main)
