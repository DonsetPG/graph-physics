import json
import os

from absl import app, flags
from flax import nnx
from loguru import logger

from jraphphysics.training.parse_parameters import (
    get_dataset,
    get_gradient_method,
    get_loss,
    get_model,
    get_preprocessing,
    get_simulator,
)
from jraphphysics.training.workflows import (
    SimpleTrainer,
    load_checkpoint,
    save_checkpoint,
)

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate")
flags.DEFINE_integer("max_train_samples", -1, "Max train samples per epoch")
flags.DEFINE_integer("max_val_samples", -1, "Max validation samples per epoch")
flags.DEFINE_string("project_name", "my_project", "Name of the WandB project")
flags.DEFINE_bool("use_wandb", True, "Enable Weights & Biases logging")
flags.DEFINE_string(
    "model_save_name",
    "checkpoints/jraphphysics_checkpoint.pkl",
    "Path to save checkpoints",
)
flags.DEFINE_string(
    "model_path",
    None,
    "Checkpoint path to resume from",
)
flags.DEFINE_bool(
    "resume_training",
    False,
    "Resume from an existing checkpoint",
)
flags.DEFINE_bool("use_previous_data", True, "Whether to use previous frame data")
flags.DEFINE_string(
    "training_parameters_path",
    None,
    "Path to training parameters JSON file",
)


def _to_limit(value: int):
    if value is None or value < 0:
        return None
    return value


def main(argv):
    del argv

    if not FLAGS.training_parameters_path:
        raise ValueError("The 'training_parameters_path' flag must be provided.")

    with open(FLAGS.training_parameters_path, "r") as fp:
        parameters = json.load(fp)

    preprocessing = get_preprocessing(parameters)
    val_preprocessing = get_preprocessing(parameters, remove_noise=True)

    train_dataset = get_dataset(
        param=parameters,
        preprocessing=preprocessing,
        use_previous_data=FLAGS.use_previous_data,
    )
    val_dataset = get_dataset(
        param=parameters,
        preprocessing=val_preprocessing,
        use_previous_data=FLAGS.use_previous_data,
        switch_to_val=True,
    )

    if FLAGS.resume_training and not FLAGS.model_path:
        raise ValueError("'resume_training' requires '--model_path'.")

    if FLAGS.model_path and os.path.isfile(FLAGS.model_path):
        logger.info(f"Loading checkpoint from {FLAGS.model_path}")
        checkpoint = load_checkpoint(FLAGS.model_path)
        simulator = checkpoint["simulator"]
        start_epoch = int(checkpoint.get("epoch", 0))
        metadata = checkpoint.get("metadata", {})
        wandb_run_id = metadata.get("wandb_run_id")
    else:
        logger.info("Building new jraphphysics simulator.")
        model = get_model(parameters, rngs=nnx.Rngs(params=FLAGS.seed, dropout=FLAGS.seed + 1))
        simulator = get_simulator(
            parameters,
            model=model,
            rngs=nnx.Rngs(params=FLAGS.seed + 2, dropout=FLAGS.seed + 3),
        )
        start_epoch = 0
        wandb_run_id = None

    wandb_run = None
    if FLAGS.use_wandb:
        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not installed. Continuing without WandB logging.")
        else:
            if FLAGS.resume_training and wandb_run_id:
                wandb_run = wandb.init(
                    project=FLAGS.project_name,
                    id=wandb_run_id,
                    resume="allow",
                )
                logger.info(f"Resuming WandB run: {wandb_run_id}")
            else:
                wandb_run = wandb.init(project=FLAGS.project_name)
            wandb_run_id = wandb_run.id
            wandb_run.config.update(
                {
                    "architecture": parameters["model"]["type"],
                    "#_layers": parameters["model"]["message_passing_num"],
                    "#_neurons": parameters["model"]["hidden_size"],
                    "#_hops": parameters["dataset"]["khop"],
                    "max_lr": FLAGS.init_lr,
                    "batch_size": 1,
                }
            )

    loss_fn, loss_name = get_loss(parameters)
    gradient_method = get_gradient_method(parameters)
    trainer = SimpleTrainer(
        simulator=simulator,
        learning_rate=FLAGS.init_lr,
        loss_fn=loss_fn,
        loss_name=loss_name,
        gradient_method=gradient_method,
        logger=wandb_run,
    )
    history = trainer.fit(
        dataset=train_dataset,
        num_epochs=FLAGS.num_epochs,
        preprocessing=preprocessing,
        val_dataset=val_dataset,
        val_preprocessing=val_preprocessing,
        max_train_samples=_to_limit(FLAGS.max_train_samples),
        max_val_samples=_to_limit(FLAGS.max_val_samples),
    )

    end_epoch = start_epoch + FLAGS.num_epochs
    checkpoint_path = save_checkpoint(
        path=FLAGS.model_save_name,
        simulator=trainer.simulator,
        epoch=end_epoch,
        metadata={
            "history": history,
            "training_parameters_path": FLAGS.training_parameters_path,
            "wandb_run_id": wandb_run_id,
        },
    )
    logger.success(f"Training complete. Checkpoint saved to {checkpoint_path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    app.run(main)
