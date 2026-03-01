import json
import os

from absl import app, flags
from loguru import logger

from jraphphysics.training.parse_parameters import (
    get_dataset,
    get_preprocessing,
)
from jraphphysics.training.workflows import (
    load_checkpoint,
    rollout,
    save_rollout_predictions,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", None, "Path to checkpoint (.pkl) file")
flags.DEFINE_string(
    "predict_parameters_path",
    None,
    "Path to prediction parameters JSON file",
)
flags.DEFINE_string(
    "prediction_save_path",
    "predictions/jraphphysics_predictions.npz",
    "Path where predictions will be saved",
)
flags.DEFINE_integer(
    "num_steps",
    -1,
    "Number of rollout steps. -1 means full dataset.",
)
flags.DEFINE_bool("use_previous_data", True, "Whether to use previous frame data")


def main(argv):
    del argv

    if not FLAGS.predict_parameters_path:
        raise ValueError("The 'predict_parameters_path' flag must be provided.")
    if not FLAGS.model_path:
        raise ValueError("The 'model_path' flag must be provided.")
    if not os.path.isfile(FLAGS.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {FLAGS.model_path}")

    with open(FLAGS.predict_parameters_path, "r") as fp:
        parameters = json.load(fp)

    preprocessing = get_preprocessing(parameters, remove_noise=True)

    predict_dataset = get_dataset(
        param=parameters,
        preprocessing=preprocessing,
        use_previous_data=FLAGS.use_previous_data,
    )

    checkpoint = load_checkpoint(FLAGS.model_path)
    simulator = checkpoint["simulator"]

    rollout_steps = None if FLAGS.num_steps < 0 else FLAGS.num_steps
    predictions = rollout(
        simulator=simulator,
        dataset=predict_dataset,
        preprocessing=preprocessing,
        num_steps=rollout_steps,
    )
    output_path = save_rollout_predictions(FLAGS.prediction_save_path, predictions)
    logger.success(f"Saved {len(predictions)} rollout steps to {output_path}")


if __name__ == "__main__":
    app.run(main)
