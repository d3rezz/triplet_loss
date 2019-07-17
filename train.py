"""Train a CNN with Triplet Loss

Example:
        $ python train.py --experiment_dir=experiments/batchall/
"""


import tensorflow as tf
import json
from absl import flags
import os

from model import model_fn
import fashion_mnist_dataset
from utils import Params

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('experiment_dir', None, 'path to experiment directory containing a params.json file with hyperparameters.\
                                    output of the experiment will be written here.')
tf.app.flags.mark_flag_as_required('experiment_dir')

def main(argv):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    #load params
    params = Params(os.path.join(FLAGS.experiment_dir, "params.json"))
    
    #create model
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=FLAGS.experiment_dir,     #directory where model parameters, graph, etc are saved.
                                    save_summary_steps=params.save_summary_steps,
                                    keep_checkpoint_max=params.keep_checkpoint_max,
                                    save_checkpoints_secs=params.save_checkpoints_secs)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} steps(s).".format(params.num_steps))

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: fashion_mnist_dataset.train_dataset_fn(params), max_steps=params.num_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: fashion_mnist_dataset.val_dataset_fn(params), start_delay_secs=params.start_delay_secs, throttle_secs=params.throttle_secs)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run(main)
