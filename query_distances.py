"""Compute the similarity of 2 images

This module determines how similar 2 images are by computing
the Euclidean distance between their embeddings.

Example:
        $ python query_distances.py --experiment_dir=experiments/batchall/ 
        --checkpoint=experiments/batchall/model.ckpt-22114
        --image1=data/test/1.png --image2=data/test/2.png
"""

import tensorflow as tf
import json
from absl import flags
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

from model import build_model
import fashion_mnist_dataset
from data import mnist_reader
from utils import Params

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('experiment_dir', None, 'path to experiment directory \
                                    containing a params.json file with hyperparameters.\
                                    output of the experiment will be written here.')
tf.app.flags.DEFINE_string('checkpoint', None, 'Checkpoint to inialize weights from (optional).')
tf.app.flags.DEFINE_string('image1', None, 'Path to image 1.')
tf.app.flags.DEFINE_string('image2', None, 'Path to image 2.')


tf.app.flags.mark_flag_as_required('experiment_dir')
tf.app.flags.mark_flag_as_required('image1')
tf.app.flags.mark_flag_as_required('image2')

def main(argv):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # Load params
    params = Params(os.path.join(FLAGS.experiment_dir, "params.json"))
    
    # Load images
    image1 = cv2.imread(FLAGS.image1, 0)
    image2 = cv2.imread(FLAGS.image2, 0)

    image1 = image1[None, :, :, None]   #make it a batch of 1 and 1 channel
    image2 = image2[None, :, :, None]


    # Load model
    images = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    model = build_model(images, tf.estimator.ModeKeys.PREDICT, params)

    saver  = tf.train.Saver()
    # Extract embeddings
    with tf.Session() as sess:

        if FLAGS.checkpoint: # restore checkpoint
            saver.restore(sess, FLAGS.checkpoint)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.experiment_dir))

        embeddings1 = sess.run([model], feed_dict={images: image1})[0]
        embeddings2 = sess.run([model], feed_dict={images: image2})[0]

    # Compute distance between embeddings
    distance = np.linalg.norm(embeddings2 - embeddings1)

    print("Distance between the embeddings of the two images: ", distance)

if __name__ == '__main__':
    tf.app.run(main)
