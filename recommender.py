"""Clothing Recommender System

Finds the most similar clothing items to the query item and displays them.

Example:
        $ python recommender.py --experiment_dir=experiments/batchall/ 
        --checkpoint=experiments/batchall/model.ckpt-22114
        --query=data/test/1.png

Note: The sklearn.neighbours implementation can be used instead 
to quickly find the nearest neighbour in the embedding space.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os, shutil
import cv2

from data import mnist_reader
import fashion_mnist_dataset
from model import model_fn
from utils import Params


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('experiment_dir', None, 'path to experiment directory containing a params.json file with hyperparameters.\
                                    output of the experiment will be written here.')
tf.app.flags.DEFINE_string('checkpoint', None, 'Checkpoint to inialize weights from (optional).')
tf.app.flags.DEFINE_integer('query', None, 'Index of item for which to find similar items.')
tf.app.flags.mark_flag_as_required('experiment_dir')
tf.app.flags.mark_flag_as_required('query')



def main(argv):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    #load params
    params = Params(os.path.join(FLAGS.experiment_dir, "params.json"))
    
    # Load model and compute embeddings
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=FLAGS.experiment_dir,
                                        save_summary_steps=params.save_summary_steps)

    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Compute embeddings on the test set
    tf.logging.info("Computing embeddings")
    predictions = estimator.predict(lambda: fashion_mnist_dataset.test_dataset_fn(params), checkpoint_path=FLAGS.checkpoint)

    embeddings = np.zeros((10000, params.embedding_size))   #TODO remove hardcoded value
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']

    tf.logging.info("Embeddings shape: {}".format(embeddings.shape))

    # Get 10 closest examples to query item
    num_results = 10
    query_embedding = embeddings[np.newaxis, FLAGS.query, :]
    
    tf.logging.info("Query embedding shape: {}".format(query_embedding.shape))
    distances = np.sqrt(np.sum(np.square(query_embedding-embeddings), axis=1)) # Compute distances to all other items
    tf.logging.info("Distances shape: {}".format(distances.shape))

    # Get most similar items, ignore closest item as it is the query (distance=0)
    closest_idxs = np.argsort(distances)[1:]
    sorted_distances = np.sort(distances)[1:]
    print("{} closest distances:\n{}".format(num_results, sorted_distances[:num_results]))

    # Display query image and 10 most similar results
    f, ax = plt.subplots(1, num_results+1, figsize=(8,2))
    query_image = cv2.imread(os.path.join(fashion_mnist_dataset.FASHION_MNIST_DIR, "test/", "{}.png".format(FLAGS.query)), 0)
    ax[0].imshow(query_image, cmap="gray")
    ax[0].title.set_text("Query")
    ax[0].axis('off')

    for i in range(num_results):
        current_image = cv2.imread(os.path.join(fashion_mnist_dataset.FASHION_MNIST_DIR, "test/", "{}.png".format(closest_idxs[i])), 0)
        ax[i+1].imshow(current_image, cmap='gray')
        ax[i+1].title.set_text("{:.3f}".format(sorted_distances[i]))
        ax[i+1].axis('off')

    plt.show()

if __name__ == '__main__':
    tf.app.run(main)


