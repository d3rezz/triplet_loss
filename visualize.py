"""TSNE Visualization

Extracts embeddings for the test set and creates a ProjectorConfig to visualize with Tensorboard.

Example:
        $ python visualize.py --experiment_dir=experiments/batchhard/
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os, shutil

from data import mnist_reader
import fashion_mnist_dataset
from model import model_fn
from utils import Params, get_sprite_image


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('experiment_dir', None, 'path to experiment directory containing a params.json file with hyperparameters.\
                                    output of the experiment will be written here.')
tf.app.flags.DEFINE_string('checkpoint', None, 'Checkpoint to inialize weights from (optional).')
tf.app.flags.mark_flag_as_required('experiment_dir')




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
    dataset = fashion_mnist_dataset.FashionMnistDataset(params)
    tf.logging.info("Computing embeddings")
    predictions = estimator.predict(lambda: dataset.test_dataset_fn(params), checkpoint_path=FLAGS.checkpoint)

    embeddings = np.zeros((dataset.test_size, params.embedding_size))
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']

    tf.logging.info("Embeddings shape: {}".format(embeddings.shape))


    # Tensorboard Projector
    viz_dir=os.path.join(FLAGS.experiment_dir, 'visualization/')
    if os.path.exists(viz_dir):
        shutil.rmtree(viz_dir)
    os.makedirs(viz_dir)

    embedding_var = tf.Variable(embeddings, name='fashion_mnist_embedding')

    # Add labels of test set to metadata to help with visualizations
    images, labels = mnist_reader.load_mnist(fashion_mnist_dataset.FASHION_MNIST_DIR, 't10k')   #TODO Not ideal to reload dataset, but useful here to get labels.tsv and sprite.png without evaluating test_dataset_fn
    np.savetxt(os.path.join(viz_dir,'labels.tsv'), np.array(labels))
    tf.logging.info("Computing sprite image")
    plt.imsave(os.path.join(viz_dir, 'zalando-mnist-sprite.png'), get_sprite_image(images), cmap='gray')

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.sprite.image_path = 'zalando-mnist-sprite.png'
    embedding.sprite.single_image_dim.extend([28, 28])      # Specify the width and height of a single thumbnail.
    embedding.metadata_path = 'labels.tsv' 

    summary_writer = tf.summary.FileWriter(viz_dir)

    # The next line writes a projector_config.pbtxt in the viz_dir. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(viz_dir, 'embeddings.ckpt'))

if __name__ == '__main__':
    tf.app.run(main)


