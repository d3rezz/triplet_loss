"""Module to load the Fashion Mnist Dataset into a tf.Dataset

Contains methods that return a train, validation or test tf.Dataset.
"""

import tensorflow as tf
import numpy as np
import glob, os

FASHION_MNIST_DIR = "data/"
SHUFFLE_SIZE = 100


def preprocess_image(image_path):
    image = tf.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.reshape(image, (28, 28, 1))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def train_dataset_fn(params):
    """Load training set of the Fashion MNIST dataset.
    Args:
        params: (dict) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    #load train images
    images_path = glob.glob(os.path.join(FASHION_MNIST_DIR, "train/", "*/*.png"))
    images_labels = [int(os.path.dirname(p).split('/')[-1]) for p in images_path]
    
    #create dataset object
    dataset = tf.data.Dataset.from_tensor_slices((images_path, images_labels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y))
    dataset = dataset.shuffle(SHUFFLE_SIZE)   
    dataset = dataset.repeat()
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to train
    return dataset


def val_dataset_fn(params):
    """Load training set of the Fashion MNIST dataset.
    Args:
        params: (dict) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    images_path = glob.glob(os.path.join(FASHION_MNIST_DIR, "val/", "*/*.png"))
    images_labels = [int(os.path.dirname(p).split('/')[-1]) for p in images_path]
    
    #create dataset object
    dataset = tf.data.Dataset.from_tensor_slices((images_path, images_labels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y))
    dataset = dataset.shuffle(SHUFFLE_SIZE)   
    dataset = dataset.repeat()
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)   # make sure you always have one batch ready to train
    return dataset



def test_dataset_fn(params):
    """Load test set of the Fashion MNIST dataset.
    Args:
        data_dir: (string) path to the data directory
        params: (dict) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    images_path = glob.glob(os.path.join(FASHION_MNIST_DIR, "test/", "*.png"))
    images_path = sorted(images_path, key=lambda x: int(os.path.basename(x).split('.')[0]))     #TODO fix this workaround to retrieve the test images later 

    #create dataset object
    dataset = tf.data.Dataset.from_tensor_slices((images_path))
    dataset = dataset.map(lambda x: preprocess_image(x))
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to test
    return dataset

