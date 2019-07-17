import os, shutil
import gzip
import mnist_reader
import matplotlib.pyplot as plt
from mnist_reader import load_mnist
import numpy as np
from sklearn.model_selection import train_test_split

FASHION_MNIST_DIR = "fashion/"
TRAIN_SPLIT = 0.8

# Create training and validation splits
images, labels = load_mnist(FASHION_MNIST_DIR, 'train')
images = np.reshape(images, (-1, 28, 28))   #reshape
images = 255 - images   #invert colors
unique_labels = set(labels.tolist())

train_images, val_images, train_labels, val_labels = train_test_split(images, labels, train_size=TRAIN_SPLIT)

os.mkdir("train/")
os.mkdir("val/")
for label in unique_labels:
    os.mkdir("train/{}/".format(str(label)))
    os.mkdir("val/{}/".format(str(label)))

for i in range(train_images.shape[0]):
    plt.imsave("train/{}/{}.png".format(str(train_labels[i]), i), train_images[i,:,:], cmap="gray")

for i in range(val_images.shape[0]):
    plt.imsave("val/{}/{}.png".format(str(val_labels[i]), i), val_images[i,:,:], cmap="gray")


# Create test split
test_images, test_labels = load_mnist(FASHION_MNIST_DIR, 't10k')
test_images = np.reshape(test_images, (-1, 28, 28))   #reshape
test_images = 255 - test_images   #invert colors
unique_labels = set(test_labels.tolist())

os.mkdir("test/")

for i in range(test_images.shape[0]):
    plt.imsave("test/{}.png".format(test_images[i,:,:]), cmap="gray")