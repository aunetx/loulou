import os
import gzip
import numpy as np

DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
path = os.path.dirname(__file__)
path = os.path.join(path, "../data/")

# Download and import the MNIST dataset from Yann LeCun's website.
# Reserve 10,000 examples from the training set for validation.
# Each image is an array of 784 (28x28) float values  from 0 (white) to 1 (black).


def load_data(one_hot: bool = True, reshape: bool = None, validation_size: int = 10000) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    x_tr = load_images('train-images-idx3-ubyte.gz')
    y_tr = load_labels('train-labels-idx1-ubyte.gz')
    x_te = load_images('t10k-images-idx3-ubyte.gz')
    y_te = load_labels('t10k-labels-idx1-ubyte.gz')

    x_tr = x_tr[:-validation_size]
    y_tr = y_tr[:-validation_size]

    if one_hot:
        y_tr, y_te = [to_one_hot(y) for y in (y_tr, y_te)]

    if reshape:
        x_tr, x_te = [x.reshape(*reshape) for x in (x_tr, x_te)]

    return x_tr, y_tr, x_te, y_te


def load_images(filename: str) -> np.ndarray:
    maybe_download(filename)
    with gzip.open(path+filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28) / np.float32(256)


def load_labels(filename: str) -> np.ndarray:
    maybe_download(filename)
    with gzip.open(path+filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


# Download the file, unless it's already here.
def maybe_download(filename: str) -> None:
    if not os.path.exists(path+filename):
        print('Please wait while downloading training dataset.')
        from urllib.request import urlretrieve
        print("Downloading %s" % filename)
        urlretrieve(DATA_URL + filename, path+filename)


# Convert class labels from scalars to one-hot vectors.
def to_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    return np.eye(num_classes)[labels]
