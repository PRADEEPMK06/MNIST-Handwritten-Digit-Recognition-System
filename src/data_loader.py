"""
data_loader.py
--------------
Loads MNIST (digits 0-9) and EMNIST-byclass (A-Z, uppercase only),
preprocesses both into a common 28x28 grayscale format, and merges them
into a single dataset with 36 classes.

    label 0-9   -> digits 0-9      (from MNIST)
    label 10-35 -> letters A-Z     (from EMNIST 'byclass' split, uppercase only)

MNIST downloads via Keras. EMNIST downloads via tensorflow-datasets (tfds),
which pulls from Google's reliable TFDS GCS bucket instead of the
often-broken original NIST mirror.

CodeAlpha ML Internship - Task 3: Handwritten Character Recognition
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds

IMG_SIZE = 28


def _load_mnist():
    """Load MNIST digits. Labels are already 0-9, no remapping needed."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train.astype(np.int64), x_test, y_test.astype(np.int64)


def _to_numpy(dataset, batch_size=10_000):
    """Materialize a tf.data pipeline to numpy in batches (fast, and far
    lighter on memory than pulling the whole filtered split at once)."""
    images, labels = [], []
    for img_batch, label_batch in tfds.as_numpy(dataset.batch(batch_size)):
        images.append(img_batch)
        labels.append(label_batch)
    return np.concatenate(images), np.concatenate(labels)


def _load_emnist_uppercase():
    """
    Load only the uppercase A-Z samples from EMNIST 'byclass'.

    We deliberately do NOT use the 'letters' split, even though it's the
    "obvious" choice for a letters-only task. EMNIST's own paper documents
    that 'letters' merges the uppercase and lowercase samples of every
    letter into one shared class, specifically to reduce case-confusion
    errors for classifiers that don't care about case. That's the wrong
    trade-off here: the app's canvas only ever produces uppercase strokes,
    but a large share of what the model learned under each label was
    lowercase handwriting - which for letters like A/a, G/g, Q/q, R/r
    looks nothing like the uppercase form. That mismatch between what the
    model was trained on and what users actually draw is what was causing
    wrong letter predictions, independent of the orientation/centering
    fixes already in place below.

    'byclass' keeps case separate: labels 0-9 are digits, 10-35 are
    uppercase A-Z, 36-61 are lowercase a-z. Filtering to 10-35 gives an
    uppercase-only training set - and those labels already line up with
    this project's own 0-9/10-35 scheme (see labels.py), so no relabeling
    is needed.
    """
    ds_train = tfds.load("emnist/byclass", split="train", as_supervised=True)
    ds_test = tfds.load("emnist/byclass", split="test", as_supervised=True)

    def is_uppercase(_image, label):
        return tf.logical_and(label >= 10, label <= 35)

    ds_train = ds_train.filter(is_uppercase)
    ds_test = ds_test.filter(is_uppercase)

    x_train, y_train = _to_numpy(ds_train)
    x_test, y_test = _to_numpy(ds_test)

    # Drop the channel dim tfds adds: (N, 28, 28, 1) -> (N, 28, 28)
    x_train = x_train.squeeze(-1)
    x_test = x_test.squeeze(-1)

    # EMNIST images (regardless of split) are stored transposed relative
    # to the usual MNIST orientation - this is a well-known quirk of the
    # original NIST byte format that tfds passes through unchanged. We fix
    # it here so every image in the combined dataset - and the live
    # drawings the user submits in the app - are oriented consistently.
    x_train = np.transpose(x_train, axes=(0, 2, 1))
    x_test = np.transpose(x_test, axes=(0, 2, 1))

    return x_train, y_train.astype(np.int64), x_test, y_test.astype(np.int64)


def load_combined_dataset(normalize=True, add_channel_dim=True):
    """
    Returns (x_train, y_train), (x_test, y_test) combining MNIST + EMNIST letters.

    x_* shape: (N, 28, 28, 1) float32 in [0, 1] if normalize=True
    y_* shape: (N,) int labels in [0, 35]
    """
    mx_train, my_train, mx_test, my_test = _load_mnist()
    ex_train, ey_train, ex_test, ey_test = _load_emnist_uppercase()

    x_train = np.concatenate([mx_train, ex_train], axis=0)
    y_train = np.concatenate([my_train, ey_train], axis=0)
    x_test = np.concatenate([mx_test, ex_test], axis=0)
    y_test = np.concatenate([my_test, ey_test], axis=0)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    if add_channel_dim:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_combined_dataset()
    print(f"Train set: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test set:  {x_test.shape}, labels: {y_test.shape}")
    print(f"Classes:   36 total (0-9 digits, 10-35 letters A-Z, uppercase only)")
