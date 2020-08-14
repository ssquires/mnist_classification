"""
Using the MNIST handwritten digits dataset, test four different naive Bayes
classifiers: GaussianNB and BernoulliNB from sklearn, and a custom
implementation using gaussian and normal distributions (see
naive_bayes_classifier.py).
"""
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from tensorflow import keras

import numpy as np
from scipy import ndimage

from naive_bayes_classifier import NaiveBayesClassifier

def threshold(images: np.ndarray, thresh: float) -> np.ndarray:
    """Thresholds all values in an array to 0.0 or 1.0, using given threshold.

    Args:
        images: An np.ndarray to threshold
        thresh: The threshold to use. All values >= thresh will be set to 1.0,
            and all values < thresh to 0.0.

    Returns:
        An np.ndarray of the same shape as images, with all elements set to 0.0
        or 1.0.
    """
    thresholded_images = np.zeros(shape=images.shape)
    thresholded_images[images >= thresh] = 1.0
    thresholded_images[images < thresh] = 0.0
    return thresholded_images

def crop_and_rescale(images: np.ndarray,
                     scale: int,
                     thresh: float) -> np.ndarray:
    """Crops images to boundng box, resizes to scale x scale, and thresholds.

    Args:
        images: a 3D np.ndarray
        scale: the width and height of the rescaled images
        thresh: the threshold to use on the rescaled images

    Returns:
        A 3D np.ndarray of cropped, rescaled, and thresholded images.
    """
    row_range = range(images[0].shape[0])
    col_range = range(images[0].shape[1])
    cc, rr = np.meshgrid(row_range, col_range)
    cropped_images = np.zeros(shape=(images.shape[0], scale, scale))
    for i, original_img in enumerate(images):
        rows = original_img * rr
        cols = original_img * cc
        row_min = int((rows + (1 - original_img) * 100).min())
        row_max = int(rows.max()) + 1
        col_min = int((cols + (1 - original_img) * 100).min())
        col_max = int(cols.max()) + 1
        cropped_img = original_img[row_min:row_max, col_min:col_max]
        cropped_images[i, :, :] = ndimage.zoom(
            cropped_img,
            (scale / cropped_img.shape[0], scale / cropped_img.shape[1])
        )
    return threshold(cropped_images, thresh)

def fit_and_test(model,
                 train_images: np.ndarray,
                 train_labels: np.ndarray,
                 test_images: np.ndarray,
                 test_labels: np.ndarray) -> None:
    """Fit and test a classifier on data.
    """
    model.fit(train_images, train_labels)
    predicted_labels = model.predict(test_images)
    print("Mislabeled", (predicted_labels != test_labels).sum(),
          "out of", len(test_images))

def main():
    """Load MNIST data and train/test four Naive Bayes classifiers on it.
    """
    # Load train and test MNIST data
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Convert train and test images to [0, 1] range and threshold
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = threshold(train_images, 0.5)
    test_images = threshold(test_images, 0.5)

    # Crop images to bounding box around digit and rescale to 20x20
    train_images = crop_and_rescale(train_images, 20, 0.5)
    test_images = crop_and_rescale(test_images, 20, 0.5)

    # Flatten each image to 1D
    train_images = train_images.reshape(len(train_images),
                                        len(train_images[0]) ** 2)
    test_images = test_images.reshape(len(test_images),
                                      len(test_images[0]) ** 2)

    # Fit and test four different classifiers on MNIST data.
    sklearn_gaussian = GaussianNB()
    sklearn_bernoulli = BernoulliNB()
    custom_gaussian = NaiveBayesClassifier()
    custom_bernoulli = NaiveBayesClassifier(dist="bernoulli")
    fit_and_test(sklearn_gaussian,
                 train_images,
                 train_labels,
                 test_images,
                 test_labels)
    fit_and_test(sklearn_bernoulli,
                 train_images,
                 train_labels,
                 test_images,
                 test_labels)
    fit_and_test(custom_gaussian,
                 train_images,
                 train_labels,
                 test_images,
                 test_labels)
    fit_and_test(custom_bernoulli,
                 train_images,
                 train_labels,
                 test_images,
                 test_labels)

if __name__ == "__main__":
    main()
