"""
Implementation of a Naive Bayes Classifier for numeric features.
"""

import math
import numpy as np
from scipy.stats import norm

class NaiveBayesClassifier:
    """Naive Bayes Classifier implementation.

    Uses a normal distribution by default, and a bernoulli distribution if any
    value other than "normal" is supplied for the "dist" parameter.
    """

    def __init__(self, dist="normal"):
        self.classes = set()
        self.probs = np.zeros(shape=(1,))
        self.params = np.zeros(shape=(1,))
        self.dist = dist

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fits a Naive Bayes Classifier to the provided training data.

        All features must be numeric.

        Args:
            features: A 2D np.ndarray of numeric feature values.
            labels: A 1D np.ndarray of classification labels.
        """
        # Find the overall probability of each label.
        label, freq = np.unique(labels, return_counts=True)
        self.classes = label
        self.probs = [f / labels.shape[0] for f in freq]

        if self.dist == "normal":
            self.params = np.zeros(shape=(len(self.classes),
                                          features.shape[1], 2))
        else:
            self.params = np.zeros(shape=(len(self.classes), features.shape[1]))
        for f in range(features.shape[1]):
            feature = features[:, f]
            for label_class in range(len(self.classes)):
                vals_from_class = feature[labels == self.classes[label_class]]
                if self.dist == "normal":
                    mu, sigma = norm.fit(vals_from_class)
                    self.params[label_class, f, 0] = mu
                    self.params[label_class, f, 1] = sigma
                else:
                    frac_nonzero = ((np.count_nonzero(vals_from_class) + 1) /
                                    len(vals_from_class))
                    self.params[label_class, f] = frac_nonzero


    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicts labels for given test features.

        Args:
            features: A 2D np.ndarray of numeric feature values.

        Returns:
            A 1D np.ndarray of predicted labels.
        """
        log_probs = np.zeros(
            shape=(features.shape[0], len(self.classes))
        )
        for c in range(len(self.classes)):
            log_probs[:, c] = math.log(self.probs[c])
        for c in range(self.params.shape[0]):
            for f in range(self.params.shape[1]):
                if self.dist == "normal":
                    mu = self.params[c, f, 0]
                    sigma = self.params[c, f, 1]
                    if sigma == 0:
                        sigma = 0.1
                    log_probs[:, c] += np.log(norm.pdf(features[:, f],
                                                       mu,
                                                       sigma))
                else:
                    p = self.params[c, f]
                    log_probs[features[:, f] > 0, c] += math.log(p)
                    log_probs[features[:, f] == 0, c] += math.log(1 - p)

        class_indices = np.argmax(log_probs, axis=1)
        return np.array([self.classes[idx] for idx in class_indices])
