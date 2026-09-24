"""
k-Nearest Neighbors from scratch - no sklearn, just numpy + a distance function.
Teaches: lazy learning (there is no "training" step), and how distance +
majority vote is a complete classifier.

The whole trick: keep the training set, and for a new point find the k
closest points and let them vote.
"""

import numpy as np
from collections import Counter


def euclidean(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


class KNNClassifier:
    def __init__(self, k=3):
        assert k >= 1
        self.k = k

    def fit(self, X, y):
        # "Training" = memorize. That's the joke and the point.
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        return self

    def predict_one(self, x):
        # distance to every training point, then take the k smallest
        dists = [euclidean(row, x) for row in self.X]
        k_idx = np.argsort(dists)[: self.k]
        votes = self.y[k_idx]
        return Counter(votes).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.predict_one(np.asarray(x, dtype=float)) for x in np.asarray(X, dtype=float)])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))
