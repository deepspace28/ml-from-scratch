"""
CART decision tree from scratch (classification, Gini impurity).
Teaches: how a tree greedily finds splits, what Gini impurity measures,
and why recursion + a stopping rule is all you need.

A decision tree is just paired questions. Each node asks "is feature j
> threshold t?" and splits the data. "Training" = searching over (j, t)
pairs for the split that lowers impurity the most.
"""

import numpy as np
from collections import Counter


def gini(y):
    """Gini impurity: chance a random label would be wrong if drawn twice."""
    n = len(y)
    if n == 0:
        return 0.0
    counts = Counter(y)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "prediction")


def _build(X, y, depth, max_depth, min_samples):
    node = _Node()
    node.prediction = None

    # Leaf conditions: pure, too small, or too deep -> just vote
    if depth >= max_depth or len(y) < min_samples or gini(y) == 0.0:
        node.prediction = Counter(y).most_common(1)[0][0]
        return node

    n, d = X.shape
    best = (gini(y), None, None)
    parent_imp = gini(y)

    for j in range(d):
        # candidate thresholds = midpoints between sorted unique values
        vals = np.unique(X[:, j])
        if len(vals) == 1:
            continue
        thresholds = (vals[:-1] + vals[1:]) / 2
        for t in thresholds:
            mask = X[:, j] <= t
            n_left = mask.sum()
            n_right = n - n_left
            if n_left == 0 or n_right == 0:
                continue
            # weighted impurity of the split
            imp = (n_left * gini(y[mask]) + n_right * gini(y[~mask])) / n
            if imp < best[0]:
                best = (imp, j, t)

    # No split improves -> leaf
    if best[1] is None or best[0] >= parent_imp:
        node.prediction = Counter(y).most_common(1)[0][0]
        return node

    _, j, t = best
    mask = X[:, j] <= t
    node.feature = j
    node.threshold = t
    node.left = _build(X[mask], y[mask], depth + 1, max_depth, min_samples)
    node.right = _build(X[~mask], y[~mask], depth + 1, max_depth, min_samples)
    return node


class DecisionTreeClassifier:
    def __init__(self, max_depth=4, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples

    def fit(self, X, y):
        self.tree = _build(
            np.asarray(X, dtype=float), np.asarray(y), 0, self.max_depth, self.min_samples
        )
        return self

    def _predict_one(self, x, node):
        if node.prediction is not None:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.tree) for x in X])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))
