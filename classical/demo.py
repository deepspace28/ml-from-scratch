"""
Demo: hand-drawn two-blob classification problem, solved three ways.
Teaches: knn memorizes the blobs; the tree draws one or two straight cuts
between them; both hit 100% on separated data - and wobble differently
when classes overlap.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from knn import KNNClassifier
from decision_tree import DecisionTreeClassifier


def make_blobs(n, center, label, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(center, 0.7, size=(n, 2))
    y = np.full(n, label)
    return X, y


def main():
    X0, y0 = make_blobs(60, (-2, -2), 0, seed=0)
    X1, y1 = make_blobs(60, (2, 2), 1, seed=1)
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    rng = np.random.default_rng(2)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.7 * len(y))
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

    knn = KNNClassifier(k=3).fit(X_tr, y_tr)
    tree = DecisionTreeClassifier(max_depth=4).fit(X_tr, y_tr)

    print(f"train/val sizes: {len(y_tr)}/{len(y_te)}")
    print(f"kNN (k=3)        test acc: {knn.score(X_te, y_te):.2%}")
    print(f"DecisionTree     test acc: {tree.score(X_te, y_te):.2%}")

    print("kNN predictions :", knn.predict(X_te[:10]).astype(int))
    print("Tree predictions:", tree.predict(X_te[:10]).astype(int))
    print("True labels     :", y_te[:10].astype(int))


if __name__ == "__main__":
    main()
