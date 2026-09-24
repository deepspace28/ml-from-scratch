import numpy as np
import pytest

from knn import KNNClassifier
from decision_tree import DecisionTreeClassifier, gini


def make_blob(rng, center, n=40):
    return rng.normal(center, 0.5, size=(n, 2))


def make_dataset(seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([make_blob(rng, (-2, -2)), make_blob(rng, (2, 2))])
    y = np.array([0] * 40 + [1] * 40)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def test_gini_impure_pure_empty():
    assert gini(np.array([1, 1, 1])) == 0.0
    assert abs(gini(np.array([0, 1])) - 0.5) < 1e-9
    assert gini(np.array([], dtype=int)) == 0.0


def test_knn_fits_separable_blobs():
    X, y = make_dataset()
    X_tr, y_tr, X_te, y_te = X[:-16], y[:-16], X[-16:], y[-16:]
    acc = KNNClassifier(k=3).fit(X_tr, y_tr).score(X_te, y_te)
    assert acc == 1.0


def test_tree_fits_separable_blobs():
    X, y = make_dataset()
    X_tr, y_tr, X_te, y_te = X[:-16], y[:-16], X[-16:], y[-16:]
    acc = DecisionTreeClassifier(max_depth=4).fit(X_tr, y_tr).score(X_te, y_te)
    assert acc >= 0.9


def test_tree_depth_zero_is_majority():
    X, y = make_dataset()  # 50/50 classes
    tree = DecisionTreeClassifier(max_depth=0).fit(X, y)
    preds = tree.predict(X)
    assert len(set(preds.tolist())) == 1


def test_knn_k_equals_one_memoizes_training_point():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    knn = KNNClassifier(k=1).fit(X, y)
    assert knn.predict([[0.001, 0.0]])[0] == 0
    assert knn.predict([[0.999, 1.0]])[0] == 1


def test_tree_predicts_new_point_between_classes():
    X, y = make_dataset()
    tree = DecisionTreeClassifier(max_depth=4).fit(X, y)
    p = tree.predict(np.array([[0.0, 0.0]]))[0]  # midpoint: either is fine
    assert p in (0, 1)
