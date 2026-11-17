import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tree import DecisionTree, entropy, gini, mad_median, variance


# ---------- Step 1: criteria ----------

def test_gini_mixed():
    assert gini(np.array([[1., 0.], [0., 1.]])) == pytest.approx(0.5)


def test_gini_pure():
    assert gini(np.array([[1., 0.], [1., 0.]])) == pytest.approx(0.0)


def test_entropy_mixed():
    assert entropy(np.array([[1., 0.], [0., 1.]])) == pytest.approx(np.log(2), abs=1e-2)


def test_entropy_pure():
    assert entropy(np.array([[0., 1.], [0., 1.]])) == pytest.approx(0.0, abs=1e-2)


def test_variance():
    assert variance(np.array([[1.], [2.], [10.]])) == pytest.approx(146 / 9)


def test_mad_median():
    assert mad_median(np.array([[1.], [2.], [10.]])) == pytest.approx(3.0)


# ---------- Already done: make_split ----------

def test_make_split():
    X = np.ones((4, 5)) * np.arange(4)[:, None]
    y = np.arange(4)[:, None] + np.array([0.2, -0.3, 0.1, 0.4])[:, None]
    (X_l, y_l), (X_r, y_r) = DecisionTree().make_split(1, 1., X, y)
    assert np.array_equal(X_l, X[:1]) and np.array_equal(X_r, X[1:])
    assert np.array_equal(y_l, y[:1]) and np.array_equal(y_r, y[1:])


# ---------- Step 2: choose_best_split ----------

# feature 0 is noise, feature 1 separates classes perfectly
X_SEP = np.array([[5., 0.], [1., 1.], [4., 2.], [2., 10.], [3., 11.], [0., 12.]])
Y_SEP = np.array([[0], [0], [0], [1], [1], [1]])


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_best_split_classification(criterion):
    tree = DecisionTree(criterion_name=criterion, n_classes=2)
    tree.criterion, tree.classification = tree.all_criterions[criterion]  # what fit() sets
    y_oh = np.eye(2)[Y_SEP[:, 0]]
    f, t = tree.choose_best_split(X_SEP, y_oh)
    assert f == 1
    assert 2 < t <= 10  # any threshold separating {0,1,2} from {10,11,12}


@pytest.mark.parametrize("criterion", ["variance", "mad_median"])
def test_best_split_regression(criterion):
    y = np.array([[1.], [1.], [1.], [50.], [50.], [50.]])
    tree = DecisionTree(criterion_name=criterion)
    tree.criterion, tree.classification = tree.all_criterions[criterion]  # what fit() sets
    f, t = tree.choose_best_split(X_SEP, y)
    assert f == 1
    assert 2 < t <= 10


# ---------- Step 3: make_tree (stop conditions, leaves) ----------

def test_fits_separable_data_perfectly():
    tree = DecisionTree(criterion_name="gini")
    tree.fit(X_SEP, Y_SEP)
    assert np.array_equal(np.asarray(tree.predict(X_SEP)).reshape(-1), Y_SEP[:, 0])


def test_unsplittable_data_does_not_hang():
    # identical rows -> no valid split, must become a leaf
    X = np.zeros((4, 2))
    y = np.array([[0], [1], [1], [0]])
    DecisionTree().fit(X, y)


def test_max_depth_one_gives_at_most_two_leaves():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = rng.normal(size=(100, 1))
    tree = DecisionTree(max_depth=1, criterion_name="variance")
    tree.fit(X, y)
    assert len(np.unique(tree.predict(X))) <= 2


def test_min_samples_split_blocks_split():
    X = np.array([[0.], [1.]])
    y = np.array([[0], [1]])
    tree = DecisionTree(min_samples_split=3)
    tree.fit(X, y)
    # root stays a leaf -> both get same class
    assert len(np.unique(tree.predict(X))) == 1


def test_refit_starts_from_scratch():
    rng = np.random.default_rng(1)
    XA, yA = rng.normal(size=(50, 2)), rng.integers(0, 2, size=(50, 1))
    XB, yB = rng.normal(size=(50, 2)), rng.integers(0, 2, size=(50, 1))
    reused = DecisionTree(max_depth=3)
    reused.fit(XA, yA)
    reused.fit(XB, yB)
    fresh = DecisionTree(max_depth=3)
    fresh.fit(XB, yB)
    assert np.array_equal(reused.predict(XB), fresh.predict(XB))


def test_clone_works_for_gridsearch():
    tree = clone(DecisionTree(max_depth=4, criterion_name="entropy"))
    assert tree.max_depth == 4 and tree.criterion_name == "entropy"


# ---------- Step 4: predict / predict_proba ----------

def test_predict_shape():
    tree = DecisionTree()
    tree.fit(X_SEP, Y_SEP)
    assert tree.predict(X_SEP).shape == (len(X_SEP), 1)


def test_predict_tie_picks_smallest_class():
    X = np.zeros((2, 1))
    y = np.array([[1], [0]])
    tree = DecisionTree()
    tree.fit(X, y)
    assert tree.predict(X)[0, 0] == 0


def test_leaf_value_variance_is_mean():
    X = np.zeros((3, 1))
    y = np.array([[1.], [2.], [10.]])
    tree = DecisionTree(criterion_name="variance")
    tree.fit(X, y)
    assert tree.predict(X)[0, 0] == pytest.approx(13 / 3)


def test_leaf_value_mad_median_is_median():
    X = np.zeros((3, 1))
    y = np.array([[1.], [2.], [10.]])
    tree = DecisionTree(criterion_name="mad_median")
    tree.fit(X, y)
    assert tree.predict(X)[0, 0] == pytest.approx(2.0)


def test_predict_proba():
    X = np.zeros((4, 1))
    y = np.array([[0], [1], [1], [2]])
    tree = DecisionTree()
    tree.fit(X, y)
    proba = tree.predict_proba(X)
    assert proba.shape == (4, 3)
    assert np.allclose(proba, [0.25, 0.5, 0.25])


# ---------- Step 5: real data (same as notebook asserts) ----------

def test_digits_accuracy():
    data = load_digits()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target[:, None], test_size=0.2, random_state=42)
    tree = DecisionTree(max_depth=10, criterion_name="gini")
    tree.fit(X_tr, y_tr)
    assert 0.84 < accuracy_score(y_te, tree.predict(X_te)) < 0.9
