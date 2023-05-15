import itertools
import os
import pathlib
import pickle
import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError, FitFailedWarning
from sklearn.model_selection import train_test_split

from budgetsvm.svm import SVC
from kernel import GaussianKernel, PrecomputedKernel


class SVCTests(unittest.TestCase):
    def test_trivial_classification_with_no_budget_works(self):
        """Test normal SVC on a small linearly separable dataset. Expect 100% accuracy on train set."""
        X, y = make_blobs(
            n_samples=10, centers=2, n_features=2, center_box=(0, 10), random_state=42
        )
        y = [x if x == 1 else -1 for x in y]

        model = SVC(C=1)
        model.fit(X, y)
        y_hat = list(model.predict(X))

        self.assertListEqual(y, y_hat)

    def test_trivial_classification_with_budget_works(self):
        """Test budget SVC on a small linearly separable dataset. Set high budget to ensure solution is found.
        Expect 100% accuracy on train set."""
        X, y = make_blobs(
            n_samples=10, centers=2, n_features=2, center_box=(0, 10), random_state=42
        )
        y = [x if x == 1 else -1 for x in y]

        model = SVC(C=1, budget=100)
        model.fit(X, y)
        y_hat = list(model.predict(X))

        self.assertListEqual(y, y_hat)

    def test_predict_before_fit_raise_error(self):
        self.assertRaises(NotFittedError, SVC().predict, np.array([0] * 10))

    def test_pickleable(self):
        X, y = make_blobs(
            n_samples=10, centers=2, n_features=2, center_box=(0, 10), random_state=42
        )
        model = SVC()
        model.fit(X, y)

        with open("delete.pkl", "wb") as f:
            pickle.dump(model, f)

        with open("delete.pkl", "rb") as f:
            _ = pickle.load(f)

        os.remove("delete.pkl")

    def test_fit_with_precomputed_kernels_checks_for_square_matrix(self):
        X, y = make_blobs(
            n_samples=100, centers=2, n_features=2, center_box=(0, 2), random_state=42
        )
        y = [x if x == 1 else -1 for x in y]

        model = SVC(C=1, kernel=PrecomputedKernel(None))

        # exception on malformed input
        self.assertRaises(ValueError, model.fit, np.empty((10, 3)), y)
        self.assertRaises(ValueError, model.fit, np.empty((len(y), 0)), y)
        self.assertRaises(ValueError, model.fit, np.empty((len(y), len(y) - 1)), y)

        # no validation exception on correct shape input
        precomp = X @ X.T
        model.fit(precomp, y)

    def test_predict_with_precomputed_kernels_checks_for_right_size_matrix(self):
        X, y = make_blobs(
            n_samples=100, centers=2, n_features=2, center_box=(0, 2), random_state=42
        )
        y = [x if x == 1 else -1 for x in y]

        model = SVC(C=1, kernel=PrecomputedKernel(None))
        precomp = X @ X.T
        model.fit(precomp, y)

        # precomp kernel fit expect a matrix of size n_samples X n_support_vectors
        self.assertRaises(ValueError, model.predict, np.empty((10, 3)))
        self.assertRaises(ValueError, model.predict, np.empty((0, 0)))
        self.assertRaises(ValueError, model.predict, np.empty((len(y), len(y) - 1)))

        model.predict(np.empty((3, model.train_set_size_)))

    def test_precomputed_kernel_model_same_as_non_precomputed_kernel_model(self):
        X, y = make_blobs(
            n_samples=100, centers=2, n_features=2, center_box=(0, 2), random_state=42
        )
        y = [x if x == 1 else -1 for x in y]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        kernel = GaussianKernel(sigma=0.1)

        model = SVC(C=1, kernel=kernel)
        model.fit(X_train, y_train)
        y_hat = list(model.predict(X_test))

        # accuracy
        #print([y == yhat for y, yhat in zip(y_test, y_hat)].count(True) / len(y_test))

        model = SVC(C=1, kernel=PrecomputedKernel(None))
        precomputed_train_kernel_values = np.array(
            [[kernel.compute(x, y) for y in X_train] for x in X_train]
        )
        model.fit(precomputed_train_kernel_values, y_train)

        precomputed_test_kernel_values = np.array(
            [[kernel.compute(x, y) for y in X_train] for x in X_test]
        )
        y_hat_precomputed = list(model.predict(precomputed_test_kernel_values))

        self.assertListEqual(y_hat, y_hat_precomputed)


if __name__ == "__main__":
    unittest.main()
