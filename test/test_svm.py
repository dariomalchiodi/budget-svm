import os
import pathlib
import pickle
import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError

from budgetsvm.svm import SVC


class SVCTests(unittest.TestCase):

    def test_trivial_classification_with_no_budget_works(self):
        """ Test normal SVC on a small linearly separable dataset. Expect 100% accuracy on train set. """
        X, y = make_blobs(n_samples=10, centers=2, n_features=2, center_box=(0, 10), random_state=42)
        y = [x if x == 1 else -1 for x in y]

        model = SVC(C=1)
        model.fit(X, y)
        y_hat = list(model.predict(X))

        self.assertListEqual(y, y_hat)

    def test_trivial_classification_with_budget_works(self):
        """ Test budget SVC on a small linearly separable dataset. Set high budget to ensure solution is found.
         Expect 100% accuracy on train set. """
        X, y = make_blobs(n_samples=10, centers=2, n_features=2, center_box=(0, 10), random_state=42)
        y = [x if x == 1 else -1 for x in y]

        model = SVC(C=1, budget=100)
        model.fit(X, y)
        y_hat = list(model.predict(X))

        self.assertListEqual(y, y_hat)

    def test_predict_before_fit_raise_error(self):
        self.assertRaises(NotFittedError, SVC().predict, np.array([0] * 10))

    def test_pickleable(self):
        X, y = make_blobs(n_samples=10, centers=2, n_features=2, center_box=(0, 10), random_state=42)
        model = SVC()
        model.fit(X, y)

        with open("delete.pkl", "wb") as f:
            pickle.dump(model, f)

        with open("delete.pkl", "rb") as f:
            _ = pickle.load(f)

        os.remove("delete.pkl")


if __name__ == '__main__':
    unittest.main()
