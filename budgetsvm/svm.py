import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import accuracy_score

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from budgetsvm import kernel
from budgetsvm import optimization as opt


class SVC(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1, kernel=kernel.GaussianKernel(), budget=None):
        self.C = C
        self.kernel = kernel
        self.budget = budget

    def __repr__(self):
        repr = 'SVM('
        if self.C != 1:
            repr += f'C={self.C}, '

        if self.kernel != kernel.GaussianKernel():
            repr += f'kernel={self.kernel}, '

        if self.budget is not None:
            repr += f'budget={self.budget}, '

        if repr[-2:] == ', ':
            repr = repr[:-2]

        return repr + ')'

    def fit(self, X, y, warn=False):
        X, y = check_X_y(X, y)

        self.classes_, y = np.unique(y, return_inverse=True)
        if len(self.classes_) == 1:
            raise ValueError("Classifier can't train when only one class is present.")
        if len(self.classes_) > 2:
            raise ValueError("Classifier can't train when more than two classes are present.")

        self._encode_label = np.vectorize(lambda original: -1 if original == 0 else 1)
        self._decode_label = np.vectorize(lambda encoded: self.classes_[0] if encoded == -1 else self.classes_[1])
        y = self._encode_label(y)

        solver = opt.GurobiSolver()
        alpha = solver.solve(X, y, C=self.C, kernel=self.kernel, budget=self.budget)

        sv_mask = (0 < alpha) & (alpha < self.C)

        self.alpha_ = alpha[sv_mask]
        self.X_ = X[sv_mask]
        self.y_ = y[sv_mask]

        def dotprod(x_new):
            return np.sum([a * y_i * self.kernel.compute(x, x_new) for x, y_i, a in zip(self.X_, self.y_, self.alpha_)])

        bs = [y_i - dotprod(x) for x, y_i, a in zip(self.X_, self.y_, self.alpha_)]
        if not bs:
            raise FitFailedWarning('no SV founds')
        b = np.mean(bs)
        if warn and np.std(bs) > 1E-4:
            print('warning: computed values for b are', bs)

        self.df = lambda x: dotprod(x) + b
        return self

    def decision_function(self, X):
        return np.array([self.df(x) for x in X])

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        encoded_label = np.sign(self.decision_function(X))
        return self._decode_label(encoded_label)

    def score(self, X, y, **kwargs):
        return accuracy_score(self.predict(X), y)


if __name__ == '__main__':
    # print('lanciato script')
    # from sklearn.model_selection import GridSearchCV

    # c_values = [0.1, 1, 10]
    # kernel_values = [kernel.GaussianKernel(0.1), kernel.GaussianKernel(3)]
    # grid = {'C': c_values, 'kernel': kernel_values}

    # X = np.random.uniform(low=0, high=1, size=(50, 2))
    # labeling_funct = lambda x: 1 if np.dot([-1, 1], x) > 0 else -1
    # y = np.array(list(map(labeling_funct, X)))

    # print('classic SVM model')
    # gs = GridSearchCV(SVC(), grid, cv=2)
    # gs.fit(X, y)
    # model = gs.best_estimator_
    # print(model)

    # print('budgeted SVM model')
    # gs = GridSearchCV(SVC(budget=40), grid, cv=2)
    # gs.fit(X, y)
    # model = gs.best_estimator_
    # print(model)
    # print(sum(model.alpha > 0))

    from kernel import LinearKernel


    def generate_data(n=10, split=0.5):
        y = np.random.uniform(size=n)
        n_pos = int(n * split)
        y_pos = y[:n_pos]
        y_neg = y[n_pos:]
        return np.array([[0.2, y] for y in y_pos] + \
                        [[0.8, y] for y in y_neg]), \
            [1] * n_pos + [-1] * (n - n_pos)


    X, y = generate_data(n=100)

    svc = SVC(C=10, kernel=LinearKernel(), budget=None)
    svc.fit(X, y)
    print(f'classic SVM used {len(svc.alpha_)} SVs.')

    svc = SVC(C=10, kernel=LinearKernel(), budget=3)
    svc.fit(X, y)
    print(f'budget SVM used {len(svc.alpha_)} SVs.')
