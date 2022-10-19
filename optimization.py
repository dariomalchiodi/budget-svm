
import numpy as np
import itertools as it
from collections.abc import Iterable
import logging
import os

logger = logging.getLogger(__name__)

from gurobipy import LinExpr, GRB, Model, Env, QuadExpr, GurobiError

def clip(x, left, right, tolerance=10**-5):
    clipped = x if abs(x - left) >= tolerance else left
    if right < np.inf:
        clipped = clipped if abs(x - right) >= tolerance else right
    return clipped

class Solver:
    """Abstract solver for optimization problems.

    The base class for solvers is :class:`Solver`: it exposes a method
    `solve` which delegates the numerical optimization process to an abstract
    method `solve_problem` and subsequently clips the results to the boundaries
    of the feasible region.
    """

    def solve_problem(self, *args):
        pass

    def solve(self, X, y, C, kernel, epsilon, budget=None):
        """Solve optimization phase.

        Build and solve the constrained optimization problem on the basis
        of the fuzzy learning procedure.

        :param X: Objects in training set.
        :type X: iterable
        :param y: Membership values for the objects in `xs`.
        :type y: iterable
        :param C: constant managing the trade-off in joint radius/error
          optimization.
        :type C: float
        :param kernel: Kernel function to be used.
        :type kernel: :class:`mulearn.kernel.Kernel`
        :raises: ValueError if C is non-positive or if xs and mus have
          different lengths.
        :returns: `list` -- optimal values for the independent variables
          of the problem."""

        if C <= 0:
            raise ValueError('C should be positive')

        y = np.array(y)
        solution = self.solve_problem(X, y, C, kernel, epsilon, budget)

        if budget is None:
            alpha, alpha_hat = solution
        else:
            alpha, alpha_hat, gamma = solution

        if budget is None:
            alpha_clipped = np.array([clip(a, 0, C) for a in alpha])
            alpha_hat_clipped = np.array([clip(a, 0, C) for a in alpha_hat])
        else:
            alpha_clipped = np.array([clip(a, 0, np.inf) for a in alpha])
            alpha_hat_clipped = np.array([clip(a, 0, np.inf) for a in alpha_hat])
            gamma_clipped = clip(gamma, 0, np.inf)

        if budget is None:
            optimal_values = alpha_clipped, alpha_hat_clipped
        else:
            optimal_values = alpha_clipped, alpha_hat_clipped, gamma_clipped

        return optimal_values


class GurobiSolver(Solver):
    """Solver based on gurobi.

    Using this class requires that gurobi is installed and activated
    with a software key. The library is available at no cost for academic
    purposes (see
    https://www.gurobi.com/downloads/end-user-license-agreement-academic/).
    Alongside the library, also its interface to python should be installed,
    via the gurobipy package.
    """

    def __init__(self, time_limit=10*60, initial_values=None):
        """
        Build an object of type GurobiSolver.

        :param time_limit: Maximum time (in seconds) before stopping iterative
          optimization, defaults to 10*60.
        :type time_limit: int
        :param initial_values: Initial values for variables of the optimization
          problem, defaults to None.
        :type initial_values: iterable of floats or None
        """
        self.time_limit = time_limit
        self.initial_values = initial_values

    def solve_problem(self, X, y, C, kernel, epsilon, budget=None):
        """Optimize via gurobi.

        Build and solve the constrained optimization problem at the basis
        of the fuzzy learning procedure using the gurobi API.

        :param X: Objects in training set.
        :type X: iterable
        :param y: Membership values for the objects in `xs`.
        :type y: iterable
        :param C: constant managing the trade-off in joint radius/error
          optimization.
        :type C: float
        :param kernel: Kernel function to be used.
        :type kernel: :class:`mulearn.kernel.Kernel`
        :raises: ValueError if C is non-positive or if xs and mus have
          different lengths.
        :returns: `list` -- optimal values for the independent variables
          of the problem."""

        m = len(X)

        with Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with Model('svr', env=env) as model:
                model.setParam('OutputFlag', 0)
                model.setParam('TimeLimit', self.time_limit)

                for i in range(m):
                    if C < np.inf and budget is None:
                        model.addVar(name=f'alpha_{i}', lb=0, ub=C,
                                     vtype=GRB.CONTINUOUS)
                        model.addVar(name=f'alphahat_{i}', lb=0, ub=C,
                                     vtype=GRB.CONTINUOUS)
                    else:
                        model.addVar(name=f'alpha_{i}', lb=0,
                                     vtype=GRB.CONTINUOUS)
                        model.addVar(name=f'alphahat_{i}', lb=0,
                                     vtype=GRB.CONTINUOUS)
                if budget is not None:
                    model.addVar(name='gamma', lb=0, vtype=GRB.CONTINUOUS)

                model.update()
                vars = model.getVars()

                gamma = vars[-1]
                alpha = np.array(vars[:m])
                alpha_hat = np.array(vars[m:2*m])

                if self.initial_values is not None:
                    for a, i in zip(alpha, self.initial_values[0]):
                        a.start = i
                    for a, i in zip(alpha_hat, self.initial_values[1]):
                        a.start = i

                    if budget is not None:
                        gamma.start = self.initial_values[2]

                obj = QuadExpr()

                for a, a_h, y_ in zip(alpha, alpha_hat, y):
                    obj.add((a + a_h), epsilon)
                    obj.add(-(a - a_h), y_)
                    if budget is not None:
                        obj.add(gamma * budget)

                for i, j in it.product(range(m), range(m)):
                    obj.add((alpha[i] - alpha_hat[i]) * \
                            (alpha[j] - alpha_hat[j]),
                            kernel.compute(X[i], X[j]) * 0.5)

                model.setObjective(obj, GRB.MINIMIZE)

                constEqual = LinExpr()
                constEqual.add(sum(alpha - alpha_hat), 1.0)

                model.addLConstr(constEqual, GRB.EQUAL, 0)

                if budget is not None:
                    for a in alpha:
                        const = LinExpr()
                        const.add(a - gamma, 1.0)
                        model.addLConstr(const,GRB.LESS_EQUAL, C)
                    for a in alpha_hat:
                        const = LinExpr()
                        const.add(a - gamma, 1.0)
                        model.addLConstr(const,GRB.LESS_EQUAL, C)

                model.optimize()

                if model.Status != GRB.OPTIMAL:
                    raise ValueError('optimal solution not found! '
                                     f'status={model.Status}')

                alpha_opt = np.array([a.x for a in alpha])
                alpha_hat_opt = np.array([a.x for a in alpha_hat])

                if budget is not None:
                    gamma_opt = gamma.x

                solution = (alpha_opt, alpha_hat_opt) \
                    if budget is None else (alpha_opt, alpha_hat_opt, gamma_opt)

                return solution

    def __repr__(self):
        return f"GurobiSolver(time_limit={self.time_limit}, " + \
               f"initial_values={self.initial_values})"
