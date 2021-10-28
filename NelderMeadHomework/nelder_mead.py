# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@'~~~     ~~~`@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@'                     `@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@'                           `@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@'                               `@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@'                                   `@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@'                                     `@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@'                                       `@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@                                         @@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@'                                         `@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@                                           @@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@                                           @@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@                       n,                  @@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@                     _/ | _                @@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@                    /'  `'/                @@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@a                 <~    .'                a@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@                 .'    |                 @@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@a              _/      |                a@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@a           _/      `.`.              a@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@a     ____/ '   \__ | |______       a@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@a__/___/      /__\ \ \     \___.a@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@/  (___.'\_______)\_|_|        \@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@|\________                       ~~~~~\@@@@@@@@@@@@@@@@@@
# ~~~\@@@@@@@@@@@@@@||       |\___________________________/|@/~~~~~~~~~~~\@@@
#     |~~~~\@@@@@@@/ |  |    | | by: S.C.E.S.W.          | ||\____________|@@
#
# Nelder-Mead optimizer written in Python
# Author: Aleksei Volkov (@AlgebraicWolf)

import numpy as np


class NelderMeadOptimizer:
    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, log_simplices=False):
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.log_simplices = log_simplices
        self.simplices = list()

    def get_simplices_log(self):
        """
        Return simplices log of the last optimization

        Retruns:
            List of 2d numpy arrays of shape (n + 1, n) -- List of simplexes at all the steps
        """

        return self.simplices

    def minimize(self, fun, initial_simplex, max_iterations=-1, fatol=None, xatol=None, vartol=None):
        """
        Run Nelder-Mead optimizer on function

        Args:
            fun: np.array[a, n] -> np.array[a] -- Function to optimize (expected to be vectorized)
            initial_simplex: 2d numpy array (n + 1, n) -- Simplex to start optimization with
            max_iterations: int -- Maximum number of iterations (set to any negative in case you want to make it unbounded)
            fatol: double -- Difference between optimal values of two iteration below which the algorithm stops
            xatol: double -- Difference between points of optimum of two iterations below which the algorithm stops
            vartol: double -- Norm of variance of points below which the algorithm terminates
        Returns:
            1d numpy array (a,) -- Minimum detected by the function
        """
        if (fatol is None) and (xatol is None and max_iterations < 0):
            raise Exception("Algorithm is guaranteed to never converge")

        self.simplices = list()

        xprev = None
        fprev = None

        xs = initial_simplex
        n = xs.shape[0] - 1
        iterations = 0
        while iterations != max_iterations:
            if self.log_simplices:
                self.simplices.append(xs)  # Save current simplex to the log

            # Step 1. Ordering simplex vertices

            fs = fun(xs)
            sorted_indices = np.argsort(fs)
            xs = xs[sorted_indices]  # Perform sorting on simplex vertices
            # Don't forget to sort function values as well
            fs = fs[sorted_indices]

            # Check termination conditions
            if fatol is not None and iterations > 0:
                if np.abs(fprev - fs[0]) < fatol:
                    break

            if xatol is not None and iterations > 0:
                if np.linalg.norm(xprev - xs[0]) < xatol:
                    break

            if vartol is not None and iterations > 0:
                if np.linalg.norm(np.var(xs, axis=0)) < vartol:
                    break

            iterations += 1

            fprev = fs[0]
            xprev = xs[0]

            # Step 2. Centroid
            x_o = (np.sum(xs, axis=0) - xs[n]) / n

            # Step 3. Reflection
            x_r = x_o + self.alpha * (x_o - xs[n])
            f_r = fun(x_r)
            if fs[0] <= f_r and f_r < fs[n - 1]:
                xs[n] = x_r
                continue

            # Step 4. Expansion
            if f_r < fs[0]:
                x_e = x_o + self.gamma * (x_r - x_o)
                f_e = fun(x_e)
                if f_e < f_r:
                    xs[n] = x_e
                else:
                    xs[n] = x_r
                continue

            # Step 5. Contraction
            x_c = x_o + self.rho * (xs[n] - x_o)
            f_c = fun(x_c)
            if f_c < fs[n]:
                xs[n] = x_c
                continue

            # Step 6. Shrink
            xs[1:] = xs[0] + self.sigma * (xs[0] - xs[1:])
            continue

        return xs[0]
