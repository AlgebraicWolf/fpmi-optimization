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
    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.f = None

    def minimize(self, fun, initial_simplex):
        # TODO Write termination condition
        # TODO Write simplex logging for fancy graphics
        """
        Run Nelder-Mead optimizer on function
        
        Args:
            fun: np.array[a, n] -> np.array[a] -- Function to optimize (expected to be vectorized)
            initial_simplex: 2d numpy array (n + 1, n) -- Simplex to start optimization with

        Returns:
            1d numpy array (a,) -- Minimum detected by the function
        """
        xs = initial_simplex
        n = xs.shape[0] - 1
        while True:
            # Step 1. Ordering simplex vertices
            fs = fun(xs)
            sorted_indices = np.argsort(fs)
            xs = xs[sorted_indices]  # Perform sorting on simplex vertices

            # Step 2. Centroid
            x_o = (np.sum(xs) - xs[n]) / n

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

            xs[1:] = xs[0] + self.sigma * (xs[0] - xs)
            continue
            break
        pass