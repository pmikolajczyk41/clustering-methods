from copy import deepcopy

import numpy as np

METHODS = ['single', 'full', 'avg', 'centr']


class Hierarchical:
    def __init__(self, method):
        def const(c):
            return lambda A, B: c

        assert method in METHODS
        if method == 'single':
            self._alpha_a, self._alpha_b, self._beta, self._gamma = const(.5), const(.5), const(0.), const(-.5)
        elif method == 'full':
            self._alpha_a, self._alpha_b, self._beta, self._gamma = const(.5), const(.5), const(0.), const(.5)
        elif method == 'avg':
            self._alpha_a = lambda A, B: A / (A + B)
            self._alpha_b = lambda A, B: B / (A + B)
            self._beta, self._gamma = const(0.), const(0.)
        elif method == 'centr':
            self._alpha_a = lambda A, B: A / (A + B)
            self._alpha_b = lambda A, B: B / (A + B)
            self._beta = lambda A, B: -A * B / ((A + B) ** 2)
            self._gamma = const(0.)

    @staticmethod
    def dist(x1, x2):
        return np.linalg.norm(x1 - x2)

    def cluster(self, x):
        n = len(x)
        cluster_ids, current_ids = [], np.array([i for i in range(n)])
        mapping = {i: i for i in range(n)}
        D = np.array([[self.dist(xi, xj) for xj in x] for xi in x], dtype=float)
        for i in range(len(x)):
            D[i, i] = np.inf
        for it in range(n - 1):
            assert len([mapping[i] for i in range(len(D))]) == len(D)
            assert len(set(current_ids)) == len(D) == n - it
            am = np.argmin(D)
            a, b = am // len(D), am % len(D)
            assert a != b
            a, b = min(a, b), max(a, b)
            A, B = np.count_nonzero(current_ids == mapping[a]), np.count_nonzero(current_ids == mapping[b])

            current_ids[current_ids == mapping[b]] = mapping[a]
            cluster_ids.append(deepcopy(current_ids))

            C = np.zeros_like(D[a])
            for d in range(len(D)):
                if d in [a, b]: continue
                C[d] = self._alpha_a(A, B) * D[d, a] + \
                       self._alpha_b(A, B) * D[d, b] + \
                       self._beta(A, B) * D[a, b] + \
                       self._gamma(A, B) * (abs(D[d, a] - D[d, b]))

            D[:, [b, -1]] = D[:, [-1, b]]
            D[[b, -1]] = D[[-1, b]]
            mapping[b] = mapping[len(D) - 1]
            C[[b, -1]] = C[[-1, b]]

            D = np.delete(D, -1, 0)
            D = np.delete(D, -1, 1)
            C = np.delete(C, -1)

            D[:, [a, -1]] = D[:, [-1, a]]
            D[[a, -1]] = D[[-1, a]]
            C[[a, -1]] = C[[-1, a]]
            C[-1] = np.inf
            D[:, -1] = D[-1, :] = C

            old_map = mapping[a]
            mapping[a] = mapping[len(D) - 1]
            mapping[len(D) - 1] = old_map

        return cluster_ids
