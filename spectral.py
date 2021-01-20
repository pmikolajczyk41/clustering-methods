import numpy as np
from numpy.linalg import eig
from scipy.linalg import fractional_matrix_power
from sklearn.preprocessing import normalize

from kmeans import KMeans


class Spectral:
    def cluster(self, x, k, delta=2.):
        def dist(i, j):
            if i == j: return 0.
            return np.exp(-delta * np.linalg.norm(x[int(i)] - x[int(j)]) ** 2)

        n = len(x)
        A = np.fromfunction(np.vectorize(dist), (n, n))
        D = np.diag(np.array([sum(A[i]) for i in range(n)]))
        DD = fractional_matrix_power(D, -0.5)
        L = np.matmul(DD, np.matmul(A, DD))

        eigen_values, eigen_vectors = eig(L)
        idx = eigen_values.argsort()[::-1]
        X = eigen_vectors[:, idx][:, :k]
        assert X.shape == (n, k)

        Y = normalize(X, axis=1, norm='l1')
        cluster_ids, _ = KMeans().cluster(Y, k)
        return cluster_ids
