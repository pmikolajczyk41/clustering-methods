import numpy as np


class KMeans:
    def _init_centroids(self, x, k):
        self._centroids = x[np.random.choice(len(x), k, replace=False)]

    @staticmethod
    def _get_cluster_elements(x, cid, cluster_ids):
        return [xi for xi, c in zip(x, cluster_ids) if c == cid]

    @staticmethod
    def compute_centroids(x, cluster_ids, k):
        return np.array([np.mean(KMeans._get_cluster_elements(x, cid, cluster_ids), axis=0)
                         for cid in range(k)], dtype=float)

    def cluster(self, x, k):
        self._cluster_ids = None
        self._init_centroids(x, k)
        while (True):
            new_cluster_ids = [np.argmin(np.linalg.norm(xi - self._centroids, axis=1)) for xi in x]
            if len(set(new_cluster_ids)) < k:
                break
            new_centroids = self.compute_centroids(x, new_cluster_ids, k)
            if np.allclose(new_centroids, self._centroids):
                break
            self._cluster_ids, self._centroids = new_cluster_ids, new_centroids
        if self._cluster_ids is None:
            return self.cluster(x, k - 1)
        return self._cluster_ids, self._centroids

    @staticmethod
    def cost(x, ids, centroids):
        return sum(np.linalg.norm(xi - centroids[id]) ** 2 for xi, id in zip(x, ids))


def get_best_kmeans_clusterization(x, k, trials=10):
    return min((KMeans().cluster(x, k) for _ in range(trials)),
               key=lambda p: KMeans.cost(x, *p))
