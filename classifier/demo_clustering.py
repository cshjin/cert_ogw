from sklearn.cluster import KMeans
import numpy as np
from ot.gromov import gromov_barycenters, gromov_wasserstein


class GW_KMeans(KMeans):
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.cluster_centers_ = []
        self.cluster_prob_ = []
        self._init_centroids()

    def _init_centroids(self):
        for i in range(self.n_clusters):
            C1 = np.random.rand(4, 2)
            C1 = cdist(C1, C1)
            self.cluster_centers_.append(C1)

    def fit(self, Cs, ps):
        tol = 1e-5
        n = len(Cs)

        distance = np.zeros((n, self.n_clusters))
        # TODO: parallel
        for i in range(n):
            for j in range(self.n_clusters):
                p = np.ones(len(self.cluster_centers_[j])) / len(self.cluster_centers_[j])
                _, log = gromov_wasserstein(Cs[i], self.cluster_centers_[j], ps[i], p, loss_fun="square_loss", log=True)
                distance[i, j] = log['gw_dist']

        error = np.linalg.norm(distance)
        labels = np.argmin(distance, axis=1)
        best = 0
        diff_tol = 1
        max_iter = 10
        it = 0
        print(f"fitting error {error:.4f}")
        while (error > best or diff_tol > tol) and it < max_iter:
            it += 1
            # update center
            for i in range(self.n_clusters):
                idx_y_ = np.where(labels == i)[0]
                N = np.median([Cs[i].shape[0] for i in idx_y_]).astype(int)
                p = np.ones(N) / N
                size = len(idx_y_)
                lambdas = np.ones(size) / size
                _Cs = [Cs[i] for i in idx_y_]
                _ps = [ps[i] for i in idx_y_]
                C = gromov_barycenters(N, _Cs, _ps, p, lambdas, loss_fun="square_loss")
                self.cluster_centers_[i] = C

            # recalculate distance
            for i in range(n):
                for j in range(self.n_clusters):
                    p = np.ones(len(self.cluster_centers_[j])) / len(self.cluster_centers_[j])
                    _, log = gromov_wasserstein(Cs[i], self.cluster_centers_[j], ps[i],
                                                p, loss_fun="square_loss", log=True)
                    distance[i, j] = log['gw_dist']
            new_error = np.linalg.norm(distance)
            # TODO
            print(f"fitting error {new_error:.4f}")
            if new_error < best:
                diff_tol = best - new_error
                best = new_error
            # print(distance)
        return self

    def predict(self, Cs, ps):
        """ predict the cluster """
        n = len(Cs)
        distance = np.zeros((n, self.n_clusters))
        for i in range(n):
            for j in range(self.n_clusters):
                p = np.ones(len(self.cluster_centers_[j])) / len(self.cluster_centers_[j])
                _, log = gromov_wasserstein(Cs[i], self.cluster_centers_[j], ps[i], p, loss_fun="square_loss", log=True)
                distance[i, j] = log['gw_dist']
        labels = np.argmin(distance, axis=1)
        # print(distance)
        return labels

    # def score(self, X, y=None, sample_weight=None):
    #     return super().score(X, y=y, sample_weight=sample_weight)


if __name__ == "__main__":
    from scipy.spatial.distance import cdist
    np.random.seed(0)

    def pseudo_d(s):
        C = np.random.rand(s, 2)
        C = cdist(C, C)
        return C

    gw_km = GW_KMeans(2)
    m, n, k = 4, 6, 5
    # ps = [np.ones(k) / k]
    Cs = [pseudo_d(np.random.randint(4, 10)) for _ in range(100)]
    ps = [np.ones(C.shape[0]) / C.shape[0] for C in Cs]
    gw_km.fit(Cs, ps)

    Ck = np.random.rand(10, 2)
    Ck = cdist(Ck, Ck)
    # ps = [np.ones(10) / 10]
    print(gw_km.predict(Cs, ps))
