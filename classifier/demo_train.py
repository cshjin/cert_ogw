import argparse
import logging
import os
import os.path as osp
import pickle

import networkx as nx
import numpy as np
import torch_geometric.utils as pygutils
# from fgw.utils import load_pyg_data
from fgw.data_loader import load_pyg_data
from fgw.dist import commute_time, cttil
from fgw.gromov_prox import projection_matrix
from fgw.gw_lb import flb, tlb
from fgw.gwtil import eval_gwtil_ub, gwtil_lb, gwtil_o, gwtil_ub
from joblib.parallel import Parallel, delayed
from ot.gromov import gromov_wasserstein
from scipy.linalg import eigvalsh
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from torch_geometric.datasets import TUDataset
from tqdm import tqdm


def precomputed_kernel_GridSearchCV(K, y, C_range, gamma_range, n_splits=10, test_size=0.1, random_state=42):
    """A version of grid search CV,
    but adapted for SVM with a precomputed kernel
    K (np.ndarray) : precomputed kernel
    y (np.array) : labels
    Cs (iterable) : list of values of C to try
    return: optimal value of C
    """
    from sklearn.model_selection import StratifiedKFold

    n = K.shape[0]
    assert len(K.shape) == 2
    assert K.shape[1] == n
    assert len(y) == n

    best_score = float('-inf')
    best_C = None

    indices = np.arange(n)
    for gamma in tqdm(gamma_range):
        for C in C_range:
            K = np.exp(-gamma * K)
            # for each value of parameter, do K-fold
            # The performance measure reported by k-fold cross-validation
            # is the average of the values computed in the loop
            scores = []
            # ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
            k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
            # for train_index, test_index in ss.split(indices):
            for train_index, test_index in k_fold.split(K, y):
                K_train = K[np.ix_(train_index, train_index)]
                K_test = K[np.ix_(test_index, train_index)]
                y_train = y[train_index]
                y_test = y[test_index]
                svc = SVC(kernel='precomputed', C=C)
                svc.fit(K_train, y_train)
                scores.append(svc.score(K_test, y_test))

            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_std = np.std(scores)
                best_C = C
                best_gamma = gamma
    print(best_score, best_std, best_C, best_gamma)

    return best_C


class GW_SVC(SVC):
    """ SVM classifier for GW-based distance.

    Note:
        The distance is `precomputed` in the SVC model.
    """

    def __init__(self, *, C=1, kernel='precomputed', degree=3, gamma='scale',
                 coef0=0, shrinking=True, probability=False, tol=0.001,
                 cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', break_ties=False, random_state=None):

        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state)

    def fit(self, D, y):
        """ Customize the fit function with RBF kernel.

        .. math::
            X = \\exp^(-\\gamma * D)
            as the precomputed matrix (gram matrix)

        Args:
            D (np.ndarray): Precalculated distance matrix with dim (n, n).
            y (array-like list): Labels
        """
        X = np.exp(-self.gamma * D)
        super().fit(X, y)


if __name__ == "__main__":
    # np.random.seed(0)

    logging.basicConfig(format='%(asctime)s - %(message)s ', level=logging.INFO)

    ROOT = osp.join(osp.expanduser("~"), 'tmp', 'data', "TUDataset")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="BZR", help="dataset")
    parser.add_argument("--topo_metric", "-tm", type=str, default="sp", help="dataset")
    parser.add_argument("--dist_func", "-df", type=str, default="gw", help="dataset")
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--force_recompute", "-f", dest="force_recompute", action="store_true")
    parser.add_argument("--normalize", "-n", dest="normalize", action="store_true")
    args = parser.parse_args()
    args = vars(args)

    SAVED_PATH = osp.join(ROOT, args["dataset"], "saved")
    if not osp.isdir(SAVED_PATH):
        logging.info("creating folder")
        os.makedirs(SAVED_PATH)

    ds_name = args['dataset']
    topo_metric = args['topo_metric']
    dist_func = args['dist_func']
    debug = args['debug']
    force_recompute = args['force_recompute']

    logging.info(f"dataset: {ds_name} topo_metric: {topo_metric} dist_func: {dist_func}")

    print(f"dataset: {ds_name} topo_metric: {topo_metric} dist_func: {dist_func}")

    D_fn = f"D_{topo_metric}_{dist_func}_normalize.pkl"if args['normalize'] else f"D_{topo_metric}_{dist_func}.pkl"
    # Load pyg data
    X, y = load_pyg_data(args["dataset"])
    # recompute D
    if args["force_recompute"] or (not osp.exists(osp.join(SAVED_PATH, D_fn))):
        # X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, np.arange(len(X)), shuffle=True)
        if topo_metric == "ct":
            Cs = [commute_time(nx.adjacency_matrix(g).toarray()) for g in X]
        elif topo_metric == "cttil":
            Cs = [cttil(nx.adjacency_matrix(g).toarray()) for g in X]
        elif topo_metric == "sp":
            Cs = [nx.floyd_warshall_numpy(g) for g in X]
        # REVIEW: normalize
        Cs = [C / C.max() for C in Cs] if args["normalize"] else [C for C in Cs]
        Ns = [C.shape[0] for C in Cs]
        ps = [np.ones(N) / N for N in Ns]
        evals = [eigvalsh(C) for C in Cs]
        size = len(X)

        logging.info("calculate distance")

        def calc_D(i, j, D):
            if dist_func == "gw":
                T, gw_log = gromov_wasserstein(Cs[i], Cs[j], ps[i], ps[j], loss_fun="square_loss", log=True)
                D[i, j] = gw_log['gw_dist']

            elif dist_func == "gw_lb":
                # D[i, j] = tlb(Cs[i], Cs[j])**2
                D[i, j] = flb(Cs[i], Cs[j])

            elif dist_func == "gwtil_lb":
                D[i, j] = gwtil_lb(Cs[i], Cs[j])
            elif dist_func == "gwtil_o":
                D[i, j] = gwtil_o(Cs[i], Cs[j])
                # raise NotImplementedError
            elif dist_func == "gwtil_lb_q1":
                gwtil_lb_val, Q1, Q2 = gwtil_lb(Cs[i], Cs[j], return_matrix=True)
                m, n = Cs[i].shape[0], Cs[j].shape[0]
                mn_sqrt = np.sqrt(m * n)
                U = projection_matrix(m)
                V = projection_matrix(n)
                em = np.ones((m, 1))
                en = np.ones((n, 1))

                P1 = 1 / mn_sqrt * em @ en.T + U @ Q1 @ V.T
                D[i, j] = eval_gwtil_ub(Cs[i], Cs[j], P1)

            elif dist_func == "gwtil_ub":
                D[i, j] = gwtil_ub(Cs[i], Cs[j])

        if args['debug']:
            ''' single thread '''
            D_mat = np.zeros((size, size))
            for i in tqdm(range(size), leave=False):
                for j in tqdm(range(i + 1, size), leave=False):
                    calc_D(i, j, D_mat)
            D_mat += D_mat.T
        else:
            ''' multi thread '''
            fn_mm = osp.join(ROOT, args["dataset"], "D_mat")
            D_mat = np.memmap(fn_mm, mode="w+", shape=(size, size), dtype=float)
            Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(calc_D)(i, j, D_mat) for i in range(size) for j in range(i + 1, size))
            D_mat += D_mat.T

        pickle.dump(D_mat, open(osp.join(SAVED_PATH, D_fn), "wb"))
    else:
        logging.info(f"load distance matrix from {osp.join(SAVED_PATH, D_fn)}")
        # load D
        D_mat = pickle.load(open(osp.join(SAVED_PATH, D_fn), "rb"))

    print(f"D_min {D_mat.min():.4f} D_max {D_mat.max()}")

    # exit()
    ''' visualize '''
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(4, 4))
    # plt.imshow(D_mat)
    # plt.colorbar()
    # plt.title(args['dist_func'])
    # plt.tight_layout()
    # plt.savefig(f"D_mat_{args['dist_func']}")
    # import pickle
    # pickle.dump(D_mat, open(f"D_mat_{args['dist_func']}.pkl", "wb"))
    # exit()

    # apply the RBF kernel
    # gamma = 1
    # Z = np.exp(-gamma * D_mat)
    logging.info("fitting SVM models")
    gw_svc = GW_SVC(verbose=False, random_state=0)
    # DEBUG: grid search on `gamma`
    C_range = list(np.logspace(-8, 0))

    gamma_range = list([2**k for k in np.linspace(-20, 20)])
    # gamma_range = [0.5]
    # gamma_range = list(np.logspace(-8, 0, 15))
    best_C = precomputed_kernel_GridSearchCV(D_mat, np.array(y), C_range, gamma_range)
    exit()
    param_grid = {"C": C_range,
                  "gamma": gamma_range}
    search = GridSearchCV(gw_svc, param_grid, cv=10)
    search.fit(D_mat, y)
    logging.info("predict SVM models")
    print("best params", search.best_params_)
    print("test mean", search.cv_results_['mean_test_score'][search.best_index_])
    print("test std", search.cv_results_['std_test_score'][search.best_index_])
    print("=" * 30)
