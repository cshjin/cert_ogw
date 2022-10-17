import argparse
import logging
import os
import pickle
import random

import numpy as np


def create_log_dir(FLAGS, name):
    import datetime
    import dateutil

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    log_dir = FLAGS.log_dir + "/" + name + "_" + timestamp
    print(log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save command line arguments
    with open(log_dir + "/hyperparameters_" + timestamp + ".csv", "w") as f:
        for arg in FLAGS.__dict__:
            f.write(arg + "," + str(FLAGS.__dict__[arg]) + "\n")

    return log_dir


def unique_repr(dictio, type_='normal'):
    """Compute a hashable unique representation of a list of dict with unashable values"""
    if 'normal':
        t = tuple((k, dictio[k]) for k in sorted(dictio.keys()))
    if 'not_normal':
        t = ()
        for k in sorted(dictio.keys()):
            if not isinstance(dictio[k], list):
                t = t + ((k, dictio[k]),)
            else:  # suppose list of dict
                listechanged = []
                for x in dictio[k]:
                    for k2 in sorted(x.keys()):
                        if not isinstance(x[k2], dict):
                            listechanged.append((k2, x[k2]))
                        else:
                            listechanged.append((k2, tuple((k3, x[k2][k3]) for k3 in sorted(x[k2].keys()))))
                tupletoadd = ((k, tuple(listechanged)),)
                t = t + tupletoadd
    return t


def save_obj(obj, name, path='obj/'):
    try:
        if not os.path.exists(path):
            print('Makedir')
            os.makedirs(path)
    except OSError:
        raise
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path):
    with open(path + name, 'rb') as f:
        return pickle.load(f)


def indices_to_one_hot(number, nb_classes, label_dummy=-1):
    """Convert an iterable of indices to one-hot encoded labels."""

    if number == label_dummy:
        return np.zeros(nb_classes)
    else:
        return np.eye(nb_classes)[number]


def dist(x1, x2=None, metric='sqeuclidean'):
    """Compute distance between samples in x1 and x2 using function scipy.spatial.distance.cdist

    Parameters
    ----------
    x1 : np.array (n1,d)
        matrix with n1 samples of size d
    x2 : np.array (n2,d), optional
        matrix with n2 samples of size d (if None then x2=x1)
    metric : str, fun, optional
        name of the metric to be computed (full list in the doc of scipy),  If a string,
        the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.

    Returns
    -------
    M : np.array (n1,n2)
        distance matrix computed with given metric
    """
    from scipy.spatial.distance import cdist
    if x2 is None:
        x2 = x1

    return cdist(x1, x2, metric=metric)


def reshaper(x):
    x = np.array(x)
    try:
        a = x.shape[1]
        return x
    except IndexError:
        return x.reshape(-1, 1)


def hamming_dist(x, y):
    # print('x',len(x[-1]))
    # print('y',len(y[-1]))
    return len([i for i, j in zip(x, y) if i != j])


def allnan(v):  # fonctionne juste pour les dict de tuples
    from math import isnan

    import numpy as np
    return np.all(np.array([isnan(k) for k in list(v)]))


def dict_argmax(d):
    l = {k: v for k, v in d.items() if not allnan(v)}
    return max(l, key=l.get)


def dict_argmin(d):
    return min(d, key=d.get)


def read_files(mypath):
    from os import listdir
    from os.path import isfile, join

    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def per_section(it, is_delimiter=lambda x: x.isspace()):
    ret = []
    for line in it:
        if is_delimiter(line):
            if ret:
                yield ret  # OR  ''.join(ret)
                ret = []
        else:
            ret.append(line.rstrip())  # OR  ret.append(line)
    if ret:
        yield ret


def split_train_test(dataset, ratio=0.9, seed=None):
    idx_train = []
    X_train = []
    X_test = []
    random.seed(seed)
    for idx, val in random.sample(list(enumerate(dataset)), int(ratio * len(dataset))):
        idx_train.append(idx)
        X_train.append(val)
    idx_test = list(set(range(len(dataset))).difference(set(idx_train)))
    for idx in idx_test:
        X_test.append(dataset[idx])
    x_train, y_train = zip(*X_train)
    x_test, y_test = zip(*X_test)
    return np.array(x_train), np.array(y_train), np.array(
        idx_train), np.array(x_test), np.array(y_test), np.array(idx_test)


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


# def padding(C1, C2, pad_value=0):
#     """ padding on the smaller matrix """
#     if C1.shape[0] > C2.shape[0]:
#         pad_width = C1.shape[0] - C2.shape[0]
#         return C1, np.pad(C2, ((0, pad_width)), pad_with)
#     elif C1.shape[0] < C2.shape[0]:
#         pad_width = C2.shape[0] - C1.shape[0]
#         return np.pad(C1, (0, pad_width), pad_with), C2
#     else:
#         return C1, C2


# def squarify(M, pad_value=0):
#     if M.shape[0] > M.shape[1]:
#         pad_width = M.shape[0] - M.shape[1]
#         return np.pad(M, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
#     elif M.shape[0] < M.shape[1]:
#         pad_width = M.shape[1] - M.shape[0]
#         return np.pad(M, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
#     else:
#         return M


def padding(C1, C2, pad_value=0):
    """ Padding smaller matrix with 0

    Args:
        C1 (np.ndarray): Input cost matrix with dim (m, m).
        C2 (np.ndarray): Input cost matrix with dim (n, n).
        pad_value (int, optional): Value to pad. Defaults to 0.

    Returns:
        tuple:
            np.ndarray: C1 matrix with dim (max(m, n), max(m, n))
            np.ndarray: C2 matrix with dim (max(m, n), max(m, n))
    """
    assert C1.shape[0] == C1.shape[1]
    assert C2.shape[0] == C2.shape[1]
    m, n = C1.shape[0], C2.shape[0]
    s = max(m, n)
    C1_ = np.zeros((s, s))
    C2_ = np.zeros((s, s))
    C1_[:m, :m] = C1
    C2_[:n, :n] = C2
    return C1_, C2_


def squarify(M):
    """ Padding 0 to rectanglar matrix to be a square matrix.

    Args:
        M (np.ndarray): Input matrix with dim (m, n).

    Returns:
        np.ndarray: Squarified matrix with dim(max(m, n), max(m, n)).
    """
    m, n = M.shape
    s = max(m, n)
    T = np.zeros((s, s))
    T[:m, :n] = M
    return T


def sym(X):
    """ Symmetrify matrix.

    Args:
        X (np.ndarray): Input matrix with dim (n, n).

    Returns:
        np.ndarray: Symmetric output matrix.
    """
    return (X + X.T) / 2


def random_perturb(A, n=1, seed=0):
    """ Random perturb graph with n edges.

    Args:
        A (sp.spmatrix): Adjacency matrix.
        n (int, optional): Number of perturbations.

    Returns:
        (sp.spmatrix): Adjacency matrix after perturbing.
    """
    from scipy.sparse.csgraph import connected_components

    np.random.seed(seed)
    nG = A.shape[0]

    all_edges = np.array([(i, j) for i in range(nG) for j in range(i + 1, nG)])
    while True:
        _A = A.copy()
        idx = np.random.choice(len(all_edges), size=n, replace=False)
        picked_edge = all_edges[idx, :]
        for u, v in picked_edge:
            if u != v:
                if _A[u, v] == 1:
                    _A[u, v] = 0
                    _A[v, u] = 0
                else:
                    _A[u, v] = 1
                    _A[v, u] = 1
        # prevent disconnected graphs
        if connected_components(_A)[0] == 1:
            return _A
        else:
            continue


def laplacian(A):
    """ Get graph Laplacian matrix

    Args:
        A (np.ndarray): Adjacency matrix

    Returns:
        (np.ndarray): Laplacian matrix
    """
    return np.diag(A.sum(1)) - A


def line_search(A, inf=0, sup=2):
    As = []
    dist = []
    search = np.linspace(inf, sup)
    for alpha in search:
        A_ = np.round(A - alpha)
        dist.append(np.linalg.norm(A - A_))
        As.append(A_)
    return As[np.argmin(dist)]
