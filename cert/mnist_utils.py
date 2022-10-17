import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.linalg import svd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from torch_geometric.data import Data


def read_file(fs):
    """ Load MNIST_2D from csv file. Each row in the file has the following format:
        `y,x0,y0,v0,x1,y1,v1,...,x351,y351,v351`

    Args:
        fs (str): filename.

    Returns:
        tuple:
            np.ndarray: X with dim (N, 351).
            np.array: y with dim (N, ).

    Notes:
        * the image size is `28x28`
        * the maximum number of active pixels is 351
        * padding with (-1,-1,-1) if the active pixels are less 351
    """
    df = pd.read_csv(fs)
    X = df[df.columns[1:]].to_numpy()
    y = df[df.columns[0]].to_numpy()
    return X, y


def load_pyg_data(fs):
    """ Load the Data into pyg data.

    Args:
        fs (str): Filename.

    Returns:
        list: List of `pyg.data`.
    """
    X, y = read_file(fs)
    all_data = []
    for i in range(X.shape[0]):
        data = X[i][X[i] >= 0].reshape((-1, 3))
        d = Data(x=torch.Tensor(data[:, 2:]), pos=torch.Tensor(data[:, :2]), y=torch.Tensor([y[i]]))
        all_data.append(d)
    return all_data


def solve_X(C, method="eig", **kwargs):
    """ Given C, solve the coordinate of X from CVX.

    .. math::
        min_X ||d(X, X) - C||_F^2

    Args:
        C (np.ndarray): Cost matrix with dim (n, n).
        method (str, optional): Choose method from ('eig', 'bfgs', 'cvx'). Defaults to "eig".

    Returns:
        np.ndarray: 2D coordinate of each pixel with dim (n, 2).

    Notes:
        * The best method is based on eigen decomposition.
        * X is not guaranteed element-wise positive.
    """
    N = C.shape[0]
    assert method in ['eig', 'bfgs', 'cvx'], "Choose a method in 'eig', 'bfgs', 'cvx'"
    if method == "eig":
        """ Solve X with eigen decomposition  """
        # ref: https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
        M = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                M[i, j] = (C[0, i]**2 + C[j, 0]**2 - C[i, j]**2) / 2
        # REVIEW: diff between eigen decomp v.s. svd
        # w, v = eigh(M)
        U, s, _ = svd(M)
        # NOTE: no need to shift the coor. of pts., due to the plt
        X_opt = U[:, :2] @ np.diag(np.power(s[:2], 0.5))

    elif method == "bfgs":
        """ Solve X in BFGS solve """
        def obj(X):
            X = X.reshape((N, 2))
            D = cdist(X, X)
            fval = np.linalg.norm(D - C)**2
            return fval

        def callback(X):
            fval = obj(X)
            print(f"obj {fval:.4f}")

        X_init = kwargs.get("X_init", np.random.rand(N * 2))
        # NOTE: no need to shift the coor. of pts., due to the plt
        # bnd = [(0, 28) for i in range(N * 2)]
        res = minimize(obj, X_init,
                       method="BFGS",
                       #    bounds=bnd,
                       callback=callback if kwargs.get("verbose", False) else None,
                       #    options={'disp': True}
                       )
        X_opt = res['x'].reshape((N, 2))

    elif method == "cvx":
        """ Solve X in CVXOPT """
        import cvxpy as cp
        X = cp.Variable((N, 2))
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                D[i, j] = ((X[i][0] - X[j][0])**2 + (X[i][1] - X[j][1])**2).value
        D += D.T
        objective = cp.Minimize(cp.norm(D - C, p='fro'))
        # NOTE: bounded for the variables.
        constraints = [0 <= X, X <= 28]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        X_opt = X.value

    return X_opt


def process_graph(X, threshold=0, **kwargs):
    """ Process numpy array into images represented as graphs.

    Args:
        X (np.ndarray): Input array containing digits with dim (N, m).
        threshold (int, optional): Threshold of pixel value to be filtered out. Defaults to 0.

    Returns:
        tuple of lists:
            Gs: list of graphs in `nx.Graph`
            Ss: size of graphs
            Ds: cost matrix of graphs
            ps: distributions of graphs.
    """
    N = X.shape[0]
    Gs = []
    Ss = []
    Ds = []
    ps = []
    for idx in range(N):
        pos_v = X[idx].reshape((-1, 3))
        pos = pos_v[:, :2]
        v = pos_v[:, 2]
        thres_idx = np.where(v >= threshold)[0]
        pos_ = pos[thres_idx]
        v_ = v[thres_idx]
        G = nx.Graph()
        n = len(thres_idx)
        Ss.append(n)
        ps.append(np.ones(n) / n)
        for i in range(n):
            G.add_node(i, pos=pos_[i], value=v_[i])
        Gs.append(G)

        C = cdist(pos_, pos_)
        Ds.append(C)
    return Gs, Ss, Ds, ps
