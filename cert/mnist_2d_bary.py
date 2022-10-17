import logging
import os.path as osp

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh, svd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from fgw.gw_bary import optim_C_gw_v2
from fgw.gwtil_bary import optim_C_gwtil_lb_v2, optim_C_gwtil_ub_v2


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-t", type=str, default='gw')
    args = parser.parse_args()
    args = vars(args)
    digit = args['d']
    topo = args['t']
    S = 5

    np.random.rand(0)
    logging.basicConfig(format='%(asctime)s - %(message)s ', level=logging.INFO)
    ROOT = osp.join(osp.expanduser("~"), "tmp", "data", "MNIST_2D")
    X, y = read_file(osp.join(ROOT, "test.csv"))
    # print(X.shape, y.shape)
    lambdas = np.ones(S) / S

    filter_idx = np.where(y == digit)[0][:S]
    Gs, Ss, Ds, ps = process_graph(X[filter_idx], 200)

    N = min(Ss)
    p = np.ones(N) / N

    # init X from the sample
    logging.info(Ss)
    min_idx = np.argmin(Ss)

    logging.info(f"solve C with digit {args['d']} under {args['t']} ")

    if topo == 'gw':
        C = optim_C_gw_v2(N, Ds, ps, p, lambdas, loss_fun="square_loss")
    elif topo == 'gwtil_ub':
        C = optim_C_gwtil_ub_v2(N, Ds, ps, p, lambdas, C_init=Ds[min_idx])
    elif topo == "gwtil_lb":
        C = optim_C_gwtil_lb_v2(N, Ds, ps, p, lambdas, C_init=Ds[min_idx])
    # C = sum([eigen_projection(C, Ds[i]) * lambdas[i] for i in range(len(Ds))])

    logging.info("solve X (coordinates)")
    # X_init is used in BFGS
    X_init = np.array(list(nx.get_node_attributes(Gs[min_idx], 'pos').values()))
    X = solve_X(C, method="eig")

    logging.info("plot graph")
    fig = plt.figure(figsize=(2 * S + 3, 2), tight_layout=True)
    for i in range(S):
        plt.subplot(1, S, i + 1)
        size = np.array(list(nx.get_node_attributes(Gs[i], "value").values())) / 10
        color = size
        nx.draw(Gs[i], nx.get_node_attributes(Gs[i], 'pos'), node_color=color, node_size=size, cmap="Greys")
        plt.axis('on')
    plt.show()

    fig = plt.figure(figsize=(2, 2), tight_layout=True)
    # plt.subplot(1, S + 1, S + 1)
    G2 = nx.Graph()
    for i in range(N):
        G2.add_node(i, pos=X[i])
    nx.draw(G2, nx.get_node_attributes(G2, 'pos'), cmap="Greys", node_size=20)
    plt.axis("on")
    # plt.title("barycenter")
    plt.show()
    # plt.savefig(f"tmp_{digit}_{topo}.png")
