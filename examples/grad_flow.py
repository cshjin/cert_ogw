import random

import networkx as nx
import numpy as np
import scipy.optimize as optim
from fgw.gromov_prox import *
from fgw.bary_utils import commute_time
# import sys
# sys.path.append('cert_gw/FGW/lib')


def laplacian(A):
    """ Get graph Laplacian matrix

    Args:
        A (np.ndarray): Adjacency matrix

    Returns:
        (np.ndarray): Laplacian matrix
    """
    return np.diag(A.sum(1)) - A


def rp(X):
    print(np.round(X.reshape(N, N), 3))


def J(C):
    C = np.reshape(C, (N, N))
    return np.linalg.norm(C)**2 + 0.5 * np.linalg.norm(D1)**2 + 0.5 * np.linalg.norm(D2)**2 - np.trace(C @ P1 @ D1 @ P1.T) - np.trace(C @ P2 @ D2 @ P2.T)


def grad_C(C):
    return 2 * C - P1 @ D1.T @ P1.T - P2 @ D2.T @ P2.T


N = 5
N1 = 6
N2 = 3

G1 = nx.binomial_graph(N1, random.random())
G2 = nx.binomial_graph(N2, random.random())
A1 = nx.adjacency_matrix(G1).toarray()
A2 = nx.adjacency_matrix(G2).toarray()
D1 = commute_time(A1)
D2 = commute_time(A2)

G = nx.binomial_graph(N, random.random())
A = nx.adjacency_matrix(G).toarray()
C = commute_time(A)


_, P1 = fused_gromov_upper_bound_rec(np.zeros([N, N1]), C, D1, 1, 'OE', True)
_, P2 = fused_gromov_upper_bound_rec(np.zeros([N, N2]), C, D2, 1, 'OE', True)


# ## $\partial J / \partial C$


grad_approx = optim.approx_fprime(C.flatten(), lambda x: J(x), np.sqrt(np.finfo(float).eps))
grad_true = 2 * C - P1 @ D1.T @ P1.T - P2 @ D2.T @ P2.T

print(rp(grad_approx))
print(rp(grad_true))


def Z(L):
    e = np.ones([N, 1])
    eet = e @ e.T / N
    return np.linalg.inv(L + eet)


Zm = Z(laplacian(A))


def vol(X):
    return X.sum()


def grad_vol():
    return np.ones([N, N])


def vol_Z(Z):
    Z = np.reshape(Z, [N, N])
    e = np.ones([N, 1])
    eet = e @ e.T
    L = np.linalg.inv(Z) - eet / N
    np.fill_diagonal(L, 0)
    return -L.sum()


def grad_vol_Z(Z):
    Z = np.reshape(Z, [N, N])
    Z_inv = np.linalg.inv(Z)
    offdiag = np.ones([N, N])
    # np.fill_diagonal(offdiag, 0)
    return - Z_inv @ Z_inv * offdiag + offdiag


print(vol_Z(Z(laplacian(A))), vol(A))

grad_approx = optim.approx_fprime(Zm.flatten(), lambda x: vol_Z(x), np.sqrt(np.finfo(float).eps)).reshape([N, N])
grad_true = grad_vol_Z(Zm)

rp(grad_approx)
rp(grad_true)

# ## $\partial C / \partial Z$

# T: see Eq. (67)


def calc_T(Z):
    n = Z.shape[0]
    T = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            T[i, j] = (Z[i, i] + Z[j, j] - Z[i, j] - Z[j, i]) if i != j else (Z[i, i] + Z[j, j]) / 2
    return T


def calc_Z(T):
    n = T.shape[0]
    Z = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            Z[i, j] = 0.5 * (T[i, i] + T[j, j] - T[i, j]) if i != j else (T[i, i] + T[j, j]) / 2
    return Z


def vol_T(T):
    return vol_Z(calc_Z(T))


def vol_X(X):
    return vol_Z()

# def T_star(T):
#     n = T.shape[0]
#     Z = np.zeros([n, n])
#     for i in range(n):
#         for j in range(n):
#             Z[i, j] = 0.5*(T[i, i] + T[j, j] - T[i, j]) if i!=j else (T[i, i]+T[j, j])/2
#     return Z


def T_star(X):
    Z = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if i != j:
                Z[i, j] = - X[i, j] - X[j, i]
            else:
                Z[i, j] = -X[i, i] + 2 * sum([X[i, k] for k in range(N)])
    return Z


M = np.random.rand(N**2).reshape([N, N]) * 10
M = M.T / 2 + M / 2
print(np.sum(M * calc_T(Zm)))
print(np.sum(T_star(M) * Zm))


def CT(Z):
    Z = np.reshape(Z, [N, N])
    T = calc_T(Z)
    e = np.ones([N, 1])
    ct = vol_Z(Z) * (T * (e @ e.T - np.identity(N)))
    np.fill_diagonal(ct, 0)
    return ct


def J_Z(Z):
    return J(CT(Z))


grad_approx = optim.approx_fprime(Zm.flatten(), lambda x: vol_Z(x), np.sqrt(np.finfo(float).eps)).reshape([N, N])

# def grad_Z(Z, X):
#     return vol_Z(Z)*T_star(X) + grad_vol_Z(Z)*CT(Z)/vol_Z(Z)


def grad_XZ(Z):
    XZ = np.zeros([N, N, N, N])
    for i in range(N):
        for j in range(N):
            if i != j:
                XZ[i, j, i, j] = -1
                XZ[j, i, i, j] = -1
                XZ[i, i, i, j] = 1
                XZ[j, j, i, j] = 1
    for i in range(N):
        XZ[:, : i, i] *= 0
    # wried observation: only the upper triangular of the last two dims are correct.
    # symmetrize over last two dims.
    for i in range(N):
        for j in range(i + 1, N):
            XZ[:, :, j, i] = XZ[:, :, i, j]
    return XZ


def vec(X):
    return X.reshape([-1, 1])


def grad_CZ(Z):
    gXZ = grad_XZ(Z)
    for i in range(N):
        gXZ[:, :, i, i] *= 0
    grad_CZ = vol_Z(Z) * gXZ.reshape([N**2, N**2]) + vec(grad_vol_Z(Z)) @ vec(CT(Z)).T / vol_Z(Z)
    return grad_CZ.reshape([N, N, N, N])


def grad_Z(Z):
    return np.einsum('pq, ijpq -> ij', grad_C(CT(Z)), grad_CZ(Z))


# grad_true = (vec(grad_C(CT(Zm))).T @ grad_CZ(Zm)).reshape([N, N])
grad_approx = optim.approx_fprime(Zm.flatten(), lambda x: J_Z(x), np.sqrt(np.finfo(float).eps) * 100).reshape([N, N])
grad_true = grad_Z(Zm)

rp(grad_approx / 2 + grad_approx.T / 2)
rp(grad_true)

# ## $\partial Z / \partial L$

L = laplacian(A)


def C(L):
    e = np.ones([N, 1])
    eet = e @ e.T / N
    return np.linalg.inv(L + eet) - eet


def J_L(L):
    L = np.reshape(L, (N, N))
    e = np.ones([N, 1])
    eet = e @ e.T / N
    return J_Z(Z(L))


def grad_L(X, L):
    return -Z(L) @ X @ Z(L)


grad_approx = optim.approx_fprime(L.flatten(), lambda x: J_L(x), np.sqrt(np.finfo(float).eps)).reshape([N, N])
grad_true = grad_L(grad_Z(Z(L)), L)

rp(grad_approx / 2 + grad_approx.T / 2)
rp(grad_true)

# ## $\partial L / \partial X$


def J_X(X):
    X = np.reshape(X, (N, N))
    return J_L(laplacian(X))


def A_star(X):
    N = X.shape[0]
    _ = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            if i != j:
                _[i, j] = -X[i, j] + 0.5 * X[i, i] + 0.5 * X[j, j]
    return _


grad_X_approx = optim.approx_fprime(A.flatten(), lambda x: J_X(x), np.sqrt(np.finfo(float).eps)).reshape([N, N])
grad_X_true = A_star(grad_L(grad_Z(Z(laplacian(A))), laplacian(A)))

rp(grad_X_approx / 2 + grad_X_approx.T / 2)
rp(grad_X_true)


def obj(A):
    return J_X(A)


def grad(A):
    return A_star(grad_L(grad_Z(Z(laplacian(A))), laplacian(A)))


grad_approx = optim.approx_fprime(A.flatten(), lambda x: obj(x), np.sqrt(np.finfo(float).eps)).reshape([N, N])
grad_totest = grad(A)

rp(grad_approx / 2 + grad_approx.T / 2)
rp(grad_totest)


CT(Z(laplacian(A)))
