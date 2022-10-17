""" Solve the gromov-barycenter problem """

import networkx as nx
import numpy as np
from ot.gromov import gwggrad, gwloss, init_matrix
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from fgw.dist import commute_time
from fgw.optim import cg
from fgw.utils import sym


def grad_gw_C(C, D, T):
    """ Grad of `gromov_wasserstein_v2` w.r.t. C given C, D, T

    .. math::
        \\partial gwtil / \\partial C = 2 * C / m^2 - 2 * (T D T^\top)

    Args:
        C (np.ndarray): Geodesic distance in source domain with dim (m, m).
        D (np.ndarray): Geodesic distance in target domain with dim (n, n).
        T (np.ndarray): Optimized P matrix with dim (m, n).

    Returns:
        np.ndarray: grad w.r.t. C with dim (m, m)

    See Also:
        `gromov_wasserstein_v2`
    """
    m = C.shape[0]
    grad_ = 2 * C / m**2 - 2 * T @ D @ T.T
    return grad_


def grad_bary_C(C, Ds, Ts, lambdas):
    """ Gradient of barycenter w.r.t. C

    Args:
        C (np.ndarray): Distance matrix of barycenter with dim (m, m).
        Ds (list): List of distance matrices from samples.
        Ts (list): List of Ts couplings calculated at each iteration.
        lambdas (np.array): Weights of samples.

    Returns:
        np.ndarray: Gradient with dim (m, m).
    """
    grad = 0
    for lamb, D, T in zip(lambdas, Ds, Ts):
        grad += lamb * grad_gw_C(C, D, T)
    return grad


def update_square_loss_gw(p, lambdas, Ts, Ds):
    """
    Updates C according to the L2 Loss kernel with the S Ts couplings
    calculated at each iteration

    Args:
    p (np.ndarray): Masses in the targeted barycenter.
    lambdas (list): List of the S spaces' weights.
    Ts (list): List of Ts couplings calculated at each iteration.
    Ds (list): list of S ndarray, shape(ns,ns). Metric cost matrices.

    Returns:
        np.ndarray: Updated C matrix with dim (nt, nt)

    Also See:
        `ot.gromov.update_square_loss`

    Note:
        The difference is the order of quadratic term tr (C T D T.T).
    """
    tmpsum = sum([lambdas[s] * np.dot(Ts[s], Ds[s]).dot(Ts[s].T)
                  for s in range(len(Ts))])
    ppt = np.outer(p, p)

    return np.divide(tmpsum, ppt)


def gromov_wasserstein_v2(C1, C2, p, q, loss_fun="square_loss", log=False, armijo=False, **kwargs):
    """
    Returns the gromov-wasserstein transport between (C1,p) and (C2,q)

    The function solves the following optimization problem:

    .. math::
        GW = \\min_T \\sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}

    Where :
    - C1 : Metric cost matrix in the source space
    - C2 : Metric cost matrix in the target space
    - p  : distribution in the source space
    - q  : distribution in the target space
    - L  : loss function to account for the misfit between the similarity matrices

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    loss_fun : str
        loss function used for the solver either 'square_loss' or 'kl_loss'

    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
    **kwargs : dict
        parameters can be directly passed to the ot.optim.cg solver

    Returns
    -------
    T : ndarray, shape (ns, nt)
        Doupling between the two spaces that minimizes:
            \\sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
    log : dict
        Convergence information and loss.

    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [13] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.

    """

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    if kwargs.get("T_init") is not None:
        T = kwargs['T_init']
    else:
        T = np.outer(p, q)
        # T = np.random.rand(len(p), len(q)) / len(p) / len(q)

    def f(T):
        return gwloss(constC, hC1, hC2, T)

    def df(T):
        return gwggrad(constC, hC1, hC2, T)

    if log:
        opt_T, log = cg(p, q, 0, 1, f, df, T, log=True, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)
        log['gw_dist'] = gwloss(constC, hC1, hC2, opt_T)
        return opt_T, log
    else:
        return cg(p, q, 0, 1, f, df, T, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)


def optim_C_gw(N, Ds, ps, p, lambdas, log=False, **kwargs):
    """ Optimize barycenter distance matrix using BFGS with the gw.

    Args:
        N (int): Size of the barycenter.
        Ds (list): List of distance matrices from samples, with length of S.
        ps (list): List of distributions from samples, with length of S.
        p (np.array): Distribution of the barycenter, with dim (m, ).
        lambdas (np.array): Weights of samples, with dim (S, ).
        log (bool, optional): Log the solver information if `True`. Defaults to False.

    Returns:
        np.ndarray: Distance matrix of barycenter.
        dict (optional): Solver information if `log=True`.

    See Also:
        `ot.gromov.gromov_barycenter`
    """
    S = len(Ds)
    # Ts = [None] * S
    Ts = [np.outer(p, ps[i]) for i in range(S)]
    gw_fvals = [0] * S

    def obj(C):
        C = C.reshape((N, -1))
        fval = 0
        for i in range(S):
            T, gwlog = gromov_wasserstein_v2(C, Ds[i], p, ps[i],
                                             log=True, T_init=Ts[i], **kwargs)
            Ts[i] = T
            gw_fvals[i] = gwlog['gw_dist'] * lambdas[i]
        fval = sum(gw_fvals)

        grad = grad_bary_C(C, Ds, Ts, lambdas)
        return fval, grad.flatten()

    def callback(C):
        fval, grad = obj(C)
        print(f"fval {fval:.4f} ||g|| {np.linalg.norm(grad):.4f}")

    if "C_init" in kwargs:
        C_init = kwargs['C_init']
    else:
        ''' init with random P.S.D '''
        C_ = np.random.rand(N, 2)
        C_init = cdist(C_, C_)
        ''' init with cycle graph '''
        # # G = nx.cycle_graph(N)
        # # C_init = nx.floyd_warshall_numpy(G)
        ''' init with sample '''
        # C_init = Ds[0]

    res = minimize(obj, C_init,
                   method="BFGS",
                   jac=True,
                   callback=callback if kwargs.get("verbose") else None
                   )
    C_opt = res['x'].reshape((N, -1))

    # C_opt = update_square_loss(p, lambdas, Ts, Cs)
    if log:
        return C_opt, res
    else:
        return C_opt


def optim_C_gw_v2(N, Cs, ps, p, lambdas, log=False, **kwargs):
    """ Optimize barycenter distance matrix using closed-form solution with the gw.

    Args:
        N (int): Size of the barycenter.
        Ds (list): List of distance matrices from samples, with length of S.
        ps (list): List of distributions from samples, with length of S.
        p (np.array): Distribution of the barycenter, with dim (m, ).
        lambdas (np.array): Weights of samples, with dim (S, ).
        log (bool, optional): Log the solver information if `True`. Defaults to False.

    Returns:
        np.ndarray: Distance matrix of barycenter.
        dict (optional): Solver information if `log=True`.

    See Also:
        `ot.gromov.gromov_barycenter`
    """

    S = len(Cs)
    max_iter = kwargs.get("max_iter", 1000)
    tol = kwargs.get("tol", 1e-15)

    Cs = [np.asarray(Cs[s], dtype=np.float64) for s in range(S)]
    lambdas = np.asarray(lambdas, dtype=np.float64)

    cpt = 0
    prev_fval, cur_fval = 2**31, -2**31
    err = abs(prev_fval - cur_fval)
    # Ts = [None] * S
    Ts = [np.outer(p, ps[i]) for i in range(S)]
    gw_fvals = [0] * S
    fvals = []

    if "C_init" in kwargs:
        C = kwargs["C_init"]
    else:
        ''' init with random P.S.D '''
        C_ = np.random.rand(N, 2)
        C = cdist(C_, C_)
        ''' init with cycle graph '''
        # # G = nx.cycle_graph(N)
        # # C = nx.floyd_warshall_numpy(G)
        ''' init with sample '''
        # C = Ds[0]

    while err > tol and cpt < max_iter:
        cpt += 1

        for i in range(S):
            # NOTE: tr(CPDP.T), different from the `gromov_barycenter`
            T, gwlog = gromov_wasserstein_v2(C, Cs[i], p, ps[i],
                                             log=True,
                                             T_init=Ts[i],
                                             **kwargs)
            Ts[i] = T
            gw_fvals[i] = gwlog['gw_dist'] * lambdas[i]
        cur_fval = sum(gw_fvals)
        err = abs(prev_fval - cur_fval)
        prev_fval = cur_fval
        fvals.append(cur_fval)
        C = update_square_loss_gw(p, lambdas, Ts, Cs)

    if log:
        return C, {"fun": fvals, "Ts": Ts}
    else:
        return C


# def _optimize_A(p, lambdas, Ds, ps, init_A=None):
#     """ direct optimize over A """
#     N = p.shape[0]
#     S = len(Ds)

#     def obj_v1(A):
#         """ with explicit grad """
#         A = A.reshape((N, -1))
#         A = sym(A)
#         np.fill_diagonal(A, 0)
#         C = commute_time(A)

#         # calculate the GW / FGW distance
#         Ts = []
#         logs = []
#         for i in range(S):
#             T, log = gromov_wasserstein_v2(C, Ds[i], p, ps[i], loss_fun="square_loss", log=True)
#             Ts.append(T)
#             logs.append(log)

#         def function_handler():
#             def obj_C(C):
#                 """ barycenter loss """
#                 bc_loss = sum([logs[i]['gw_dist'] for i in range(S)])
#                 return bc_loss

#             def grad_C(C):
#                 """ grad of barycenter wrt C """

#                 _grad_C = 0
#                 for i in range(S):
#                     _grad_C += lambdas[i] * (2 * C / N**2 - 2 * Ts[i] @ Ds[i].T @ Ts[i].T)
#                 _grad_C = sym(_grad_C)
#                 np.fill_diagonal(_grad_C, 0)
#                 return _grad_C
#                 # return grad_bary_C(C, Ds, Ts, lambdas)
#             return obj_C, grad_C

#         fval, grad = barycenter_optimizer3(function_handler, N)
#         print(f"obj {fval(A):.5f}")
#         return fval(A), grad(A).flatten()

#     def obj_v0(A):
#         """ without explicit grad """
#         A = A.reshape((N, -1))
#         A = sym(A)
#         np.fill_diagonal(A, 0)
#         A = np.round(A)
#         C = commute_time(A)
#         bc_loss = 0
#         for i in range(S):
#             T, log = gromov_wasserstein_v2(C, Ds[i], p, ps[i], loss_fun="square_loss", log=True)
#             bc_loss += lambdas[i] * log['gw_dist']
#         print(f"obj {bc_loss:.5f}")
#         return bc_loss

#     def callback(A):
#         import matplotlib.pyplot as plt
#         np.set_printoptions(precision=3)
#         print(A.reshape((N, -1)))
#         A = np.round(A)
#         _A = A.reshape((N, -1))

#         plt.subplot(1, 2, 2)
#         _G = nx.from_numpy_array(_A)
#         nx.draw(_G, pos=nx.kamada_kawai_layout(_G))

#         plt.subplot(1, 2, 1)
#         plt.imshow(_A)
#         plt.colorbar()
#         plt.show()
#         # print(obj_v1(A))

#     bnd = [(0, 1) for i in range(N**2)]

#     # DEBUG: init A randomly
#     # init_A = np.random.randint(0, 2, size=(N, N))
#     # np.fill_diagonal(init_A, 0)
#     # init_A = (init_A + init_A.T) / 2

#     # DEBUG: init A with path / cycle / random graph
#     G = nx.cycle_graph(N)
#     # G = nx.erdos_renyi_graph(N, p=0.5, seed=0)
#     # G = nx.path_graph(N)
#     if init_A is None:
#         init_A = nx.adjacency_matrix(G).toarray().flatten()

#     # init_A = random_gamma_init(p, ps[0])
#     # res = minimize(obj_v0, init_A,
#     #                method="L-BFGS-B",
#     #                bounds=bnd,
#     #                callback=callback,
#     #                options={"maxiter": 500, "eps": 1e-3})

#     res = minimize(obj_v1, init_A,
#                    method="L-BFGS-B",
#                    jac=True,
#                    bounds=bnd,
#                    # callback=callback,
#                    callback=None,
#                    options={"maxiter": 500})
#     A_opt = res['x'].reshape((N, -1))
#     return A_opt


# def _cvx_solver_A(p, lambdas, Cs, ps):
#     # NOTE: not working in cvx
#     import cvxpy as cp

#     def cvx_commute_time(A):
#         N = A.shape[0]
#         A = A.value
#         volG = np.sum(A)
#         L = np.diag(np.sum(A, 1)) - A
#         _L_pinv = np.linalg.pinv(L)
#         C = np.zeros((N, N))
#         E = np.eye(N)
#         for i in range(N):
#             for j in range(i + 1, N):
#                 e_ij = E[:, i] - E[:, j]
#                 # C[i, j] = cp.matrix_frac(e_ij, L).value
#                 C[i, j] = np.inner(e_ij, np.array(_L_pinv) @ e_ij.reshape((-1, 1)).squeeze())
#         C += C.T
#         C *= volG
#         return C

#     N = p.shape[0]
#     S = len(Cs)
#     G = nx.cycle_graph(N)
#     A_init = nx.adjacency_matrix(G).toarray()
#     A = cp.Variable((N, N), integer=True)
#     A.value = A_init
#     C = cvx_commute_time(A)
#     # exit()
#     loss = 0
#     for i in range(S):
#         T, log = gromov_wasserstein_v2(C, Cs[i], p, ps[i], loss_fun="square_loss", log=True)
#         loss += log['gw_dist']
#     # obj = cp.Minimize(cp.norm(C, p="fro"))
#     obj = cp.Minimize(loss)

#     const = [0 <= A, A <= 1]
#     prob = cp.Problem(obj, const)
#     prob.solve(solver=cp.CPLEX, warm_start=True)
#     print(prob.solver_stats.extra_stats)
#     return A.value


# def _gw_barycenter_A(N, Ds, ps, p, lambdas,
#                      loss_fun="square_loss", max_iter=1000,
#                      tol=1e-9, verbose=False, log=False,
#                      init_A=None, random_state=None):
#     """ GW + CT
#     Optimize over A
#     """
#     # S = len(Ds)
#     # if init_A is None:
#     #     G = nx.cycle_graph(N)
#     #     init_A = nx.adjacency_matrix(G)
#     #     init_A = init_A.toarray()

#     A = optimize_A(p, lambdas, Ds, ps)
#     return A
