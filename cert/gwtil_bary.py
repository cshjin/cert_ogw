""" barycenter based gwtil """
import networkx as nx
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from fgw.cttil_bary import grad_C_T, grad_T_Z
from fgw.dist import calc_T, cttil_Z
from fgw.gromov_prox import projection_matrix
from fgw.gwtil import (eval_gwtil_lb, eval_gwtil_ub, gwtil_lb, gwtil_lb_lb,
                       gwtil_ub)
from fgw.utils import padding

################################################################################
# evaluate barycenter loss function
################################################################################


def eval_bary_gwtil_lb(C, Ds, lambdas, Q1s, Q2s):
    """ Evaluate the barycenter with gwtil_lb

    Args:
        C (np.ndarray): Distance matrix from barycenter.
        Ds (list): List of distance matrices from samples.
        lambdas (np.array): Weights of samples.
        Q1s (list): List of optimal Q1 matrices from gwtil_lb.
        Q2s (list): List of optimal Q2 matrices from gwtil_lb.

    Returns:
        float: Loss from barycenter problem.

    See Also:
        `eval_gwtil_lb`
    """
    fval = 0
    for D, lamb, Q1, Q2 in zip(Ds, lambdas, Q1s, Q2s):
        fval += lamb * eval_gwtil_lb(C, D, Q1, Q2)
    return fval


def eval_bary_gwtil_ub(C, Ds, lambdas, Ps):
    """ Evaluate the barycenter with gwtil_ub

    Args:
        C (np.ndarray): Distance matrix from barycenter.
        Ds (list): List of distance matrices from samples.
        lambdas (np.array): Weights of samples.
        Ps (list): List of optimal P matrices from gwtil_ub.

    Returns:
        float: Loss from barycenter problem.

    See Also:
        `eval_gwtil_ub`
    """
    fval = 0
    for D, lamb, P in zip(Ds, lambdas, Ps):
        fval += lamb * eval_gwtil_ub(C, D, P)
    return fval

################################################################################
# Update barycenter structure
################################################################################


def update_square_loss_gwtil_ub(p, lambdas, Ps, Ds):
    """ Update barycenter C using the optimal Ps from gwtil_ub.

    .. math::
        C^* = \\sum_i lambdas_i P_i D_i P_i^\top / m / n_i
        due to tr(C P D P.T)

    Args:
        p (np.array): Distribution of the barycenter, with dim (m, ).
        lambdas (np.array): Weights of samples, with dim (S, ).
        Ps (list): List of P in gwtil_ub, with length of S.
        Ds (list): List of distance matrices from samples, with length of S.

    Returns:
        np.ndarray: Updated distance matrix of barycenter.

    See Also:
        `ot.gromov.update_square_loss`
    """
    m = len(p)
    S = len(lambdas)
    Ss = [D.shape[0] for D in Ds]
    tmpsum = sum([lambdas[i] * (Ps[i] @ Ds[i] @ Ps[i].T) / m / Ss[i] for i in range(S)])

    # tmpsum = sum([lambdas[s] * np.dot(Ps[s], Ds[s]).dot(Ps[s].T) / N / Ds[s].shape[0]
    #               for s in range(len(Ps))])
    ppt = np.outer(p, p)

    return np.divide(tmpsum, ppt)


def update_square_loss_gwtil_lb(p, lambdas, Q1s, Q2s, Ds):
    """ Update barycenter C using the optimal Q1s and Q2s from gwtil_lb.

    .. math::
        C^* = \\sum_i lambdas_i P_i D_i P_i^\top / m / n_i
        where P_i = 1/sqrt(mn) + U Q V^\top

    Args:
        p (np.array): Distribution of the barycenter, with dim (m, ).
        lambdas (np.array): Weights of samples, with dim (S, ).
        Q1s (list): List of Q1 in gwtil_lb, with length of S.
        Q2s (list): List of Q2 in gwtil_lb, with length of S.
        Ds (list): List of distance matrices from samples, with length of S.

    Returns:
        np.ndarray: Updated distance matrix of barycenter.

    See Also:
        `ot.gromov.update_square_loss`
    """
    ppt = np.outer(p, p)
    m = len(p)
    S = len(Ds)
    Ss = [D.shape[0] for D in Ds]
    tmpsum = 0
    em = np.ones((m, 1))
    U = projection_matrix(m)
    for i in range(S):
        n = Ss[i]
        en = np.ones((n, 1))
        mn = m * n
        mn_sqrt = np.sqrt(mn)
        V = projection_matrix(n)
        const_ = np.ones((m, m)) * Ds[i].sum() / mn
        lin_ = 1 / mn_sqrt * (em @ en.T @ Ds[i] @ V @ Q2s[i].T @ U.T + U @ Q2s[i] @ V.T @ Ds[i] @ en @ em.T)
        quad_ = U @ Q1s[i] @ V.T @ Ds[i] @ V @ Q1s[i].T @ U.T
        tmpsum += (const_ + lin_ + quad_) * lambdas[i] / mn
    return np.divide(tmpsum, ppt)


def update_square_loss_gwtil_lb_lb(p, lambdas, Ps, Cs):
    """ Update barycenter C using the optimal Ps from gwtil_lb_lb.

    .. math::
        C^* = \\sum_i lambdas_i P_i D_i P_i^\top / m / n_i
        due to tr(C P D P.T)

    Args:
        p (np.array): Distribution of the barycenter, with dim (m, ).
        lambdas (np.array): Weights of samples, with dim (S, ).
        Ps (list): List of P in gwtil_lb_lb, with length of S.
        Ds (list): List of distance matrices from samples, with length of S.

    Returns:
        np.ndarray: Updated distance matrix of barycenter.

    See Also:
        `ot.gromov.update_square_loss`
    """
    return update_square_loss_gwtil_ub(p, lambdas, Ps, Cs)

################################################################################
# Gradient w.r.t to barycenter C
################################################################################


def grad_Qcal_lb_C(D, P):
    """ Grad of `Qcal_lb` w.r.t. C given D, P
    .. math::
        \\partial Qcal_lb / \\partial C = P D P.T

    Args:
        D (np.ndarray): Geodesic distance in target domain with dim (n, n).
        P (np.ndarray): Optimized P matrix with dim (m, n).

    Returns:
        np.ndarray: grad w.r.t. C with dim (m, m).
    """
    return P @ D @ P.T


def grad_Qcal_ub_C(D, Q1, Q2):
    """ Grad of `Qcal_ub` w.r.t. C given D, Q1, Q2
    .. math::
        \\partial Qcal_ub / \\partial C = P D P.T

    Args:
        D (np.ndarray): Geodesic distance in target domain with dim (n, n).
        Q1 (np.ndarray): Optimized Q1 matrix with dim (m-1, n-1).
        Q2 (np.ndarray): Optimized Q2 matrix with dim (m-1, n-1).

    Returns:
        np.ndarray: grad w.r.t. C with dim (m, m).
    """
    m = Q1.shape[0] + 1
    n = D.shape[0]
    mn = m * n
    mn_sqrt = np.sqrt(mn)

    U = projection_matrix(m)
    V = projection_matrix(n)

    eet = np.ones((m, n))
    grad_const_ = eet @ D @ eet.T
    grad_linear_ = eet @ D @ V @ Q2.T @ U.T + U @ Q2 @ V.T @ D @ eet.T
    grad_quad_ = U @ Q1 @ V.T @ D @ V @ Q1.T @ U.T
    grad_ = grad_const_ / mn + grad_linear_ / mn_sqrt + grad_quad_
    np.testing.assert_array_almost_equal(grad_, grad_.T)
    return grad_


def grad_Qcal_ub_v2_C(D, Q1, Q2, Q3):
    """ Grad of `Qcal_ub` w.r.t. C given D, Q1, Q2
    .. math::
        \\partial Qcal_ub / \\partial C = P D P.T

    Args:
        D (np.ndarray): Geodesic distance in target domain with dim (n, n).
        Q1 (np.ndarray): Optimized Q1 matrix with dim (m-1, n-1).
        Q2 (np.ndarray): Optimized Q2 matrix with dim (m-1, n-1).

    Returns:
        np.ndarray: grad w.r.t. C with dim (m, m).
    """
    m = Q1.shape[0] + 1
    n = D.shape[0]
    mn = m * n
    mn_sqrt = np.sqrt(mn)

    U = projection_matrix(m)
    V = projection_matrix(n)

    eet = np.ones((m, n))
    grad_const_ = eet @ D @ eet.T
    grad_linear_ = eet @ D @ V @ Q2.T @ U.T + U @ Q3 @ V.T @ D @ eet.T
    grad_quad_ = U @ Q1 @ V.T @ D @ V @ Q1.T @ U.T
    grad_ = grad_const_ / mn + grad_linear_ / mn_sqrt + grad_quad_
    # np.testing.assert_array_almost_equal(grad_, grad_.T)
    return grad_


def grad_gwtil_ub_C(C, D, P):
    """ Grad of `gwtil_ub` w.r.t. C given C, D, P

    .. math::
        \\partial gwtil / \\partial C = 2 * C / m^2 - 2 / mn * (P D P^\top)

    Args:
        C (np.ndarray): Geodesic distance in source domain with dim (m, m).
        D (np.ndarray): Geodesic distance in target domain with dim (n, n).
        P (np.ndarray): Optimized P matrix with dim (m, n).

    Returns:
        np.ndarray: grad w.r.t. C with dim (m, m)

    See Also:
        `gwtil_ub`
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n

    grad_ = 2 / m**2 * C - 2 / mn * (P @ D @ P.T)
    np.testing.assert_array_almost_equal(grad_, grad_.T)
    return grad_


def grad_gwtil_lb_C(C, D, Q1, Q2):
    """ Grad of `gwtil_lb` w.r.t. C given C, D, Q1, Q2

    .. math::
        \\partial gwtil / \\partial C = 2 * C / m^2 - 2 / mn * (\\partial Qcal / \\parital C)
        \\partial Qcal / \\partial C
                = 11^\top sD
                + 2 / \\sqrt{mn} 11^\top D V Q2^\top U^\top
                + U Q1 V^\top D V Q1^\top U^\top

    Args:
        C (np.ndarray): Geodesic distance in source domain with dim (m, m).
        D (np.ndarray): Geodesic distance in target domain with dim (n, n).
        Q1 (np.ndarray): Optimized Q1 matrix with dim (m-1, n-1).
        Q2 (np.ndarray): Optimized Q2 matrix with dim (m-1, n-1).

    Returns:
        np.ndarray: grad w.r.t. C with dim (m, m)

    See Also:
        `gwtil_lb`
    """
    m = C.shape[0]
    n = D.shape[0]
    mn = m * n
    mn_sqrt = np.sqrt(mn)

    U = projection_matrix(m)
    V = projection_matrix(n)

    eet = np.ones((m, n))
    # _grad_const = np.ones((m, m)) * D.sum()
    _grad_const = eet @ D @ eet.T
    # NOTE: need to explicit use two terms -> symmetric in grad
    _grad_linear = eet @ D @ V @ Q2.T @ U.T + U @ Q2 @ V.T @ D @ eet.T
    _grad_quad = U @ Q1 @ V.T @ D @ V @ Q1.T @ U.T
    grad_ = 2 / m**2 * C - 2 / mn * (
        _grad_const / mn + _grad_linear / mn_sqrt + _grad_quad
    )
    np.testing.assert_array_almost_equal(grad_, grad_.T)
    return grad_


def grad_gwtil_lb_lb_C(C, D, P):
    """ Grad of `gwtil_lb_lb` w.r.t. C given C, D, P

    Note:
        This is similar to `grad_gwtil_ub_C`.

    Args:
        C (np.ndarray): Geodesic distance in source domain with dim (m, m).
        D (np.ndarray): Geodesic distance in target domain with dim (n, n).
        P (np.ndarray): Optimized P matrix with dim (m, n).

    Returns:
        np.ndarray: grad w.r.t. C with dim (m, m)

    See Also:
        `gwtil_lb_lb`
    """
    return grad_gwtil_ub_C(C, D, P)

################################################################################
# Optimize over barycenter C
################################################################################


def optim_C_gwtil_ub(N, Ds, ps, p, lambdas, log=False, **kwargs):
    """ Optimize barycenter distance matrix using BFGS with the gwtil_ub

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
    """
    S = len(Ds)
    Ps = [None] * S
    gwtil_fval = [0] * S

    def obj(C):
        C = C.reshape((N, N))
        for i in range(S):
            # NOTE: init P with previous step
            val, P = gwtil_ub(C, Ds[i], return_matrix=True, P_init=Ps[i])
            # update P and loss
            Ps[i] = P
            gwtil_fval[i] = val * lambdas[i]
        fval = sum(gwtil_fval)

        # return fval
        grad = 0
        for i in range(S):
            grad += lambdas[i] * grad_gwtil_ub_C(C, Ds[i], Ps[i])
        return fval, grad.flatten()

    def callback(C):
        fval, grad = obj(C)
        print(f"obj {fval:.4f} ||g|| {np.linalg.norm(grad):.4f}")

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

    bnd = [(0, None)] * N**2
    res = minimize(obj, C_init,
                   method="BFGS",
                   jac=True,
                   callback=callback if kwargs.get("verbose") else None
                   )
    C_opt = res['x'].reshape((N, N))

    if log:
        return C_opt, res
    else:
        return C_opt


def optim_C_gwtil_ub_v2(N, Ds, ps, p, lambdas, log=False, **kwargs):
    """ Optimize barycenter distance matrix using closed-form solution with the gwtil_ub.

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
    """
    S = len(Ds)
    max_iter = kwargs.get("max_iter", 1000)
    tol = kwargs.get("tol", 1e-15)

    cpt = 0
    prev_fval, cur_fval = 2**31, -2**31
    err = abs(prev_fval - cur_fval)
    Ps = [None] * S
    gwtil_fvals = [0] * S
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
            val, P = gwtil_ub(C, Ds[i], return_matrix=True, P_init=Ps[i], **kwargs)
            Ps[i] = P
            gwtil_fvals[i] = val * lambdas[i]

        cur_fval = sum(gwtil_fvals)
        err = abs(prev_fval - cur_fval)
        prev_fval = cur_fval
        fvals.append(cur_fval)
        C = update_square_loss_gwtil_ub(p, lambdas, Ps, Ds)
        if kwargs.get("verbose"):
            print(f"fval {cur_fval:.4f}")

    if log:
        return C, {"fun": fvals, "Ps": Ps}
    else:
        return C


def optim_C_gwtil_lb(N, Ds, ps, p, lambdas, log=False, **kwargs):
    """ Optimize barycenter distance matrix using BFGS with the gwtil_lb

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
    """
    S = len(Ds)
    Q1s = [None] * S
    Q2s = [None] * S
    gwtil_fval = [0] * S

    def obj(C):
        C = C.reshape((N, -1))
        # np.fill_diagonal(C, 0)
        for i in range(S):
            val, Q1, Q2 = gwtil_lb(C, Ds[i], return_matrix=True)
            # update Q1, Q2 and loss
            Q1s[i] = Q1
            Q2s[i] = Q2
            gwtil_fval[i] = val * lambdas[i]
        fval = sum(gwtil_fval)

        grad = 0
        for i in range(S):
            grad += lambdas[i] * grad_gwtil_lb_C(C, Ds[i], Q1s[i], Q2s[i])
        # np.fill_diagonal(grad, 0)
        return fval, grad.flatten()

    def callback(C):
        fval, grad = obj(C)
        print(f"obj {fval:.4f} ||g|| {np.linalg.norm(grad):.4f}")

    if "C_init" in kwargs:
        C_init = kwargs['C_init']
    else:
        ''' init with random P.S.D '''
        C_ = np.random.rand(N, 2)
        C_init = cdist(C_, C_)
        ''' init with cycle graph '''
        # G = nx.cycle_graph(N)
        # C_init = nx.floyd_warshall_numpy(G)
        ''' init with sample '''
        # C_init = Ds[0]

    bnd = [(0, None)] * N**2
    res = minimize(obj, C_init,
                   #    method="L-BFGS-B",
                   method="BFGS",
                   jac=True,
                   #    bounds=bnd,
                   callback=callback if kwargs.get("verbose") else None
                   )
    C_opt = res['x'].reshape((N, N))
    # NOTE: with the optimal Q1s Q2s, the update C is same from BFGS.
    # C_opt = update_square_loss_gwtil_lb(p, lambdas, Q1s, Q2s, Ds)
    if log:
        return C_opt, res
    else:
        return C_opt


def optim_C_gwtil_lb_v2(N, Ds, ps, p, lambdas, log=False, **kwargs):
    """ Optimize barycenter distance matrix using closed-form solution with the gwtil_lb.

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
    """
    S = len(Ds)
    tol = kwargs.get("tol", 1e-15)
    max_iter = kwargs.get("max_iter", 1000)

    cpt = 0
    prev_fval, cur_fval = 2**31, -2**31
    err = abs(prev_fval - cur_fval)
    Q1s = [None] * S
    Q2s = [None] * S
    gwtil_fvals = [0] * S
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
            _fval, Q1, Q2 = gwtil_lb(C, Ds[i], return_matrix=True)
            Q1s[i] = Q1
            Q2s[i] = Q2
            gwtil_fvals[i] = _fval * lambdas[i]

        cur_fval = sum(gwtil_fvals)
        err = abs(prev_fval - cur_fval)
        prev_fval = cur_fval
        fvals.append(cur_fval)
        C = update_square_loss_gwtil_lb(p, lambdas, Q1s, Q2s, Ds)
        if kwargs.get("verbose"):
            print(f"fval {cur_fval:.4f}")

    if log:
        return C, {"fun": fvals, "Q1s": Q1s, "Q2s": Q2s}
    else:
        return C


def optim_C_gwtil_lb_lb(N, Ds, ps, p, lambdas, log=False, **kwargs):
    """ Optimize barycenter distance matrix using BFGS with the gwtil_lb_lb

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
    """
    S = len(Ds)
    Ps = [None] * S

    def obj(C):
        C = C.reshape((N, N))
        losses = [0] * S
        for i in range(S):
            val, P = gwtil_lb_lb(C, Ds[i], return_matrix=True)
            # update P and loss
            Ps[i] = P
            losses[i] = val * lambdas[i]
        fval = sum(losses)

        # return fval
        grad_ = 0
        for i in range(S):
            grad_ += lambdas[i] * grad_gwtil_lb_lb_C(C, Ds[i], Ps[i])
        return fval, grad_.flatten()

    def callback(C):
        fval, grad = obj(C)
        print(f"obj {fval:.4f}", f"||g|| {np.linalg.norm(grad):.4f}")

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
                   callback=callback
                   )
    C_opt = res['x'].reshape((N, N))
    # REVIEW: update with Ps or NOT?
    # C_opt = update_square_loss_v2(p, lambdas, Ps, Ds)

    if log:
        return C_opt, res
    else:
        return C_opt


def optim_C_gwtil_lb_lb_v2(N, Ds, ps, p, lambdas, log=False, **kwargs):
    """ Optimize barycenter distance matrix using closed-form solution with the gwtil_lb_lb.

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
    """
    S = len(Ds)

    tol = kwargs.get("tol", 1e-15)
    max_iter = kwargs.get("max_iter", 1000)
    cpt = 0
    prev_fval, cur_fval = 2**31, -2**31
    err = abs(prev_fval - cur_fval)
    Ps = [None] * S
    gwtil_fvals = [0] * S
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
            _fval, P = gwtil_lb_lb(C, Ds[i], return_matrix=True)
            Ps[i] = P
            gwtil_fvals[i] = _fval * lambdas[i]

        cur_fval = sum(gwtil_fvals)
        err = abs(prev_fval - cur_fval)
        prev_fval = cur_fval
        fvals.append(cur_fval)
        C = update_square_loss_gwtil_lb_lb(p, lambdas, Ps, Ds)
        if kwargs.get("verbose"):
            print(f"fval {cur_fval:.4f}")

    if log:
        return C, {"fun": fvals, "Ps": Ps}
    else:
        return C


def eigen_projection(C, D):
    """ Using the eigen_projection to recover an aligned barycenter.

    .. math::
        C = U L U^-1 => L = U^-1 C U
        D = V L V^-1 => L = V^-1 D V
        ==> U^-1 C U = V^-1 D V
        ==> V U^-1 C U V^-1 = D
        Due P^-1 C P = D
        ==> P = U V^-1
        C_hat_star = U.T @ C_star @ U
        ==> C_star = U @ C_hat_star @ U.T + Y

        alternatively:
        C_star = \\sum_i^{m-1} \\sigma_i s_i s_i^\top
        \\sigma_i is i-th eigenvalue of C_hat
        s_i is i-th eigenvector of D_hat

    Args:
        C (np.ndarray): Distance matrix from optimal barycenter.
        D (np.ndarray): Distance matrix from sample.

    Returns:
        np.ndarray: The proper distance matrix after projection.

    References:
        * https://math.stackexchange.com/a/1163116/60983
        * https://aimath.org/knowlepedia/Beezer/

    REVIEW: check the order of eigen values / vectors
    """
    m, n = C.shape[0], D.shape[0]
    U = projection_matrix(m)
    V = projection_matrix(n)
    C_hat = U.T @ C @ U
    D_hat = V.T @ D @ V
    chat_vals, chat_vecs = eigh(C_hat)
    dhat_vals, dhat_vecs = eigh(D_hat)
    # chat_vals = chat_vals[::-1]
    # dhat_vals = dhat_vals[::-1]

    _C_hat, _D_hat = padding(C_hat, D_hat)

    _chat_vals, _chat_vecs = eigh(_C_hat)
    _dhat_vals, _dhat_vecs = eigh(_D_hat)

    # NOTE: from theorem 3, if m = n, then
    # np.testing.assert_array_almost_equal(_chat_vals, _dhat_vals)

    C_hat_star = 0
    for i in range(m - 1):
        C_hat_star += chat_vals[i] * np.outer(_dhat_vecs[:, i], _dhat_vecs[:, i])

    # DEBUG:
    # C_hat_star = 0
    # if m > n:
    #     # the active n eigen value
    #     for i in range(n - 1):
    #         C_hat_star += chat_vals[i] * np.outer(_dhat_vecs[:, i], _dhat_vecs[:, i])
    # elif m == n:
    #     for i in range(m - 1):
    #         C_hat_star += chat_vals[i] * np.outer(_dhat_vecs[:, i], _dhat_vecs[:, i])
    # else:
    #     for i in range(m - 1):
    #         C_hat_star += chat_vals[i] * np.outer(_dhat_vecs[:, i][:m - 1], _dhat_vecs[:, i][:m - 1])

    k = max(m, n)
    T = projection_matrix(k)
    C_star = T @ C_hat_star @ T.T

    # construct Y
    # NOTE: compact implementation: Y = -(d1.T + 1 d.T) / 2
    d_ = np.diag(C_star)
    Y = -(np.outer(d_, np.ones(k)) + np.outer(np.ones(k), d_)) / 2

    C_star = Y + C_star

    return C_star[:m, :m]


################################################################################
# DEPRECATED
################################################################################

def _solve_d_cvx(C):
    import cvxpy as cp
    k = C.shape[0]
    U = projection_matrix(k)
    Y = cp.Variable((k, k), symmetric=True)
    constraints = [cp.diag(C + Y) == 0, U.T @ Y @ U == 0]
    obj = cp.Minimize(cp.norm(Y, p="fro"))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    print(Y.value)
    return Y.value


def optimize_Z_gwtil_lb(N, Ds, Zs, ps, p, lambdas, Z_init=None):

    S = len(Ds)
    debug = False

    Q1s = [np.zeros(D.shape) for D in Ds]
    Q2s = [np.zeros(D.shape) for D in Ds]

    def obj(Z):
        Z = Z.reshape((N, N))
        # DEBUG: force-symmetry
        Z = (Z + Z.T) / 2
        C = cttil_Z(Z)
        T_mat = calc_T(Z)
        losses = []
        for i in range(S):
            # tr(CPDP.T)
            val, Q1, Q2 = gwtil_lb(C, Ds[i], return_matrix=True)
            Q1s[i] = Q1
            Q2s[i] = Q2
            losses.append(val * lambdas[i])
        fval = sum(losses)

        grad_ = 0
        for i in range(S):
            grad_ += lambdas[i] * np.einsum("ij,ijkl,klpq->pq",
                                            # (2 * C / N**2 - 2 * Ps[i].T @ Ds[i] @ Ps[i]),
                                            grad_gwtil_lb_C(C, Ds[i], Q1s[i], Q2s[i]),
                                            grad_C_T(T_mat),
                                            grad_T_Z(Z))
        return fval, grad_.flatten()

    def callback(Z):
        fval, grad = obj(Z)
        print(f"obj {fval:.4f}", f"||g|| {np.linalg.norm(grad):.4f}")

    if Z_init is None:
        Z_init = np.zeros(N**2)

    res = minimize(obj, Z_init, method="BFGS", jac=True, callback=callback)

    Z_opt = res['x'].reshape((N, N))
    # DEBUG: how to update Z w.r.t Q1, Q2
    # Z_opt = update_square_loss_v2(p, lambdas, Ps, Zs)
    return Z_opt


def optimize_Z_gwtil_ub(N, Ds, Zs, ps, p, lambdas, Z_init=None):
    S = len(Ds)
    debug = False

    # Ps = [np.zeros(D.shape) for D in Ds]
    Ps = [None] * S

    def obj(Z):
        Z = Z.reshape((N, N))
        # DEBUG: force-symmetry
        Z = (Z + Z.T) / 2
        C = cttil_Z(Z)
        T_mat = calc_T(Z)
        losses = []
        for i in range(S):
            # tr(CPDP.T)
            val, P = gwtil_ub(C, Ds[i], return_matrix=True, P_init=Ps[i])
            Ps[i] = P
            losses.append(val * lambdas[i])
        fval = sum(losses)

        grad_ = 0
        for i in range(S):
            grad_ += lambdas[i] * np.einsum("ij,ijkl,klpq->pq",
                                            # (2 * C / N**2 - 2 * Ps[i].T @ Ds[i] @ Ps[i]),
                                            grad_gwtil_ub_C(C, Ds[i], Ps[i]),
                                            grad_C_T(T_mat),
                                            grad_T_Z(Z))
        return fval, grad_.flatten()

    def callback(Z):
        fval, grad = obj(Z)
        print(f"obj {fval:.4f}", f"||g|| {np.linalg.norm(grad):.4f}")

    if Z_init is None:
        # Z_init = np.zeros(N**2)
        Z_init = np.random.rand(N**2)

    res = minimize(obj, Z_init, method="BFGS", jac=True, callback=callback)

    Z_opt = res['x'].reshape((N, N))
    # DEBUG: having the update or not
    Z_opt = update_square_loss_gwtil_ub(p, lambdas, Ps, Zs)
    return Z_opt
