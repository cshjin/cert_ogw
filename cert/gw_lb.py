################################################################################
# Copyright (c) Thomas Giacomo Nies
# Reference: https://github.com/gnies/Gromov-Wasserstein-Lower-Bounds
#
# Note: simplified API without (x, y)
#
# Updates: with POT ver 0.8.1, there is no need to apply the order `p`
################################################################################

import numpy as np
import ot
from ot.lp import emd2_1d


def wasserstein_1d(x_a, x_b, a=None, b=None, p=1., log=False):
    """ p-th order Wasserstein distance.

    Args:
        x_a (np.ndarray): Source diract locations with dim (m, ).
        x_b (np.ndarray): Target diract locations with dim (n, ).
        a (np.ndarray, optional): Source histogram with dim (m, ).
            Defaults to None (uniform weight).
        b (np.ndarray, optional): Target histogram with dim (n, ).
            Defaults to None (uniform weight).
        p (float, optional): The order of the p-Wasserstein distance to be computed. Defaults to `1.`.
        log (bool, optional): If True, returns a dictionary containing the transportation matrix.
            Otherwise returns only the loss. Defaults to False.

    Returns:
        float: p-Wasserstein distance.

    Note:
        This is the same implementation as `wasserstein` function, with additional
        log to be the option for return.
        `ot` use the compiled `cpython.so` file as the internal solver.
    """
    cost_emd, log = emd2_1d(x_a=x_a, x_b=x_b, a=a, b=b, metric='minkowski', p=p,
                            dense=False, log=True)
    log['dist'] = cost_emd
    if log:
        return np.power(cost_emd, 1. / p), log
    else:
        return np.power(cost_emd, 1. / p)


def wasserstein(x, y, p=2, a=None, b=None):
    """ p-th order wasserstein distance.

    .. math::
        \\min_T [\\sum_i \\sum_j T_ij ||x_i - x_j||^p]^{1/p}

        s.t. T 1 = a,
             T.T 1 = b,
             T \\geq 0

    Args:
        x (list-like array): source list.
        y (list-like array): target list.
        p (float, optional): p-th order. Defaults to 2.
        a (np.array, optional): distribution on source list. Defaults to None.
        b (np.array, optional): distribution on target list. Defaults to None.

    Returns:
        float: wasserstein distance

    See also:
        `ot.wasserstein_1d`

    Reference:
        * [1] Peyré, G., & Cuturi, M. (2017). “Computational Optimal Transport”, 2018.
    """
    m = len(x)
    n = len(y)
    if a is None:
        a = np.ones(m) / m
    if b is None:
        b = np.ones(n) / n
    #  cumulative distirbutions
    ca = np.cumsum(a)
    cb = np.cumsum(b)

    # points on which we need to evaluate the quantile functions
    cba = np.sort(np.hstack([ca, cb]))

    # construction of first quantile function
    # bins need some small tollerance to avoid numerical rounding errors
    bins = ca + 1e-10
    bins = np.hstack([-np.Inf, bins, np.Inf])
    # right=True becouse quantile function is right continuous.
    index_qx = np.digitize(cba, bins, right=True) - 1
    x = np.sort(x)
    qx = x[index_qx]

    # constuction of second quantile function
    bins = cb + 1e-10
    bins = np.hstack([-np.Inf, bins, np.Inf])
    # right=True becouse quantile function is right continuous.
    index_qy = np.digitize(cba, bins, right=True) - 1
    y = np.sort(y)
    qy = y[index_qy]

    # weights for the integral
    h = np.diff(np.hstack([0, cba]))

    # evaluation of integral
    res = np.sum(np.abs(qy - qx)**p * h)**(1 / p)
    return res


def flb(d_x, d_y, p=2., a=None, b=None):
    """ First-order lower bound of GW.

    Args:
        d_x (np.ndarray): Distance matrix in the source domain with dim (m, m).
        d_y (np.ndarray): Distance matrix in the target domain with dim (n, n)
        p (float, optional): The order of the p-Wasserstein distance to be computed. Defaults to 2.
        a (np.ndarray, optional): Source histogram with dim (m, ).
            Defaults to None (uniform weight).
        b (np.ndarray, optional): Target histogram with dim (n, ).
            Defaults to None (uniform weight).

    Returns:
        float: First-order lower bound of GW.

    Notes:
        Time complexity: O(n^2)

    References:
        * Mémoli, Facundo. "Gromov–Wasserstein distances and the metric approach to object matching."
        Foundations of computational mathematics 11.4 (2011): 417-487.

    See Also:
        `slb`, `tlb`
    """
    m = len(d_x)
    n = len(d_y)
    if a is None:
        a = np.ones(m) / m
    if b is None:
        b = np.ones(n) / n
    # computing eccentricities
    e_x = np.sum(d_x**p * a, axis=1)**(1 / p)
    e_y = np.sum(d_y**p * b, axis=1)**(1 / p)

    # res = wasserstein(e_x, e_y, p, a, b)
    # REVIEW: replace with ot
    res = ot.wasserstein_1d(e_x, e_y, a, b, p)
    return res


def slb(d_x, d_y, p=2., a=None, b=None):
    """ Second-order lower bound of GW.

    Args:
        d_x (np.ndarray): Distance matrix in the source domain with dim (m, m).
        d_y (np.ndarray): Distance matrix in the target domain with dim (n, n)
        p (float, optional): The order of the p-Wasserstein distance to be computed. Defaults to 2.
        a (np.ndarray, optional): Source histogram with dim (m, ).
            Defaults to None (uniform weight).
        b (np.ndarray, optional): Target histogram with dim (n, ).
            Defaults to None (uniform weight).

    Returns:
        float: Second-order lower bound of GW.

    Notes:
        Time complexity: O(n^4)

    References:
        * Mémoli, Facundo. "Gromov–Wasserstein distances and the metric approach to object matching."
        Foundations of computational mathematics 11.4 (2011): 417-487.

    See Also:
        `flb`, `tlb`
    """
    m = len(d_x)
    n = len(d_y)
    if a is None:
        a = np.ones(m) / m
    if b is None:
        b = np.ones(n) / n
    # product measures are created
    aa = a.reshape((1, -1)) * a.reshape((-1, 1))
    bb = b.reshape((1, -1)) * b.reshape((-1, 1))

    # now everything gets flattened..
    d_x = d_x.reshape((-1))
    d_y = d_y.reshape((-1))
    aa = aa.reshape((-1))
    bb = bb.reshape((-1))
    # res = wasserstein(d_x, d_y, p, a=aa, b=bb)
    # REVIEW: replace with ot
    res = ot.wasserstein_1d(d_x, d_y, aa, bb, p)
    return res


def tlb(d_x, d_y, p=2., a=None, b=None):
    """ Second-order lower bound of GW.

    Args:
        d_x (np.ndarray): Distance matrix in the source domain with dim (m, m).
        d_y (np.ndarray): Distance matrix in the target domain with dim (n, n)
        p (float, optional): The order of the p-Wasserstein distance to be computed. Defaults to 2.
        a (np.ndarray, optional): Source histogram with dim (m, ).
            Defaults to None (uniform weight).
        b (np.ndarray, optional): Target histogram with dim (n, ).
            Defaults to None (uniform weight).

    Returns:
        float: Second-order lower bound of GW.

    Notes:
        Time complexity: O(n^5)

    References:
        * Mémoli, Facundo. "Gromov–Wasserstein distances and the metric approach to object matching."
        Foundations of computational mathematics 11.4 (2011): 417-487.

    See Also:
        `flb`, `slb`
    """
    m = len(d_x)
    n = len(d_y)
    if a is None:
        a = np.ones(m) / m
    if b is None:
        b = np.ones(n) / n
    #  cumulative distirbutions
    ca = np.cumsum(a)
    cb = np.cumsum(b)

    # points on which we need to evaluate the quantile functions
    cba = np.sort(np.hstack([ca, cb]))

    # construction of first quantile function
    bins = ca + 1e-10  # bins need some small tollerance to avoid numerical rounding errors
    bins = np.hstack([-np.Inf, bins, np.Inf])
    index_qx = np.digitize(cba, bins, right=True) - 1    # right=True becouse quantile function is right continuous.

    # constuction of second quantile function
    bins = cb + 1e-10
    bins = np.hstack([-np.Inf, bins, np.Inf])
    index_qy = np.digitize(cba, bins, right=True) - 1    # right=True becouse quantile function is right continuous.

    d_x = np.sort(d_x, axis=1)
    d_y = np.sort(d_y, axis=1)

    # quantiles of the r.v. d_x(X, x_i) evaluated in the points cba
    qx = d_x[:, index_qx]
    qy = d_y[:, index_qy]

    # weights for the inegral
    h = np.diff(np.hstack([0, cba]))

    # Evaluation of integral in different locations
    # I use a loop to avoid potential memory problems that could arise if qy - qx is vectorized
    omega = np.empty((m, n))
    for i in range(m):
        # if i%50 == 0:
        #     print("iteration ", i)
        integrand = np.abs(qy - qx[i, :])**p * h  # notice that p-root is omitted
        omega_i = np.sum(integrand, axis=1)
        omega[i, :] = omega_i
    # omega = np.sum(np.abs(qy.reshape((1, n, m, -1)) - qx.reshape((n, 1, n, m))**p * h )**(1/p)
    plan = ot.emd(a, b, omega)
    cost = np.sum(plan * omega)
    return cost
