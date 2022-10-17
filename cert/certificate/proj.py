# Copyright (c) Adrien Gaidon, 2011.
# https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246
import numpy as np


def proj_simplex_l2(M, budget):
    """ Minimum L2 projection of vector onto the probability simplex
    .. math::
        proj(M) := ||M||_2 <= budget
    """
    m = M.flatten()
    N = m.shape[0]
    mu = sorted(m, reverse=True)
    sm = 0
    row = -1
    sm_row = sm
    for i in range(1, N + 1):
        sm = sm + mu[i - 1]
        if mu[i - 1] - (1 / i) * (sm - 1) > 0:
            row = i
            sm_row = sm
    theta = np.maximum(0, (1 / row) * (sm_row - budget))
    w = np.maximum(m - theta, 0)
    return w.reshape(M.shape)


def proj_simplex_l1(M, budget):
    """ Minimum L1 Projection of vector onto
    .. math::
        proj(M) := ||M||_1 <= budget

    Note:
    """
    v = M.flatten()
    u = np.abs(v)
    if u.sum() <= budget:
        return v.reshape(M.shape)
    w = euclidean_proj_simplex(u, budget)
    w *= np.sign(v)
    return w.reshape(M.shape)


def euclidean_proj_simplex(v, budget):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \\sum_i w_i = s, w_i >= 0

    Parameters
    ----------
        v: (n,) numpy array,
        n-dimensional vector to project
        s: int, optional, default: 1,
        radius of the simplex

    Returns
    -------
        w: (n,) numpy array,
        Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert budget > 0, "Radius s must be strictly positive (%d <= 0)" % budget
    n, = v.shape  #
    # check if we are already on the simplex
    if v.sum() == budget and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - budget))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - budget) / rho
    # compute the projection by thresholding v using theta
    w = np.maximum(v - theta, 0)
    return w
