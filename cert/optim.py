# -*- coding: utf-8 -*-
"""
Optimization algorithms for OT
"""


import operator
import warnings

import numpy as np
from ot.optim import line_search_armijo
from scipy._lib._util import check_random_state
from scipy.optimize import (OptimizeResult, OptimizeWarning,
                            linear_sum_assignment)
from ot.lp import emd
from ot.optim import solve_1d_linesearch_quad


class StopError(Exception):
    pass


class NonConvergenceError(Exception):
    pass


def cg(a, b, M, reg, f, df, G0=None, numItermax=200, numItermaxEmd=100000,
       stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False, **kwargs):
    """
    Solve the general regularized OT problem with conditional gradient

        The function solves the following optimization problem:

    .. math::
        \\gamma = arg\\min_\\gamma <\\gamma,M>_F + reg*f(\\gamma)

        s.t. \\gamma 1 = a

             \\gamma^T 1= b

             \\gamma\\geq 0
    where :

    - M is the (ns,nt) metric cost matrix
    - :math:`f` is the regularization term ( and df is its gradient)
    - a and b are source and target weights (sum to 1)

    The algorithm used for solving the problem is conditional gradient as discussed in  [1]_


    Parameters
    ----------
    a : ndarray, shape (ns,)
        samples weights in the source domain
    b : ndarray, shape (nt,)
        samples in the target domain
    M : ndarray, shape (ns, nt)
        loss matrix
    reg : float
        Regularization term >0
    G0 :  ndarray, shape (ns,nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    numItermaxEmd : int, optional
        Max number of iterations for emd
    stopThr : float, optional
        Stop threshol on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshol on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    **kwargs : dict
             Parameters for linesearch

    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.

    See Also
    --------
    ot.lp.emd : Unregularized optimal ransport
    ot.bregman.sinkhorn : Entropic regularized optimal transport

    """

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        G = np.outer(a, b)
    else:
        G = G0

    def cost(G):
        return np.sum(M * G) + reg * f(G)

    f_val = cost(G)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, 0, 0))

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        Mi = M + reg * df(G)
        # set M positive
        Mi += Mi.min()

        # solve linear program
        Gc = emd(a, b, Mi, numItermax=numItermaxEmd)

        deltaG = Gc - G

        # line search
        alpha, fc, f_val = solve_linesearch(cost, G, deltaG, Mi, f_val, reg=reg, M=M, Gc=Gc, **kwargs)

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)
        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, relative_delta_fval, abs_delta_fval))

    if log:
        return G, log
    else:
        return G


def solve_linesearch(cost, G, deltaG, Mi, f_val,
                     armijo=True, C1=None, C2=None, reg=None, Gc=None, constC=None, M=None, **kwargs):
    """
    Solve the linesearch in the FW iterations
    Parameters
    ----------
    cost : method
        Cost in the FW for the linesearch
    G : ndarray, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : ndarray (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    Mi : ndarray (ns,nt)
        Cost matrix of the linearized transport problem. Corresponds to the gradient of the cost
    f_val :  float
        Value of the cost at G
    armijo : bool, optional
            If True the steps of the line-search is found via an armijo research. Else closed form is used.
            If there is convergence issues use False.
    C1 : ndarray (ns,ns), optional
        Structure matrix in the source domain. Only used and necessary when armijo=False
    C2 : ndarray (nt,nt), optional
        Structure matrix in the target domain. Only used and necessary when armijo=False
    reg : float, optional
          Regularization parameter. Only used and necessary when armijo=False
    Gc : ndarray (ns,nt)
        Optimal map found by linearization in the FW algorithm. Only used and necessary when armijo=False
    constC : ndarray (ns,nt)
             Constant for the gromov cost. See [24]. Only used and necessary when armijo=False
    M : ndarray (ns,nt), optional
        Cost matrix between the features. Only used and necessary when armijo=False
    Returns
    -------
    alpha : float
            The optimal step size of the FW
    fc : int
         nb of function call. Useless here
    f_val :  float
             The value of the cost for the next iteration
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    if armijo:
        alpha, fc, f_val = line_search_armijo(cost, G, deltaG, Mi, f_val)
    else:  # requires symetric matrices
        dot1 = np.dot(C1, deltaG)
        dot12 = dot1.dot(C2)
        a = -2 * reg * np.sum(dot12 * deltaG)
        b = np.sum((M + reg * constC) * deltaG) - 2 * reg * (np.sum(dot12 * G) + np.sum(np.dot(C1, G).dot(C2) * deltaG))
        c = cost(G)

        alpha = solve_1d_linesearch_quad(a, b, c)
        fc = None
        f_val = cost(G + alpha * deltaG)

    return alpha, fc, f_val


def solve_1d_linesearch_quad_funct(a, b, c):
    # solve min f(x)=a*x**2+b*x+c sur 0,1
    f0 = c
    df0 = b
    f1 = a + f0 + df0

    if a > 0:  # convex
        minimum = min(1, max(0, -b / (2 * a)))
        # print('entrelesdeux')
        return minimum
    else:  # non convexe donc sur les coins
        if f0 > f1:
            # print('sur1 f(1)={}'.format(f(1)))
            return 1
        else:
            # print('sur0 f(0)={}'.format(f(0)))
            return 0


def do_linesearch(cost, G, deltaG, Mi, f_val, amijo=True, C1=None, C2=None, reg=None, Gc=None, constC=None, M=None):
    """
    Solve the linesearch in the FW iterations

    Gc = st
    G = xt
    deltaG = st - xt
    Gc+alpha * deltaG = st + alpha(st - xt)

    Parameters
    ----------
    cost : method
        The FGW cost
    G : ndarray, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : ndarray (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    Mi : ndarray (ns,nt)
        Cost matrix of the linearized transport problem. Corresponds to the gradient of the cost
    f_val :  float
        Value of the cost at G
    amijo : bool, optionnal
            If True the steps of the line-search is found via an amijo research. Else closed form is used.
            If there is convergence issues use False.
    C1 : ndarray (ns,ns), optionnal
        Structure matrix in the source domain. Only used when amijo=False
    C2 : ndarray (nt,nt), optionnal
        Structure matrix in the target domain. Only used when amijo=False
    reg : float, optionnal
          Regularization parameter. Corresponds to the alpha parameter of FGW. Only used when amijo=False
    Gc : ndarray (ns,nt)
        Optimal map found by linearization in the FW algorithm. Only used when amijo=False
    constC : ndarray (ns,nt)
             Constant for the gromov cost. See [3]. Only used when amijo=False
    M : ndarray (ns,nt), optionnal
        Cost matrix between the features. Only used when amijo=False
    Returns
    -------
    alpha : float
            The optimal step size of the FW
    fc : useless here
    f_val :  float
             The value of the cost for the next iteration
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    if amijo:
        alpha, fc, f_val = line_search_armijo(cost, G, deltaG, Mi, f_val)
    else:
        dot1 = np.dot(C1, deltaG)
        dot12 = dot1.dot(C2)  # C1 dt C2
        a = -2 * reg * np.sum(dot12 * deltaG)  # -2*alpha*<C1 dt C2,dt> si qqlun est pas bon c'est lui
        b = np.sum((M + reg * constC) * deltaG) - 2 * reg * (np.sum(dot12 * G) + np.sum(np.dot(C1, G).dot(C2) * deltaG))
        c = cost(G)  # f(xt)

        alpha = solve_1d_linesearch_quad_funct(a, b, c)
        fc = None
        f_val = cost(G + alpha * deltaG)

    return alpha, fc, f_val


def fast_quadratic_assignment(A, B,
                              maximize=False, partial_match=None, rng=None,
                              P0="barycenter", shuffle_input=False, maxiter=30,
                              tol=0.03, **unknown_options):
    r"""Solve the quadratic assignment problem (approximately).
    This function solves the Quadratic Assignment Problem (QAP) and the
    Graph Matching Problem (GMP) using the Fast Approximate QAP Algorithm
    (FAQ) [1]_.
    Quadratic assignment solves problems of the following form:
    .. math::
        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}\\
    where :math:`\mathcal{P}` is the set of all permutation matrices,
    and :math:`A` and :math:`B` are square matrices.
    Graph matching tries to *maximize* the same objective function.
    This algorithm can be thought of as finding the alignment of the
    nodes of two graphs that minimizes the number of induced edge
    disagreements, or, in the case of weighted graphs, the sum of squared
    edge weight differences.
    Note that the quadratic assignment problem is NP-hard. The results given
    here are approximations and are not guaranteed to be optimal.
    Parameters
    ----------
    A : 2-D array, square
        The square matrix :math:`A` in the objective function above.
    B : 2-D array, square
        The square matrix :math:`B` in the objective function above.
    method :  str in {'faq', '2opt'} (default: 'faq')
        The algorithm used to solve the problem. This is the method-specific
        documentation for 'faq'.
        :ref:`'2opt' <optimize.qap-2opt>` is also available.
    Options
    -------
    maximize : bool (default: False)
        Maximizes the objective function if ``True``.
    partial_match : 2-D array of integers, optional (default: None)
        Fixes part of the matching. Also known as a "seed" [2]_.
        Each row of `partial_match` specifies a pair of matched nodes:
        node ``partial_match[i, 0]`` of `A` is matched to node
        ``partial_match[i, 1]`` of `B`. The array has shape ``(m, 2)``, where
        ``m`` is not greater than the number of nodes, :math:`n`.
    rng : {None, int, `numpy.random.Generator`,
           `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    P0 : 2-D array, "barycenter", or "randomized" (default: "barycenter")
        Initial position. Must be a doubly-stochastic matrix [3]_.
        If the initial position is an array, it must be a doubly stochastic
        matrix of size :math:`m' \times m'` where :math:`m' = n - m`.
        If ``"barycenter"`` (default), the initial position is the barycenter
        of the Birkhoff polytope (the space of doubly stochastic matrices).
        This is a :math:`m' \times m'` matrix with all entries equal to
        :math:`1 / m'`.
        If ``"randomized"`` the initial search position is
        :math:`P_0 = (J + K) / 2`, where :math:`J` is the barycenter and
        :math:`K` is a random doubly stochastic matrix.
    shuffle_input : bool (default: False)
        Set to `True` to resolve degenerate gradients randomly. For
        non-degenerate gradients this option has no effect.
    maxiter : int, positive (default: 30)
        Integer specifying the max number of Frank-Wolfe iterations performed.
    tol : float (default: 0.03)
        Tolerance for termination. Frank-Wolfe iteration terminates when
        :math:`\frac{||P_{i}-P_{i+1}||_F}{\sqrt{m')}} \leq tol`,
        where :math:`i` is the iteration number.
    Returns
    -------
    res : OptimizeResult
        `OptimizeResult` containing the following fields.
        col_ind : 1-D array
            Column indices corresponding to the best permutation found of the
            nodes of `B`.
        fun : float
            The objective value of the solution.
        nit : int
            The number of Frank-Wolfe iterations performed.
    Notes
    -----
    The algorithm may be sensitive to the initial permutation matrix (or
    search "position") due to the possibility of several local minima
    within the feasible region. A barycenter initialization is more likely to
    result in a better solution than a single random initialization. However,
    calling ``quadratic_assignment`` several times with different random
    initializations may result in a better optimum at the cost of longer
    total execution time.
    Examples
    --------
    As mentioned above, a barycenter initialization often results in a better
    solution than a single random initialization.
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> n = 15
    >>> A = rng.random((n, n))
    >>> B = rng.random((n, n))
    >>> res = quadratic_assignment(A, B)  # FAQ is default method
    >>> print(res.fun)
    46.871483385480545  # may vary
    >>> options = {"P0": "randomized"}  # use randomized initialization
    >>> res = quadratic_assignment(A, B, options=options)
    >>> print(res.fun)
    47.224831071310625 # may vary
    However, consider running from several randomized initializations and
    keeping the best result.
    >>> res = min([quadratic_assignment(A, B, options=options)
    ...            for i in range(30)], key=lambda x: x.fun)
    >>> print(res.fun)
    46.671852533681516 # may vary
    The '2-opt' method can be used to further refine the results.
    >>> options = {"partial_guess": np.array([np.arange(n), res.col_ind]).T}
    >>> res = quadratic_assignment(A, B, method="2opt", options=options)
    >>> print(res.fun)
    46.47160735721583 # may vary
    References
    ----------
    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik,
           S.G. Kratzer, E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and
           C.E. Priebe, "Fast approximate quadratic programming for graph
           matching," PLOS one, vol. 10, no. 4, p. e0121002, 2015,
           :doi:`10.1371/journal.pone.0121002`
    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,
           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):
           203-215, :doi:`10.1016/j.patcog.2018.09.014`
    .. [3] "Doubly stochastic Matrix," Wikipedia.
           https://en.wikipedia.org/wiki/Doubly_stochastic_matrix
    """

    def _check_unknown_options(unknown_options):
        if unknown_options:
            msg = ", ".join(map(str, unknown_options.keys()))
            # Stack level 4: this is called from _minimize_*, which is
            # called from another function in SciPy. Level 4 is the first
            # level in user code.
            warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)

    def _common_input_validation(A, B, partial_match):
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)

        if partial_match is None:
            partial_match = np.array([[], []]).T
        partial_match = np.atleast_2d(partial_match).astype(int)

        msg = None
        if A.shape[0] != A.shape[1]:
            msg = "`A` must be square"
        elif B.shape[0] != B.shape[1]:
            msg = "`B` must be square"
        elif A.ndim != 2 or B.ndim != 2:
            msg = "`A` and `B` must have exactly two dimensions"
        elif A.shape != B.shape:
            msg = "`A` and `B` matrices must be of equal size"
        elif partial_match.shape[0] > A.shape[0]:
            msg = "`partial_match` can have only as many seeds as there are nodes"
        elif partial_match.shape[1] != 2:
            msg = "`partial_match` must have two columns"
        elif partial_match.ndim != 2:
            msg = "`partial_match` must have exactly two dimensions"
        elif (partial_match < 0).any():
            msg = "`partial_match` must contain only positive indices"
        elif (partial_match >= len(A)).any():
            msg = "`partial_match` entries must be less than number of nodes"
        elif (not len(set(partial_match[:, 0])) == len(partial_match[:, 0]) or
              not len(set(partial_match[:, 1])) == len(partial_match[:, 1])):
            msg = "`partial_match` column entries must be unique"

        if msg is not None:
            raise ValueError(msg)

        return A, B, partial_match

    def _doubly_stochastic(P, tol=1e-3):
        # Adapted from @btaba implementation
        # https://github.com/btaba/sinkhorn_knopp
        # of Sinkhorn-Knopp algorithm
        # https://projecteuclid.org/euclid.pjm/1102992505

        max_iter = 1000
        c = 1 / P.sum(axis=0)
        r = 1 / (P @ c)
        P_eps = P

        for it in range(max_iter):
            if ((np.abs(P_eps.sum(axis=1) - 1) < tol).all() and
                    (np.abs(P_eps.sum(axis=0) - 1) < tol).all()):
                # All column/row sums ~= 1 within threshold
                break

            c = 1 / (r @ P)
            r = 1 / (P @ c)
            P_eps = r[:, None] * P * c

        return P_eps

    def _calc_score(A, B, perm):
        # equivalent to objective function but avoids matmul
        return np.sum(A * B[perm][:, perm])

    def _split_matrix(X, n):
        # definitions according to Seeded Graph Matching [2].
        upper, lower = X[:n], X[n:]
        return upper[:, :n], upper[:, n:], lower[:, :n], lower[:, n:]

    _check_unknown_options(unknown_options)

    maxiter = operator.index(maxiter)

    # ValueError check
    A, B, partial_match = _common_input_validation(A, B, partial_match)

    msg = None
    if isinstance(P0, str) and P0 not in {'barycenter', 'randomized'}:
        msg = "Invalid 'P0' parameter string"
    elif maxiter <= 0:
        msg = "'maxiter' must be a positive integer"
    elif tol <= 0:
        msg = "'tol' must be a positive float"
    if msg is not None:
        raise ValueError(msg)

    rng = check_random_state(rng)
    n = len(A)  # number of vertices in graphs
    n_seeds = len(partial_match)  # number of seeds
    n_unseed = n - n_seeds

    # [1] Algorithm 1 Line 1 - choose initialization
    if not isinstance(P0, str):
        P0 = np.atleast_2d(P0)
        if P0.shape != (n_unseed, n_unseed):
            msg = "`P0` matrix must have shape m' x m', where m'=n-m"
        elif ((P0 < 0).any() or not np.allclose(np.sum(P0, axis=0), 1)
              or not np.allclose(np.sum(P0, axis=1), 1)):
            msg = "`P0` matrix must be doubly stochastic"
        if msg is not None:
            raise ValueError(msg)
    elif P0 == 'barycenter':
        P0 = np.ones((n_unseed, n_unseed)) / n_unseed
    elif P0 == 'randomized':
        J = np.ones((n_unseed, n_unseed)) / n_unseed
        # generate a nxn matrix where each entry is a random number [0, 1]
        # would use rand, but Generators don't have it
        # would use random, but old mtrand.RandomStates don't have it
        K = _doubly_stochastic(rng.uniform(size=(n_unseed, n_unseed)))
        P0 = (J + K) / 2

    # check trivial cases
    if n == 0 or n_seeds == n:
        score = _calc_score(A, B, partial_match[:, 1])
        res = {"col_ind": partial_match[:, 1], "fun": score, "nit": 0}
        return OptimizeResult(res)

    obj_func_scalar = 1
    if maximize:
        obj_func_scalar = -1

    nonseed_B = np.setdiff1d(range(n), partial_match[:, 1])
    if shuffle_input:
        nonseed_B = rng.permutation(nonseed_B)

    nonseed_A = np.setdiff1d(range(n), partial_match[:, 0])
    perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
    perm_B = np.concatenate([partial_match[:, 1], nonseed_B])

    # definitions according to Seeded Graph Matching [2].
    A11, A12, A21, A22 = _split_matrix(A[perm_A][:, perm_A], n_seeds)
    B11, B12, B21, B22 = _split_matrix(B[perm_B][:, perm_B], n_seeds)
    const_sum = A21 @ B21.T + A12.T @ B12

    P = P0
    # [1] Algorithm 1 Line 2 - loop while stopping criteria not met
    for n_iter in range(1, maxiter + 1):
        # [1] Algorithm 1 Line 3 - compute the gradient of f(P) = -tr(APB^tP^t)
        grad_fp = (const_sum + A22 @ P @ B22.T + A22.T @ P @ B22)
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        _, cols = linear_sum_assignment(grad_fp, maximize=maximize)
        Q = np.eye(n_unseed)[cols]

        # [1] Algorithm 1 Line 5 - compute the step size
        # Noting that e.g. trace(Ax) = trace(A)*x, expand and re-collect
        # terms as ax**2 + bx + c. c does not affect location of minimum
        # and can be ignored. Also, note that trace(A@B) = (A.T*B).sum();
        # apply where possible for efficiency.
        R = P - Q
        b21 = ((R.T @ A21) * B21).sum()
        b12 = ((R.T @ A12.T) * B12.T).sum()
        AR22 = A22.T @ R
        BR22 = B22 @ R.T
        b22a = (AR22 * B22.T[cols]).sum()
        b22b = (A22 * BR22[cols]).sum()
        a = (AR22.T * BR22).sum()
        b = b21 + b12 + b22a + b22b
        # critical point of ax^2 + bx + c is at x = -d/(2*e)
        # if a * obj_func_scalar > 0, it is a minimum
        # if minimum is not in [0, 1], only endpoints need to be considered
        if a * obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = np.argmin([0, (b + a) * obj_func_scalar])

        # [1] Algorithm 1 Line 6 - Update P
        P_i1 = alpha * P + (1 - alpha) * Q
        if np.linalg.norm(P - P_i1) / np.sqrt(n_unseed) < tol:
            P = P_i1
            break
        P = P_i1
    # [1] Algorithm 1 Line 7 - end main loop

    # REVIEW: remove
    # [1] Algorithm 1 Line 8 - project onto the set of permutation matrices
    # _, col = linear_sum_assignment(P, maximize=True)
    # perm = np.concatenate((np.arange(n_seeds), col + n_seeds))

    unshuffled_perm = np.zeros(n, dtype=int)
    # unshuffled_perm[perm_A] = perm_B[perm]

    # score = _calc_score(A, B, unshuffled_perm)
    score = np.trace(A @ P @ B @ P.T)
    res = {"col_ind": unshuffled_perm, "fun": score, "nit": n_iter, "x": P}
    return OptimizeResult(res)


if __name__ == "__main__":
    from fgw.gromov_prox import quad_solver
    A = np.random.rand(10, 10)
    A += A.T
    np.fill_diagonal(A, 0)

    B = np.random.rand(10, 10)
    B += B.T
    np.fill_diagonal(B, 0)

    # print(A, B)
    res = fast_quadratic_assignment(A, B)
    print(res['fun'])

    res = quad_solver(A, B, domain="O")
    print(res)
