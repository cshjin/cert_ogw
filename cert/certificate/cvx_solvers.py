import cvxpy as cp
import numpy as np


def cvxpy_solve_A(U, A_org, gamma, delta_g, Z=None, delta_l=None, A_init=None):
    """ Solve A in co A with SDP solver.

    .. math::
        min_{A \\in co \\Acal}
            tr(U^T A) - gamma^2 logdet(\\phi(A))
                s.t. ||A - A_org||_1 \\le 2 * delta_g

        where \\phi(A) = -A + diag(A1) + 11^T/n

    Args:
        U (np.ndarray): Input matrix.
        A_org (np.ndarray): Original adjacency matrix.
        gamma (float): Input constant.
        delta_g (int): Global budget.
        Z (np.ndarray, optional): Additional term.
        delta_l (np.array, optional): Local budget.

    Raises:
        Exception: Unable to solve the problem.

    Returns:
        tuple: (float): Optimal function value.
               (np.ndarray): Optimal X.

    Note:
        With the additional input of Z, the objective function has one more term
        `- gamma^2 (Z^T \\phi(A))`

    See Also:
        `cvxpy_solve_X`: they should be equivalent with each other, but some numerical differences.
    """
    n = A_org.shape[0]
    eet = np.ones((n, n))
    A_var = cp.Variable(U.shape, symmetric=True)
    warm_start = False
    if isinstance(A_init, np.ndarray):
        A_var.value = A_init
        warm_start = True
    Phi_A = -A_var + cp.diag(cp.sum(A_var, 1)) + eet / n
    loss = - gamma * cp.log_det(Phi_A) + cp.trace(U.T @ A_var)
    # have additional term
    if isinstance(Z, np.ndarray):
        loss -= gamma * cp.trace(Z.T @ Phi_A)
    const = [cp.diag(A_var) == 0,
             cp.sum(A_var - A_org) <= 2 * delta_g,
             A_var - A_org >= 0,
             A_var - A_org <= 1]
    # have additional local budget
    if isinstance(delta_l, np.ndarray):
        const += [cp.sum(A_var - A_org, 0) <= delta_l]
    # DEBUG: failed in bundle method, DCP error from cvxpy
    obj = cp.Minimize(loss)
    problem = cp.Problem(obj, const)
    problem.solve(cp.MOSEK, warm_start=warm_start)
    if problem.status not in ['infeasible', 'unbounded']:
        A_opt = A_var.value
        fval_opt = obj.value
        return fval_opt, A_opt
    else:
        raise Exception


def cvxpy_solve_X(U, K, gamma, delta_g):
    """ Solve X in co X in SDP solver.

    .. math::
        \\min_{X \\in co \\Xcal}
            tr(U^T X) - gamma logdet(C(X) + K)
                s.t. ||X||_1 \\le 2 * delta_g

    Args:
        U (np.ndarray): Input matrix.
        K (np.ndarray): Input matrix.
        gamma (float): Input constant.
        delta_g (int): Global budget.

    Raises:
        Exception: Unable to solve the problem.

    Returns:
        tuple: (float): Optimal function value.
               (np.ndarray): Optimal X.
    """
    X = cp.Variable(U.shape, symmetric=True)
    CX = -X + cp.diag(cp.sum(X, 1))
    loss = cp.trace(U.T @ X) - gamma * cp.log_det(CX + K)
    const = [X >= 0, X <= 1, cp.diag(X) == 0, cp.sum(X) <= 2 * delta_g]
    obj = cp.Minimize(loss)
    problem = cp.Problem(obj, const)
    # solver: SCS / MOSEK
    problem.solve(cp.MOSEK)
    if problem.status not in ["infeasible", "unbounded"]:
        X_opt = X.value
        fval_opt = obj.value
        return fval_opt, X_opt
    else:
        raise Exception
