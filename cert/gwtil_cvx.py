import numpy as np
from scipy.linalg import svd
from scipy.optimize import check_grad, minimize

from fgw.gromov_prox import eig_decom
from fgw.gwtil import Qcal_ub, projection_matrix
from fgw.gwtil_bary import grad_Qcal_ub_C
from fgw.utils import sym

norm = np.linalg.norm


def omega_circ(C, D, return_matrix=False, **kwargs):
    """ Optimize the Omega_circ given C and D.
    A convex lower bound of the original OGW.

    .. math::
        \\Omega_{\\circ} (C)
        = \\frac{1}{n^2} \\max_{R} (tr(R.T C) - \\frac{1}{4} ||R||_F^2 - Qcal_ub(R, D)).

    Args:
        C (np.ndarray): Distance matrix in source domain.
        D (np.ndarray): Distance matrix in target domain.
        return_matrix (bool, optional): Return optimal R if `True`.

    Returns:
        (tuple):
            float: optimal function value.
            np.ndarray, optional: optimal S if `return_matrix` is `True`.
    """
    assert C.shape[0] == D.shape[0]
    n = D.shape[0]
    VERBOSE = kwargs.get("verbose", False)

    def obj(R):
        R = R.reshape((n, -1))
        # NOTE: force R in the symmetric domain, dom f^* \in \Scal
        R = sym(R)
        fval_Qcal_ub, Q1, Q2 = Qcal_ub(R, D, return_matrix=True)
        fval = np.trace(R.T @ C) - 1 / 4 * norm(R)**2 - fval_Qcal_ub

        grad = C - R / 2 - grad_Qcal_ub_C(D, Q1, Q2)
        return -fval / n**2, -grad.flatten() / n**2

    def callback(R):
        R = R.reshape((n, -1))
        fval, grad = obj(R)
        print(f"fval {-fval:.4f} ||g|| {norm(grad):.4f}")

    R_init = np.random.rand(n**2)
    ''' check grad '''
    if kwargs.get("check_grad"):
        grad_diff = check_grad(lambda x: obj(x)[0], lambda x: obj(x)[1], R_init)
        print(f"||g||_diff {grad_diff:.2e}")

    res = minimize(obj, R_init,
                   method="BFGS",
                   jac=True,
                   callback=callback if VERBOSE else None,
                   #    options={"disp": 1}
                   )
    fval_opt = -res['fun']
    R_opt = res['x'].reshape((n, -1))

    if return_matrix:
        return fval_opt, R_opt
    else:
        return fval_opt


def omega_circ_bidual(C, D, return_matrix=False, **kwargs):
    """ Optimize the bidual of Omega given C and D.
    NOTE: this is the exact formulation from bi-conjugate.

    .. math::
        \\Omega_{\\circ} (C)
        = \\max_{S} [tr(S.T C) - \\frac{n^2}{4} ||S||_F^2 - Qcal_ub(n^2 S, D)/n^2].

    Args:
        C (np.ndarray): Distance matrix in source domain.
        D (np.ndarray): Distance matrix in target domain.
        return_matrix (bool, optional): Return optimal R if `True`.

    Returns:
        (tuple):
            float: optimal function value.
            np.ndarray, optional: optimal S if `return_matrix` is `True`.
    """
    assert C.shape[0] == D.shape[0]
    n = D.shape[0]
    VERBOSE = kwargs.get("verbose", False)

    def obj(S):
        S = S.reshape((n, -1))
        # NOTE: force R in the symmetric domain, dom f^* \in \Scal
        S = sym(S)
        fval_Qcal_ub, Q1, Q2 = Qcal_ub(n**2 * S, D, return_matrix=True)
        fval = np.trace(S.T @ C) - n**2 / 4 * norm(S)**2 - fval_Qcal_ub / n**2

        grad = C - n**2 * S / 2 - grad_Qcal_ub_C(D, Q1, Q2)
        return -fval, -grad.flatten()

    def callback(S):
        S = S.reshape((n, -1))
        fval, grad = obj(S)
        print(f"fval {-fval:.4f} ||g|| {norm(grad):.4f}")

    S_init = np.random.rand(n**2)
    ''' check grad '''
    if kwargs.get("check_grad"):
        grad_diff = check_grad(lambda x: obj(x)[0], lambda x: obj(x)[1], S_init)
        print(f"||g||_diff {grad_diff:.2e}")

    res = minimize(obj, S_init,
                   method="BFGS",
                   jac=True,
                   callback=callback if VERBOSE else None,
                   #    options={"disp": 1}
                   )
    fval_opt = -res['fun']
    S_opt = res['x'].reshape((n, -1))

    if return_matrix:
        return fval_opt, S_opt
    else:
        return fval_opt


def omega_circ_star(S, D, **kwargs):
    """ Given S, solve the omega_circ_star function

    ..math::
        \\Omega^\\circ (S) = n^2/4 \nbr{S}_F^2 + Qcal(n^2 S, D) / n^2

    Args:
        S (np.ndarray): Input S matrix with dim (n, n)
        D (np.ndarray): The cost matrix from target domain with dim (n, n).

    Returns:
        tuple:
            float: optimal function value.
            np.ndarray: grad with dim (n, n).
    """
    n = D.shape[0]
    fval_qcal_ub, Q1, Q2 = Qcal_ub(n**2 * S, D, return_matrix=True)
    fval = n**2 / 4 * norm(S)**2 + fval_qcal_ub / n**2

    # derivative of omega_circ_star
    grad = n**2 / 2 * S + grad_Qcal_ub_C(D, Q1, Q2)
    return fval, grad


def omega_circ_star_v2(S, D, eps=1e-3):
    """ Given S, solve the omega_circ_star function


    ..math::
        \\Omega^\\circ (S) = n^2/4 \nbr{S}_F^2 + Qcal(n^2 S, D) / n^2

    Args:
        S (np.ndarray): Input S matrix with dim (n, n)
        D (np.ndarray): The cost matrix from target domain with dim (n, n).

    Returns:
        tuple:
            float: optimal function value.
            np.ndarray: grad with dim (n, n).

    Notes:
        * Smoothing trick: add a strong concave term  `- \\epsilon ||\\sigma||^2`
        * Compact implementation for the case the same dimension.
    """
    n = D.shape[0]
    eet = np.ones((n, n))
    V = projection_matrix(n)

    ''' solve Qcal_ub '''
    C = n**2 * S
    sC = C.sum()
    sD = D.sum()

    C_hat = V.T @ C @ V
    D_hat = V.T @ D @ V
    E_hat = 2 / n * V.T @ C @ eet @ D @ V

    ''' quad term: max_Q tr(C_hat Q D_hat Q^T)'''
    evals_C, evecs_C = eig_decom(C_hat)
    evals_D, evecs_D = eig_decom(D_hat)
    # fval_q = np.multiply(evals_C, evals_D).sum()
    # Q1 = evecs_C @ evecs_D.T

    # smoothing trick
    sigma = np.sign(evals_C * evals_D) * np.minimum(1, 1 / eps * np.abs(evals_C * evals_D))
    fval_q = (evals_C * evals_D * sigma).sum() - eps / 2 * norm(sigma)**2
    Q1 = evecs_C @ np.diag(sigma) @ evecs_D.T

    ''' linear term: max_Q tr(E_hat^T Q) '''
    u, s, vh = svd(E_hat)
    # fval_l = s.sum()
    # Q2 = u @ vh

    # smoothing trick
    sigma = np.minimum(1, 1 / eps * s)
    fval_l = np.dot(s, sigma) - eps / 2 * norm(sigma)**2
    Q2 = u @ np.diag(sigma) @ vh

    ''' const term'''
    fval_c = sC * sD / n**2
    fval_qcal_ub = fval_q + fval_l + fval_c

    # grad in C
    grad_q = V @ Q1 @ D_hat @ Q1.T @ V.T
    grad_l = sym(eet @ D @ V @ Q2 @ V.T) / n
    grad_c = sD * eet / n**2
    grad_qcal_ub = grad_q + grad_l + grad_c

    ''' \\Omega_\\circ^* '''
    fval = n**2 / 4 * norm(S)**2 + fval_qcal_ub / n**2
    grad = n**2 / 2 * S + grad_qcal_ub

    return fval, grad
