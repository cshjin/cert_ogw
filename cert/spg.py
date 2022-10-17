""" Implementation of the spectral projected gradient (SPG) method
    for optimizing over a (closed) convex set X

Original matlab code by Mark Schmidt:
https://www.cs.ubc.ca/~schmidtm/Software/minConf.html

copyright (C) https://github.com/levinboimtomer/SPG
V0.2 Feb 15th 2014
Python code by: Tomer Levinboim (first.last at usc.edu)

NOTE:
* NOT IMPLEMENTED:
    Cubic line search is not implemented (Currently only halving)
"""
import sys

import numpy as np


class SPGOptions():
    def __init__(self, maxIter=500, verbose=2, suffDec=1e-4, progTol=1e-9, optTol=1e-5, curvilinear=False, memory=10,
                 useSpectral=True, bbType=1, interp=2, numdiff=0, testOpt=True) -> None:
        """ Solver's options

        Args:
            maxIter (int, optional): Maximum number of calls to funObj. Defaults to 500.
            verbose (int, optional): Level of verbosity. Defaults to 2.
                                        0: no output,
                                        1: final,
                                        2: iter (default),
                                        3: debug
            suffDec (float, optional): Sufficient decrease parameter in Armijo condition. Defaults to 1e-4.
            progTol (float, optional): Tolerance used to check for lack of progress. Defaults to 1e-9.
            optTol (float, optional): Tolerance used to check for optimality. Defaults to 1e-5.
            curvilinear (bool, optional): Backtrack along projection arc. Defaults to False.
            memory (int, optional): Number of steps to look back in non-monotone Armijo condition. Defaults to 10.
            useSpectral (bool, optional): Use spectral method. Defaults to True.
            bbType (int, optional): Type of Barzilai-Borwein step. Defaults to 1.
            interp (int, optional): Method for back tracking. Defaults to 2.
                                        0: none,
                                        2: cubic (for the most part.. see below).
            numdiff (int, optional): Compute derivatives numerically. Defaults to 0.
                                        0: use user-supplied derivatives (default),
                                        1: use finite differences.
            testOpt (bool, optional): Test the optimizer. Defaults to True.
        """
        self.maxIter = maxIter
        self.verbose = verbose
        self.suffDec = suffDec
        self.progTol = progTol
        self.optTol = optTol
        self.curvilinear = curvilinear
        self.memory = memory
        self.useSpectral = useSpectral
        self.bbType = bbType
        self.interp = interp
        self.numdiff = numdiff
        self.testOpt = testOpt


default_options = SPGOptions()


def SPG(funObj0, funProj, x, options=default_options):
    """ Spectral projected gradient method.

    Args:
        funObj0 (callable): Objective function (and gradient, if not use autodiff).
        funProj (callable): Projection function.
        x (np.array): Initial value.
        options (SPGOptions instance): Solver's options.

    Returns:
        np.narray: optimal x
        float: optimal f
    """
    x = funProj(x)
    i = 1  # iteration

    funEvalMultiplier = 1
    if options.numdiff == 1:
        def funObj(x):
            return auto_grad(x, funObj0, options)
        funEvalMultiplier = len(x) + 1
    else:
        funObj = funObj0

    f, g = funObj(x)
    projects = 1
    funEvals = 1

    if options.verbose >= 2:
        if options.testOpt:
            print('%10s %10s %10s %15s %15s %15s' %
                  ('Iteration', 'FunEvals', 'Projections', 'Step Length', 'Function Val', 'Opt Cond'))
        else:
            print('%10s %10s %10s %15s %15s' %
                  ('Iteration', 'FunEvals', 'Projections', 'Step Length', 'Function Val'))

    while funEvals <= options.maxIter:
        if i == 1 or not options.useSpectral:
            alpha = 1.0
        else:
            y = g - g_old
            s = x - x_old
            assertVector(y)
            assertVector(s)

            # type of BB step
            if options.bbType == 1:
                alpha = np.dot(s.T, s) / np.dot(s.T, y)
            else:
                alpha = np.dot(s.T, y) / np.dot(y.T, y)

            if alpha <= 1e-10 or alpha > 1e10:
                alpha = 1.0

        d = -alpha * g
        f_old = f
        x_old = x
        g_old = g

        if not options.curvilinear:
            d = funProj(x + d) - x
            projects += 1

        gtd = np.dot(g, d)

        if gtd > -options.progTol:
            log(options, 1, 'Directional Derivative below progTol')
            break

        if i == 1:
            t = min([1, 1.0 / np.sum(np.absolute(g))])
        else:
            t = 1.0

        if options.memory == 1:
            funRef = f
        else:
            if i == 1:
                old_fvals = np.tile(-np.inf, (options.memory, 1))

            if i <= options.memory:
                old_fvals[i - 1] = f
            else:
                old_fvals = np.vstack([old_fvals[1:], f])

            funRef = np.max(old_fvals)

        if options.curvilinear:
            x_new = funProj(x + t * d)
            projects += 1
        else:
            x_new = x + t * d

        f_new, g_new = funObj(x_new)
        funEvals += 1
        lineSearchIters = 1
        while f_new > funRef + options.suffDec * np.dot(g.T, (x_new - x)) or not isLegal(f_new):
            temp = t
            # Halfing step size
            if options.interp == 0 or ~isLegal(f_new):
                log(options, 3, 'Halving Step Size')
                t /= 2.0
            elif options.interp == 2 and isLegal(g_new):
                log(options, 3, 'Cubic Backtracking')
                gtd_new = np.dot(g_new, d)
                t = polyinterp2(np.array([[0, f, gtd],
                                          [t, f_new, gtd_new]]))
            elif lineSearchIters < 2 or ~isLegal(f_prev):
                log(options, 3, 'Quadratic Backtracking')
                t = polyinterp2(np.array([[0, f, gtd],
                                          [t, f_new, 1j]])).real
            else:
                # t = polyinterp([0 f gtd; t f_new sqrt(-1);t_prev f_prev sqrt(-1)]);
                # not implemented.
                # fallback on halving.
                t /= 2.0

            if t < temp * 1e-3:
                log(options, 3, 'Interpolated value too small, Adjusting: ' + str(t))
                t = temp * 1e-3
            elif t > temp * 0.6:
                log(options, 3, 'Interpolated value too large, Adjusting: ' + str(t))
                t = temp * 0.6
            # Check whether step has become too small
            if np.max(np.absolute(t * d)) < options.progTol or t == 0:
                log(options, 3, 'Line Search failed')
                t = 0.0
                f_new = f
                g_new = g
                break

            # Evaluate New Point
            f_prev = f_new
            t_prev = temp

            if options.curvilinear:
                x_new = funProj(x + t * d)
                projects += 1
            else:
                x_new = x + t * d

            f_new, g_new = funObj(x_new)
            funEvals += 1
            lineSearchIters += 1

        # Take Step
        x = x_new
        f = f_new
        g = g_new

        if options.testOpt:
            optCond = np.max(np.absolute(funProj(x - g) - x))
            projects += 1

        if hasattr(options, "early_stopping"):
            if f < 0:
                return x, f

        # Output Log
        if options.verbose >= 2:
            if options.testOpt:
                print('{:10d} {:10d} {:10d} {:15.5e} {:15.5e} {:15.5e}'.format(i, funEvals * funEvalMultiplier,
                                                                               projects, t, f, optCond))
            else:
                print('{:10d} {:10d} {:10d} {:15.5e} {:15.5e}'.format(i, funEvals * funEvalMultiplier, projects, t, f))

        # Check optimality
        if options.testOpt:
            if optCond < options.optTol:
                log(options, 1, 'First-Order Optimality Conditions Below optTol')
                break

        if np.max(np.absolute(t * d)) < options.progTol:
            log(options, 1, 'Step size below progTol')
            break

        if np.absolute(f - f_old) < options.progTol:
            log(options, 1, 'Function value changing by less than progTol')
            break

        if funEvals * funEvalMultiplier > options.maxIter:
            log(options, 1, 'Function Evaluations exceeds maxIter')
            break

        i += 1

    return x, f


def polyinterp2(points):
    """ Polynomial interpolation between new and old x and fval.

    Args:
        points (np.array): Inputs

    Returns:
        float: New point after interpolation.

    NOTE:
     * Code for most common case:
      - cubic interpolation of 2 points w/ function and derivative values for both
      - no xminBound/xmaxBound
     * Solution in this case (where x2 is the farthest point):
       d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
       d2 = sqrt(d1^2 - g1*g2);
       minPos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
       t_new = min(max(minPos,x1),x2);
    """
    minPos = np.argmin(points[:, 0])
    # minVal = points[minPos, 0]
    notMinPos = -minPos + 1
    d1 = points[minPos, 2] + points[notMinPos, 2] - 3 * \
        (points[minPos, 1] - points[notMinPos, 1]) / (points[minPos, 0] - points[notMinPos, 0])
    d2 = np.sqrt(d1**2 - points[minPos, 2] * points[notMinPos, 2])
    if np.isreal(d2):
        t = points[notMinPos, 0] - (points[notMinPos, 0] - points[minPos, 0]) * \
            ((points[notMinPos, 2] + d2 - d1) / (points[notMinPos, 2] - points[minPos, 2] + 2 * d2))
        minPos = min([max([t, points[minPos, 0]]), points[notMinPos, 0]])
    else:
        minPos = np.mean(points[:, 0])
    return minPos


def auto_grad(x, funObj, options):
    """ Auto gradient if funObj only provides objective function.

    Args:
        x (np.array): Variable.
        funObj (callable): Objective function.
        options (SPGOptions): Solver's options.

    Returns:
        float: function value
        np.array: gradient value
    """
    p = len(x)
    f = funObj(x)
    if isinstance(f, type(())):
        f = f[0]

    mu = 2 * np.sqrt(1e-12) * (1 + np.linalg.norm(x)) / np.linalg.norm(p)
    diff = np.zeros((p,))
    for j in range(p):
        e_j = np.zeros((p,))
        e_j[j] = 1
        # this is somewhat wrong, since we also need to project,
        # but practically (and locally) it doesn't seem to matter.
        v = funObj(x + mu * e_j)
        if isinstance(v, type(())):
            diff[j] = v[0]
        else:
            diff[j] = v

    g = (diff - f) / mu

    return f, g


def log(options, level, msg):
    """ Output log message

    Args:
        options (SPGOptions): Solver's options.
        level (int): Verbose level.
        msg (str): Log messages.
    """
    if options.verbose >= level:
        print(msg, file=sys.stderr)


def assertVector(v):
    """ Assert the vector is in 1D dimension.

    Args:
        v (np.array): Input array.
    """
    assert len(v.shape) == 1


def isLegal(v):
    """ Check the vector is in real space.

    Args:
        v (np.array): Numpy array.

    Returns:
        bool: `True` if vector is in real space.
    """
    no_complex = v.imag.any().sum() == 0
    no_nan = np.isnan(v).sum() == 0
    no_inf = np.isinf(v).sum() == 0
    return no_complex and no_nan and no_inf
