"""
    Solve the L1TF problem ADMM as done in Klinger
"""
import numpy as np
from matplotlib import pylab as plt


from numpy.linalg import inv


def soft_threshold(k, a):
    """
    Soft threshold function, proximal operator for l1 norm
    vectorized version
    :param k: number
    :param a: number
    :return: number
    """
    n = len(a)
    result = np.zeros(n)
    mask = a > k
    result[mask] = a[mask] - k
    mask = a < -k
    result[mask] = a[mask] + k
    return result


def memo(f):
    """ Memoization decorator for a function taking a single argument """
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__


@memo
def get_second_derivative_matrix(n):
    """
    :param n: The size of the input dimension
    :return: A matrix D such that D * x is the second derivative of x
    """
    m = n - 2
    D = np.zeros((m, n))
    v = np.array([1.0, -2.0, 1.0])
    for row_num in range(0,m):
        D[row_num, row_num:row_num+3] = v
    return D

@memo
def get_sec_der_and_m_inv(n_rho):
    n, rho = n_rho
    D = get_second_derivative_matrix(n)
    M = np.eye(n) + rho * D.T.dot(D)
    # This could be improved some by using Cholesky decomp
    # rather than general matrix inversion (maybe)
    M_inv = inv(M)
    return D, M_inv


def l1tf(y, iter_max=1000, rho=1.0, lam=3.0, prompt=False,
         tol=1e-8, verbose=False):
    """
    Find the bets fit L1TF solution
    :param y: the data vector
    :param iter_max: maximum number of iterations
    :param rho: the ADMM step parameter
    :param lam: the problem's l1 regularization parameter
    :param prompt: show plots and print stuff at each step
                (default False)
    :param tol: Stop if max change between steps is lower than this
                times the max value of y
    :param verbose: Print stuff (default False)
    :return: the best fit regularized trend vector
    """
    tol_max = tol*np.max(y)
    rho = float(rho)
    lam = float(lam)
    n = len(y)
    m = n - 2

    D, M_inv = get_sec_der_and_m_inv((n, rho))

    # initialize
    x = y.copy()
    z = np.zeros(m)
    u = np.zeros(m)
    ratio = lam/rho

    iter = max_delta = 0

    for iter in range(0, iter_max):
        x_last = x
        x = M_inv.dot(y + rho*D.T.dot(z-u))
        q = D.dot(x) + u
        z = soft_threshold(ratio, q)
        u += D.dot(x) - z
        if prompt:
            print('iter: %s' % iter)
            print(x-x_last)
            if iter == 0:
                plt.clf()
                plt.plot(y)

            plt.plot(x, alpha=0.3)
            plt.show()
            ok = raw_input('ok?')
            if ok == 'q':
                return
            if ok == 'n':
                prompt = False

        max_delta = abs(x - x_last).max()
        if max_delta < tol_max:
            break

    if verbose:
        print("Max change: %s" % max_delta)
        print("niter: %s" % iter)

    return x

def run_l1tf(y, plot=True):
    """
    :param iter_max: maximum number of iterations
    :param rho: the ADMM step parameter
    :param lam: the problem's l1 regularization parameter
    :param prompt: show plots and print stuff at each step
                (default False)
    :param tol: Stop if max change between steps is lower than this
                times the max value of y
    :param verbose: Print stuff (default False)
    :return:
    """

    iter_max=1000
    rho=1.0
    lam=0.5
    prompt=False
    tol=1e-26
    verbose=True

    #important that y is sampled equally and with meaningfull values (e.g. every 100m)
    x = l1tf(y, iter_max=iter_max, rho=rho, lam=lam, tol=tol,
             prompt=prompt, verbose=verbose)
    if plot is True:
        plt.figure()
        plt.plot(y, 'bo-', alpha=0.7)
        plt.plot(x, color='red', alpha=0.7, linewidth=2)
        plt.show()
    return x
