import numpy as np
from typing import Union


def is_uncertain(x):
    """
    check if a network element is uncertaint. else deterministic
    :param x:   Network element (pump station, well, tank)
    :return:    bool if uncertain
    """
    if hasattr(x, 'uncertainty') and x.uncertainty != 0:
        return True
    else:
        return False


def constant_correlation_mat(size, rho):
    mat = np.ones((size, size)) * rho
    diag = np.diag_indices(size)
    mat[diag] = 1.
    return mat


def observations_cov(data: np.array, rho: Union[float, np.array]):
    """
        :param data:    matrix of observations (rows) of multiple features (columns)
        :param rho:     constant correlation coefficient  between all features
        :return:        cov matrix
    """

    sigma = np.zeros((data.shape[1], data.shape[1]))
    std = data.std(axis=0)
    np.fill_diagonal(sigma, std)

    if isinstance(rho, (int, float)):
        corr = constant_correlation_mat(data.shape[1], rho)
    elif isinstance(rho, np.array):
        corr = rho

    cov = sigma @ corr @ sigma
    return cov


def uset_from_cvo(cov):
    delta = np.linalg.cholesky(cov)
    return delta


def uset_from_std(std: np.array, rho: Union[float, np.array] = 0):
    """
    :param std:     array of standard deviation for each element
    :param rho:     constant correlation or correlation matrix between the elements
    :return:        affine map of the correlated elements
    """
    if not np.any(std):
        return

    n = len(std)  # number of correlated elements
    sigma = np.zeros((n, n))
    np.fill_diagonal(sigma, std)

    if isinstance(rho, (int, float)):
        corr = constant_correlation_mat(n, rho)
    elif isinstance(rho, np.array):
        corr = rho

    cov = sigma @ corr @ sigma
    delta = np.linalg.cholesky(cov)
    return delta