import numpy as np
from typing import Union

def constant_correlation_mat(size, rho):
    mat = np.ones((size, size)) * rho
    diag = np.diag_indices(size)
    mat[diag] = 1.
    return mat


def get_constant_correlation_set(data: np.array, rho: Union[float, np.array]):
    """
        :param data:    matrix of observations (rows) of multiple features (columns)
        :param rho:     constant correlation coefficient  between all features
    """

    data = data[:, np.all(data, axis=0)]
    print(data)
    sigma = np.zeros((data.shape[1], data.shape[1]))
    std = data.std(axis=0)
    np.fill_diagonal(sigma, std)

    if isinstance(rho, float):
        corr = constant_correlation_mat(data.shape[1], rho)
    elif isinstance(rho, np.array):
        corr = rho

    cov = sigma @ corr @ sigma
    delta = np.linalg.cholesky(cov)
    return delta
