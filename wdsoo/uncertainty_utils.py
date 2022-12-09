import os
import json
import numpy as np
import pandas as pd
from typing import Union


class Ufile:
    def __init__(self, path):
        self.path = path

    def add_element(self):
        pass

    def add_ucategory(self):
        pass

    def edit_element(self):
        pass

    def edit_category(self):
        pass

    def remove_element(self):
        pass


class UCategory:
    def __init__(self, name, elements: dict, elements_correlation):
        self.name = name
        self.elements = elements
        self.elements_correlation = elements_correlation
        self.corr_mat = self.get_corr_matrix()

        self.elements_dim = self.validate_dimensions()
        self.sigma = self.construct_matrix()  # Equivalent to cov matrix
        self.delta = np.linalg.cholesky(self.sigma)  # Affine mapping matrix

    def get_corr_matrix(self):
        if self.elements_correlation is not None:
            if isinstance(self.elements_correlation, float):
                return constant_correlation_mat(len(self.elements), self.elements_correlation)
            elif isinstance(self.elements_correlation, np.ndarray):
                if not validate_symmetric(self.elements_correlation):
                    raise Exception("Correlation matrix is not symmetric")
                else:
                    return self.elements_correlation

    def validate_dimensions(self):
        l = [len(e.std) for e_name, e in self.elements.items()]
        if [l[0]]*len(l) == l:
            return l[0]
        else:
            raise Exception("Uncertainty std dimensions not valid")

    def construct_matrix(self):
        t = self.elements_dim
        n = len(self.elements)
        mat = np.zeros((t*n, t*n))

        for i, (ei_name, ei) in enumerate(self.elements.items()):
            mat[i*t: i*t+t, i*t: i*t+t] = ei.cov
            for j, (ej_name, ej) in enumerate(self.elements.items()):
                if j == i:
                    continue
                else:
                    r = self.corr_mat[i, j]
                    mat[i*t: i*t+t, j*t: j*t+t] = np.multiply(ei.std, ej.std) * r

        if not is_pd(mat):
            print(f'Warning: {self.name} COV matrix not positive defined')
            mat = nearest_positive_defined(mat)
        return mat


class UElement:
    def __init__(self, ucat, name, idx, element_type, std, corr):
        self.ucat = ucat
        self.name = name
        self.idx = idx
        self.element_type = element_type
        self.std = std
        self.corr = corr

        if self.std is not None:
            self.cov = build_cov_from_std(self.std, self.corr)


def init_uncertainty(udata_path: str):
    ucategories = {}
    with open(os.path.join(udata_path)) as f:
        udata = json.load(f)

        for ucat, ucat_data in udata.items():
            cat_elements = {}
            for idx, (e_name, e_data) in enumerate(ucat_data['elements'].items()):
                if e_data['std'] is not None:
                    ue = UElement(ucat, e_name, idx, **e_data)
                    cat_elements[e_name] = ue

            if len(cat_elements) > 0:
                uc = UCategory(ucat, cat_elements, np.array(ucat_data['elements_correlation']))
                ucategories[ucat] = uc

    return ucategories


def constant_correlation_mat(size, rho):
    mat = np.ones((size, size)) * rho
    diag = np.diag_indices(size)
    mat[diag] = 1.
    return mat


def build_cov_from_std(std, rho: Union[float, np.array] = 0):
    n = len(std)
    sigma = np.zeros((n, n))
    np.fill_diagonal(sigma, std)

    if isinstance(rho, (int, float)):
        corr = constant_correlation_mat(n, rho)
    elif isinstance(rho, np.array):
        corr = rho

    cov = sigma @ corr @ sigma
    return cov


def observations_cov(data: np.array, rho: Union[float, np.array]):
    """
        :param data:    matrix of observations (rows) of multiple features (columns)
        :param rho:     constant correlation coefficient  between all features
        :return:        cov matrix
    """
    std = data.std(axis=0)
    cov = build_cov_from_std(std, rho)
    return cov


def affine_map_from_cov(cov):
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
    else:
        corr = np.eye(n)

    cov = sigma @ corr @ sigma
    delta = np.linalg.cholesky(cov)
    return delta


def validate_symmetric(mat):
    if (mat == mat.T).all():
        return True
    else:
        return False


def nearest_positive_defined(A):
    """
    source: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def is_pd(B):
    """
    Returns true when input is positive-definite, via Cholesky
    source: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    """
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

