import os
import json
import numpy as np
import pandas as pd
from typing import Union

from . import graphs


class Ufile:
    def __init__(self, path):
        self.path = path
        self.data = self.read()

    def read(self):
        with open(self.path, 'r') as file:
            return json.load(file)

    def write(self, output):
        with open(output, 'w') as file:
            json.dump(self.data, file, indent=4)

    def remove_category(self, cat):
        del self.data[cat]

    def edit_element(self, category: str, element: str, feature, value):
        self.data[category]['elements'][element][feature] = value

    def edit_category(self, category: str, feature, value):
        self.data[category][feature] = value


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
        if self.elements_correlation is None:
            return np.zeros((len(self.elements), len(self.elements)))

    def validate_dimensions(self):
        elements_dim = [len(e.std) for e_name, e in self.elements.items()]
        if [elements_dim[0]] * len(elements_dim) == elements_dim:
            return elements_dim[0]
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
                    np.fill_diagonal(mat[i*t: i*t+t, j*t: j*t+t], np.multiply(ei.std, ej.std) * r)

        w, v = np.linalg.eigh(mat)
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
            self.time_corr = self.get_time_corr()
            self.cov = build_cov_from_std(self.std, self.time_corr)

    def get_time_corr(self):
        r = np.zeros((len(self.std), len(self.std)))
        for i in range(len(self.std)):
            for j in range(len(self.std) - i):

                if self.corr == 0:
                    rr = 0
                else:
                    # rr = (1 - 0.5 * j * self.corr)  #  Linear decline correlation
                    # rr = (1 - np.sin(j*2*np.pi/(len(self.std)*2)) * self.corr)  # sinusoidal correlation
                    rr = np.exp(- j * self.corr)  # Exponential decline

                r[i, i + j] = rr
                r[j + i, i] = rr
        return r


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
                if ucat_data['elements_correlation'] is not None:
                    cat_corr = np.array(ucat_data['elements_correlation'])
                else:
                    cat_corr = None
                uc = UCategory(ucat, cat_elements, cat_corr)
                ucategories[ucat] = uc

    return ucategories


def constant_correlation_mat(size, rho):
    mat = np.ones((size, size)) * rho
    diag = np.diag_indices(size)
    mat[diag] = 1.
    return mat


def build_cov_from_std(std, rho: Union[int, float, np.array] = 0):
    n = len(std)
    sigma = np.zeros((n, n))
    np.fill_diagonal(sigma, std)

    if isinstance(rho, (int, float)):
        corr = constant_correlation_mat(n, rho)
    elif isinstance(rho, np.ndarray):
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


def nearest_positive_defined(mat):
    """
    source: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    """
    b = (mat + mat.T) / 2
    _, s, v = np.linalg.svd(b)

    h = np.dot(v.T, np.dot(np.diag(s), v))
    mat2 = (b + h) / 2
    mat3 = (mat2 + mat2.T) / 2
    if is_pd(mat3):
        return mat3

    spacing = np.spacing(np.linalg.norm(mat))
    k = 1
    while not is_pd(mat3):
        mineig = np.min(np.real(np.linalg.eigvals(mat3)))
        mat3 += np.eye(mat.shape[0]) * (-mineig * k**2 + spacing)
        k += 1

    return mat3


def is_pd(mat):
    """
    Returns true when input is positive-definite, via Cholesky
    source: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    """
    try:
        _ = np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False

