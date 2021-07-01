"""
Module for reading and converting hessian data
"""
# Standard library imports
import contextlib
import os
import re

# Third party imports
import numpy as np


def suppress_stdout(func):
    """Decorator to redirect output

    .. note::

        Find out where this is used and remove it

    :param func: A python function

    :return: Decorated function

    """
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                func(*a, **ka)
    return wrapper


def split_sipopt_string(output_string):
    """Split the data from sipopt to get the ipopt string and hessian string

    :param str output_string: The raw solver data string

    :return: The separated ipopt and hessian strings in a tuple
    :rtype: tuple

    """
    start_hess = output_string.find('DenseSymMatrix')
    ipopt_string = output_string[:start_hess]
    hess_string = output_string[start_hess:]
    return (ipopt_string, hess_string)


def split_k_aug_string(output_string):
    """Split the data from k_aug to get the ipopt string and hessian string

    :param str output_string: The raw solver data string

    :return: The separated ipopt and hessian strings in a tuple
    :rtype: tuple

    """
    start_hess = output_string.find('')
    ipopt_string = output_string[:start_hess]
    hess_string = output_string[start_hess:]
    return (ipopt_string, hess_string)


def read_reduce_hessian2(hessian_string, n_vars):
    """Conversion method for hessian data

    :param str hessian_string: The raw string version of the Hessian
    :param int n_vars: The number of variables in the Hessian (n x n)

    :return numpy.ndarray hessian: The converted Hessian

    """
    hessian_string = re.sub('RedHessian unscaled\[', '', hessian_string)
    hessian_string = re.sub('\]=', ',', hessian_string)

    hessian = np.zeros((n_vars, n_vars))
    for i, line in enumerate(hessian_string.split('\n')):
        if i > 0:
            hess_line = line.split(',')
            if len(hess_line) == 3:
                row = int(hess_line[0])
                col = int(hess_line[1])
                hessian[row, col] = float(hess_line[2])
                hessian[col, row] = float(hess_line[2])
    return hessian


def read_reduce_hessian(hessian_string, n_vars):
    """Conversion method for hessian data

    :param str hessian_string: The raw string version of the Hessian
    :param int n_vars: The number of variables in the Hessian (n x n)

    :return numpy.ndarray hessian: The converted Hessian

    """
    hessian = np.zeros((n_vars, n_vars))
    for i, line in enumerate(hessian_string.split('\n')):
        if i > 0:  # ignores header
            if line not in ['', ' ', '\t']:
                hess_line = line.split(']=')
                if len(hess_line) == 2:
                    value = float(hess_line[1])
                    column_line = hess_line[0].split(',')
                    col = int(column_line[1])
                    row_line = column_line[0].split('[')
                    row = int(row_line[1])
                    hessian[row, col] = float(value)
                    hessian[col, row] = float(value)
    return hessian


def read_reduce_hessian_k_aug(hessian_string, n_vars):
    """Conversion method for hessian data

    :param str hessian_string: The raw string version of the Hessian
    :param int n_vars: The number of variables in the Hessian (n x n)

    :return numpy.ndarray hessian: The converted Hessian

    """
    hessian = np.zeros((n_vars, n_vars))
    for i, line in enumerate(hessian_string.split('\n')):
        if i > 0:  # ignores header
            if line not in ['', ' ', '\t']:
                hess_line = line.split(']=')
                if len(hess_line) == 2:
                    value = float(hess_line[1])
                    column_line = hess_line[0].split(',')
                    col = int(column_line[1])
                    row_line = column_line[0].split('[')
                    row = int(row_line[1])
                    hessian[row, col] = float(value)
                    hessian[col, row] = float(value)
    return hessian


def save_eig_red_hess(matrix): 
    """Saves the eigenvalues of the Hessian as a txt file

    .. warning::

        This is to be removed. This may not even be used anymore.

    :param numpy.ndarray matrix: The covariance matrix used for the reduced hessian decomposition

    :return: None

    """
    redhessian = np.linalg.inv(matrix)
    np.savetxt('redhessian.out', redhessian, delimiter=',')
    eigH2, vH2 = np.linalg.eig(redhessian)
    np.savetxt('eigredhessian.out', eigH2)
    
    return None
