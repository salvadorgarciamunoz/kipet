"""
Parameter selection and update methods
"""
# Third party imports
import numpy as np
import pandas as pd


def parameter_ratios(model_object, reduced_hessian, Se, epsilon=1e-16):
    """This is Eq. 26 from Chen and Biegler 2020 where the ratio of the 
    standard deviation for each parameter is calculated.
    
    :param ConcreteModel model_object: The current Pyomo model
    :param numpy.ndarray reduced_hessian: The current reduced hessian
    :param list Se: The list of free parameters
    :param float epsilon: Tolerance parameter for small values

    :return: The ratio of predicted standard deviations to the parameter values and the eigenvalues
    :rtype: tuple

    """
    eigenvalues, eigenvectors = np.linalg.eigh(reduced_hessian)
    red_hess_inv = np.dot(np.dot(eigenvectors, np.diag(1.0 / abs(eigenvalues))), eigenvectors.T)
    d = red_hess_inv.diagonal()
    d_sqrt = np.asarray(np.sqrt(d)).ravel()
    rp = [d_sqrt[i] / max(epsilon, model_object.P[k].value) for i, k in enumerate(Se)]

    return rp, eigenvalues


def rank_parameters(model_object, reduced_hessian, param_list, epsilon=1e-16, eta=1e-1):
    """Performs the parameter ranking based using the Gauss-Jordan
    elimination procedure.
    
    :param ConcreteModel model_object: The current Pyomo model
    :param numpy.ndarray reduced_hessian: The current reduced hessian
    :param list param_list: The list of model parameters
    :param float epsilon: Tolerance parameter for small values
    :param float eta: Tolerance for variances
    
    :return list Se_update: The updated list of parameters in Se.
    :return list Sf_update: The updated list of parameters if Sf.
    :rtype: tuple

    """
    eigenvector_tolerance = 1e-15
    parameter_tolerance = 1e-12
    squared_term_1 = 0
    squared_term_2 = 0
    Sf_update = []
    Se_update = []
    M = {}

    param_set = set(param_list)
    param_elim = set()
    eigenvalues, U = np.linalg.eigh(reduced_hessian)

    df_eigs = pd.DataFrame(np.diag(eigenvalues),
                           index=param_list,
                           columns=[i for i, x in enumerate(param_list)])
    df_U_gj = pd.DataFrame(U,
                           index=param_list,
                           columns=[i for i, x in enumerate(param_list)])

    # Gauss-Jordan elimination
    for i, p in enumerate(param_list):
        piv_col = i
        piv_row = df_U_gj.loc[:, piv_col].abs().idxmax()
        piv = (piv_row, piv_col)
        rows = list(param_set.difference(param_elim))
        M[i] = piv_row
        df_U_gj = _gauss_jordan_step(df_U_gj, piv, rows)
        param_elim.add(piv_row)
        df_eigs.drop(index=[param_list[piv_col]], inplace=True)
        df_eigs.drop(columns=[piv_col], inplace=True)

    # Parameter ranking
    eigenvalues, eigenvectors = np.linalg.eigh(reduced_hessian)
    ranked_parameters = {k: M[abs(len(M) - 1 - k)] for k in M.keys()}

    for k, v in ranked_parameters.items():
        name = v.split('[')[-1].split(']')[0]
        squared_term_1 += abs(1 / max(eigenvalues[-(k + 1)], parameter_tolerance))
        squared_term_2 += (eta ** 2 * max(abs(model_object.P[name].value), epsilon) ** 2)

        if squared_term_1 >= squared_term_2:
            Sf_update.append(name)
        else:
            Se_update.append(name)

    if len(Se_update) == 0:
        Se_update.append(ranked_parameters[0])
        Sf_update.remove(ranked_parameters[0])

    return Se_update, Sf_update


def _gauss_jordan_step(df_U_update, pivot, rows):
    """Performs the Gauss-Jordan Elimination step in W. Chen's method
    
    :param pandas.DataFrame df_U_update: A pandas DataFrame instance of the U matrix from the
            eigenvalue decomposition of the reduced hessian (can be obtained
            through the HessianObject).
    :param str pivot: The element where to perform the elimination.
    :param set rows: A set containing the rows that have already been eliminated.
        
    :return: The df_U_update DataFrame is returned after one row is eliminated.
    :rtype: pandas.DataFrame

    """
    if isinstance(pivot, tuple):
        pivot = dict(r=pivot[0],
                     c=pivot[1]
                     )

    uij = df_U_update.loc[pivot['r'], pivot['c']]

    for col in df_U_update.columns:
        if col == pivot['c']:
            continue

        factor = df_U_update.loc[pivot['r'], col] / uij

        for row in rows:
            if row not in rows:
                continue
            else:
                df_U_update.loc[row, col] -= factor * df_U_update.loc[row, pivot['c']]

    df_U_update[abs(df_U_update) < 1e-15] = 0
    df_U_update.loc[pivot['r'], pivot['c']] = 1

    return df_U_update
