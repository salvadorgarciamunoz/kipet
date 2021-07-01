"""
Various calculations surrounding Beer Lambert's Law
"""
# Third party imports
import numpy as np

# KIPET library imports
from kipet.model_tools.pyomo_model_tools import convert


def D_from_SC(model, results, sigma_d=0):

    """Given the S and C matrices, the D matrix can be calculated

    :param ConcreteModel model: A Pyomo model
    :param ResultsObject results: The results from a solved model
    :param float sigma_d: The device variance

    :return: D-matrix
    :rtype: np.ndarray

    """
    
    C = convert(model.C)
    C = C.loc[:, [c for c in model.abs_components]]
    
    S = convert(model.S)
    D = C @ S.T

    if sigma_d > 0:
        d_noise = np.random.normal(np.zeros(D.shape), sigma_d)
        D += d_noise

    results.D = D
    return D.values


def S_from_DC(model, C_dataFrame, tee=False, with_bounds=False, max_iter=200):
    """Solves a basic least squares problems for determining S from D and C
    data.

    :param pandas.DataFrame C_dataFrame: data frame with concentration values
    :param bool tee: Option to output least_squares results
    :param bool with_bounds: Option to set lower bound to zero
    :param int max_iter: The maximum number of iterations used in least_squares
    
    :return s_shaped: DataFrame with estimated S_values
    :rtype: pandas.DataFrame

    """
    D_data = convert(model.D)
    D_vector = D_data.values.flatten()
    
    if not with_bounds:
        # Use simple matrix multiplication to get S
        
        # Imports used only here
        from kipet.calculation_tools.interpolation import interpolate_trajectory2
        
        C_orig = C_dataFrame
        C = interpolate_trajectory2(list(D_data.index), C_orig)
    
        #model.mixture_components.ordered_data()
    
        # non_abs_species = r1.components.get_match('absorbing', False)
        # C = C.drop(columns=non_abs_species)
        indx_list = list(D_data.index)
        for i, ind in enumerate(indx_list):
            indx_list[i] = round(ind, 6)
        
        D_data.index = indx_list
        
        assert C.shape[0] == D_data.values.shape[0]
        
        M1 = np.linalg.inv(C.T @ C)
        M2 = C.T @ D_data.values
        S = (M1 @ M2).T
        S.columns = C.columns
        S = S.set_index(D_data.columns)
    
    else:
        # Use least_squares to get S using zero as the lower bound
        
        # Imports only used here
        from scipy.optimize import least_squares
        from scipy.sparse import coo_matrix
        import pandas as pd
    
        num_lambdas = D_data.shape[1]
        lambdas = list(D_data.columns)
        num_times = D_data.shape[0]
        times = list(D_data.index)
        components = C_dataFrame.columns
        num_components = len(components)

        row  = []
        col  = []
        data = []    
        for i,t in enumerate(times):
            for j,l in enumerate(lambdas):
                for k,c in enumerate(components):
                    row.append(i*num_lambdas+j)
                    col.append(j*num_components+k)
                    data.append(C_dataFrame[c][t])
                    
        Bd = coo_matrix((data, (row, col)),
                        shape=(num_times*num_lambdas,
                                num_components*num_lambdas)
                        )

        x0 = np.zeros(num_lambdas*num_components)+1e-2
        M = Bd.tocsr()
        
        def F(x, M, rhs):
            return rhs - M.dot(x)

        def JF(x, M, rhs):
            return -M

        verbose = 2 if tee else 0
        res_lsq = least_squares(
            F,
            x0,
            JF,
            bounds=(0.0, np.inf),
            max_nfev=max_iter,
            verbose=verbose,
            args=(M, D_vector)
        )
        
        s_shaped = res_lsq.x.reshape((num_lambdas, num_components))
        S = pd.DataFrame(s_shaped, columns=components, index=lambdas)

    return S.values
