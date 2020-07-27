import numpy as np
import pandas as pd

def compute_D_given_SC(var_est_object, results,sigma_d=0):
    # this requires results to have S and C computed already
    d_results = []
    
    if hasattr(var_est_object, '_abs_components'):  # added for removing non_abs ones from first term in obj CS
        for i, t in enumerate(var_est_object._allmeas_times):
            if t in var_est_object._meas_times:
                for j, l in enumerate(var_est_object._meas_lambdas):
                    suma = 0.0
                    for w, k in enumerate(var_est_object._abs_components):
                        Cs = results.Cs[k][t]  # just the absorbing ones
                        Ss = results.S[k][l]
                        suma += Cs * Ss
                    if sigma_d:
                        suma += np.random.normal(0.0, sigma_d)
                    d_results.append(suma)
    
    else:
        for i, t in enumerate(var_est_object._allmeas_times):
            if t in var_est_object._meas_times:
                for j, l in enumerate(var_est_object._meas_lambdas):
                    suma = 0.0
                    for w, k in enumerate(var_est_object._mixture_components):
                        C = results.C[k][t]
                        S = results.S[k][l]
                        suma += C * S
                    if sigma_d:
                        suma += np.random.normal(0.0, sigma_d)
                    d_results.append(suma)
    
    d_array = np.array(d_results).reshape((var_est_object._n_meas_times, var_est_object._n_meas_lambdas))
    results.D = pd.DataFrame(data=d_array,
                             columns=var_est_object._meas_lambdas,
                             index=var_est_object._meas_times)
