import numpy as np
import pandas as pd


def true_c_profiles(times):
    
    n_times = times.size
    k = 0.01
    C = np.zeros((2,n_times))
    C[0,:] = np.exp(-k*times)
    C[1,:] = 1-C[0,:] 
    return C

def true_s_profiles(lambdas):
    
    n_lambdas = lambdas.size
    S = np.zeros((2,n_lambdas))
    S[0,:] = 2*np.exp(-(lambdas-50)**2/3e4)+1.3*np.exp(-(lambdas-200)**2/1e3)
    S[1,:] = np.exp(-(lambdas-150)**2/1e2)+0.2*np.exp(-(lambdas-170)**2/3e4) + 2*np.exp(-(lambdas-200)**2/1e2) + np.exp(-(lambdas-250)**2/1e2)
    return S

    delta_t = 1
    measured_times = np.arange(0,200+delta_t,delta_t,dtype='float64')
    true_C = true_c_profiles(measured_times)
    C_frame = pd.DataFrame(data=true_C.T, columns=mixture_components, index=measured_times)

    delta_l = 1
    measured_lambdas = np.arange(0,500+delta_l,delta_l,dtype='float64')
    true_S = true_s_profiles(measured_lambdas)
    S_frame = pd.DataFrame(data=true_S.T, columns=mixture_components, index=measured_lambdas)
   
    D_array = np.dot(np.transpose(true_C),true_S)

    D_frame = pd.DataFrame(data=D_array, columns=measured_lambdas, index=measured_times)
    
