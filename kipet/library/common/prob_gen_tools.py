"""
Problem Generation tools for Kipet
"""
import numpy as np
import pandas as pd
  
def gaussian_single_peak(wl,alpha,beta,gamma):
    """
    helper function to generate absorption data based on 
    lorentzian parameters
    """
    return alpha*np.exp(-(wl-beta)**2/gamma)

def absorbance(wl,alphas,betas,gammas):
    """
    helper function to generate absorption data based on 
    lorentzian parameters
    """
    return sum(gaussian_single_peak(wl,alphas[i],betas[i],gammas[i]) for i in range(len(alphas)))

def generate_absorbance_data(wl_span,parameters_dict):
    """
    helper function to generate absorption data based on 
    lorentzian parameters
    """
    components = parameters_dict.keys()
    n_components = len(components)
    n_lambdas = len(wl_span)
    array = np.zeros((n_lambdas,n_components))
    for i,l in enumerate(wl_span):
        j = 0
        for k, p in parameters_dict.items():
            alphas = p['alphas']
            betas  = p['betas']
            gammas = p['gammas']
            array[i,j] = absorbance(l,alphas,betas,gammas)
            j+=1

    data_frame = pd.DataFrame(data=array,
                              columns = components,
                              index=wl_span)
    return data_frame


def generate_random_absorbance_data(wl_span,component_peaks,component_widths=None,seed=None):

    np.random.seed(seed)
    parameters_dict = dict()
    min_l = min(wl_span)
    max_l = max(wl_span)
    #mean=1000.0
    #sigma=1.5*mean
    for k,n_peaks in component_peaks.items():
        params = dict()
        if component_widths:
            width = component_widths[k]
        else:
            width = 1000.0
        params['alphas'] = np.random.uniform(0.1,1.0,n_peaks)
        params['betas'] = np.random.uniform(min_l,max_l,n_peaks)
        params['gammas'] = np.random.uniform(1.0,width,n_peaks)
        parameters_dict[k] = params

    return generate_absorbance_data(wl_span,parameters_dict)

def add_noise_to_signal(signal, size):
    """
    Adds a random normally distributed noise to a clean signal. Used mostly in Kipet
    To noise absorbances or concentration profiles obtained from simulations. All
    values that are negative after the noise is added are set to zero
    Args:
        signal (data): the Z or S matrix to have noise added to it
        size (scalar): sigma (or size of distribution)
    Returns:
        pandas dataframe
    """
    clean_sig = signal    
    noise = np.random.normal(0,size,clean_sig.shape)
    sig = clean_sig+noise    
    df= pd.DataFrame(data=sig)
    df[df<0]=0
    return df
