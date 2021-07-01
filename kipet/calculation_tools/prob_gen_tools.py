"""
Problem Generation tools for Kipet
"""
import numpy as np
import pandas as pd


def gaussian_single_peak(wl, alpha, beta, gamma):
    """
    Helper function to generate absorption data based on 
    Lorentzian parameters
    
    :param float wl: Wavelength
    :param float beta: Beta
    :param float alpha: Alpha
    :param float gamma: gamma
    
    :return: The single peak
    :rtype: float

    """
    return alpha*np.exp(-(wl-beta)**2/gamma)


def absorbance(wl, alphas, betas, gammas):
    """
    Helper function to generate absorption data based on 
    Lorentzian parameters
    
    :param float wl: Wavelength
    :param array-like betas: Beta
    :param array-like alphas: Alpha
    :param array-like gammas: gamma
    
    :return: The sum of single peaks
    :rtype: float

    """
    return sum(gaussian_single_peak(wl,alphas[i],betas[i],gammas[i]) for i in range(len(alphas)))


def generate_absorbance_data(wl_span, parameters_dict):
    """
    Helper function to generate absorption data based on 
    Lorentzian parameters
    
    :param array-like wl_span: Array of wavelengths
    :param dict parameters_dict: The dictionary of alphas, betas, and gammas
   
    :return: data_frame
    :rtype: pandas.DataFrame

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


def generate_random_absorbance_data(wl_span, component_peaks, 
                                    component_widths=None, seed=None):
    """
    Helper function to generate absorption data based on 
    Lorentzian parameters
    
    :param array-like wl_span: Array of wavelengths
    :param dict component_peaks: Dictionary with number of component peaks
    :param dict component_widths: Optional widths for components (otherwise 1000)
    :param int seed: Random seed number
   
    :return: data_frame
    :rtype: pandas.DataFrame

    """
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
    
    :param pandas.DataFrame signal: The Z or S matrix to have noise added to it
    :param float size: sigma (or size of distribution)
    Returns:
    :return: Noised data
    :rtype: pandas.DataFrame
    
    """
    clean_sig = signal    
    noise = np.random.normal(0,size,clean_sig.shape)
    sig = clean_sig+noise    
    df= pd.DataFrame(data=sig)
    df[df < 0] = 0
    return df
