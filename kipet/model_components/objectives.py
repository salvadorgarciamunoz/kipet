"""
This file contains the object functions used throughout Kipet modules in one place. This should reduce the redundancy
in using objective terms

.. note::

    These methods are not complete in all cases and may not be used in future versions of KIPET.

"""
from pyomo.environ import Objective


def get_objective(model, *args, **kwargs):
    """Main method to gather the objective terms

    :param ConcreteModel model: The model for which the objective will be built
    :param tuple args: The arguments to be passed
    :param dict kwargs: The arguments to be passed

    :return: An objective component for the model
    :rtype: Objective

    """
    objective_type = kwargs.get('objective_type', 'concentration')
    
    if objective_type == 'concentration':
        objective_expr = conc_objective(model, *args, **kwargs)

    return Objective(rule=objective_expr)


def conc_objective(model, *args, **kwargs):
    """Method to build concentration terms in the objective function

    :param Pyomo ConcreteModel model: This is the current model used in parameter fitting
    :param tuple args: Arguments to be passed
    :param dict kwargs: Arguments to be passed

    :return: obj
    :rtype: expression

    """
    obj = 0
    source = kwargs.get('source', 'concentration')
    if source == 'concentration':
        if model.mixture_components & model.measured_data:
            for index, values in model.Cm.items():
                obj += _concentration_term(model, index, var='Cm', **kwargs)
    elif source == 'spectra':
        if model.mixture_components:
            for index, values in model.C.items():
                obj += _concentration_term(model, index, var='C', **kwargs)
      
    return obj


def comp_objective(model, *args, **kwargs):
    """Method to build individual complementary state terms in the objective function

    :param Pyomo ConcreteModel model: This is the current model used in parameter fitting
    :param tuple args: Arguments to be passed
    :param dict kwargs: Arguments to be passed

    :return: obj
    :rtype: expression

    """
    obj=0
    
    if model.complementary_states & model.measured_data:
        for index, values in model.U.items():
            obj += _complementary_state_term(model, index)
        
    return obj


def spectra_objective(model, *args, **kwargs):
    """Method to build individual spectral terms in the objective function

    :param Pyomo ConcreteModel model: This is the current model used in parameter fitting
    :param tuple args: Arguments to be passed
    :param dict kwargs: Arguments to be passed

    :return: obj
    :rtype: expression

    """
    obj=0

    for index, values in model.D.items():
        obj += _spectra_term(model, index)
        
    return obj


def absorption_objective(model, *args, **kwargs):
    """Method to build individual absorption terms in the objective function

    :param Pyomo ConcreteModel model: This is the current model used in parameter fitting
    :param tuple args: Arguments to be passed
    :param dict kwargs: Arguments to be passed

    :return: obj
    :rtype: expression

    """
    sigma_device = kwargs.get('device_variance', 1)
    g_option = kwargs.get('g_option', None)
    with_d_vars = kwargs.get('with_d_vars', True)
    shared_spectra = kwargs.get('shared_spectra', True)
    list_components = kwargs.get('species_list', None)

    obj=0

    for index, values in model.D.items():
        obj += _spectral_term_MEE(model,
                                  index,
                                  sigma_device,
                                  g_option,
                                  shared_spectra,
                                  with_d_vars,
                                  list_components)
    return obj

# def calc_D_bar(model, D_bar_use, list_components):
    
#     if D_bar_use is False:
#         D_bar = model.D_bar
#     else:
#         D_bar = {}
#         if hasattr(model, '_abs_components'):
#             d_bar_list = model._abs_components
#             c_var = 'Cs'
#         else:
#             d_bar_list = list_components
#             c_var = 'C'    
            
#         if hasattr(model, 'huplc_absorbing') and hasattr(model, 'solid_spec_arg1'):
#             d_bar_list = [k for k in d_bar_list if k not in model.solid_spec_arg1]
                 
#         for t in model.meas_times:
#             for l in model.meas_lambdas:
#                 D_bar[t, l] = sum(getattr(model, c_var)[t, k] * model.S[l, k] for k in d_bar_list)

#     return D_bar
        

def _concentration_term(model, index, var='C', **kwargs):
    """Method to build individual concentration terms in the objective function

    :param Pyomo ConcreteModel model: This is the current model used in parameter fitting
    :param tuple index: Index of the concentration point
    :param str var: The type of concentration to fit (C/Cm)
    :param dict kwargs: This is basically the variance

    :return: objective_concentration_term
    :rtype: expression

    """
    custom_sigma = kwargs.get('variance', None)
    
    if custom_sigma is None:
        variance = model.sigma[index[1]]
    else:
        variance = custom_sigma[index[1]]
        
    if variance is None:
        variance = 1
    
    objective_concentration_term = (getattr(model, var)[index] - model.Z[index]) ** 2 / variance
    
    return objective_concentration_term


def _complementary_state_term(model, index, **kwargs):
    """Method to build individual complementary state terms in the objective function

    :param Pyomo ConcreteModel model: This is the current model used in parameter fitting
    :param tuple index: Index of the concentration point
    :param str var: The type of concentration to fit (C/Cm)
    :param dict kwargs: This is basically the variance

    :return: objective_concentration_term
    :rtype: expression

    """
    custom_sigma = kwargs.get('variance', None)
    
    if custom_sigma is None:
        variance = model.sigma[index[1]]
    else:
        variance = custom_sigma[index[1]]
    
    objective_complementary_state_term = (model.U[index] - model.X[index]) ** 2 / variance
    
    return objective_complementary_state_term


def _spectra_term(model, index, use_sigma=True):
    """Method to build individual spectral terms in the objective function

    :param Pyomo ConcreteModel model: This is the current model used in parameter fitting
    :param tuple index: Index of the concentration point
    :param bool use_sigma: Option to use the variance in the terms

    :return: objective_spectral_term
    :rtype: expression

    """
    objective_spectral_term = 0.5*(model.D[index] - model.D_bar[index]) ** 2
    
    if use_sigma:
        objective_spectral_term /= model.sigma['device']**2

    return objective_spectral_term


def _absorption_term(model, index, sigma_device=1, D_bar=None, g_options=None):
    """Method to build individual concentration terms in the objective function

    :param Pyomo ConcreteModel model: This is the current model used in parameter fitting
    :param tuple index: Index of the concentration point
    :param float sigma_device: Device variance
    :param GeneralVar D_bar: Optional D variable use
    :param dict g_options: Unwanted contribution data

    :return: objective_absorption_term
    :rtype: expression

    """
    if g_options['unwanted_G'] or g_options['time_variant_G']:
        objective_absorption_term = (model.D[index] - D_bar[index] - model.qr[index[0]]*model.g[index[1]]) ** 2 / sigma_device
    elif g_options['time_invariant_G_no_decompose']:
        objective_absorption_term = (model.D[index] - D_bar[index] - model.g[index[1]]) ** 2 / sigma_device
    else:
        objective_absorption_term = (model.D[index] - D_bar[index]) ** 2 / sigma_device

    return objective_absorption_term


def _spectral_term_MEE(model, index, sigma_device, g_option, shared_spectra, with_d_vars, list_components):
    """Method to build individual spectral terms in the objective function for MEE

    :param Pyomo ConcreteModel model: This is the current model used in parameter fitting
    :param tuple index: Index of the concentration point
    :param float sigma_device: Device variance
    :param dict g_option: Unwanted contribution data
    :param bool shared_spectra: Option to share spectra across experiments
    :param bool with_d_vars: Option to use D variables (always True)...
    :param list list_components: A list of model components

    :return: objective_spectral_term
    :rtype: expression

    """
    t = index[0]
    l = index[1]
    if with_d_vars:
        base = model.D[t, l] - model.D_bar[t, l]
    else:
        D_bar = sum(model.C[t, k] * model.S[l, k] for k in list_components)
        base = model.D[t, l] - D_bar
        
    G_term = 0
    if g_option == 'time_variant_G':
        G_term -= model.qr[t]*model.g[l]
    elif g_option == 'time_invariant_G_decompose' and shared_spectra or g_option == 'time_invariant_G_no_decompose':
        G_term -= model.g[l]
   
    objective_spectral_term = (base + G_term)**2/sigma_device
    return objective_spectral_term
