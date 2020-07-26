"""
KIPET 2020

This file contains the object functions used throughout Kipet modules in one
place.

"""
from pyomo.environ import (
    Objective,
    )

def get_objective(model, *args, **kwargs):
    
    objective_type = kwargs.get('objective_type', 'concentration')
    
    if objective_type == 'concentration':
        objective_expr = conc_objective(model, *args, **kwargs)

    return Objective(rule=objective_expr)


def conc_objective(model, *args, **kwargs):
    """
    
    Parameters
    ----------
    m : Pyomo ConcreteModel
        This is the current used in parameter fitting

    Returns
    -------
    obj : Objective function for Pyomo models
        This is the concentration based objective function

    """
    obj=0
    
    for index, values in model.C.items():
        obj += _concentration_term(model, index, **kwargs)
        
    return obj

def comp_objective(model, *args, **kwargs):
    """
    
    Parameters
    ----------
    m : Pyomo ConcreteModel
        This is the current used in parameter fitting

    Returns
    -------
    obj : Objective function for Pyomo models
        This is the concentration based objective function

    """
    obj=0
    
    for index, values in model.U.items():
        obj += _complementary_state_term(model, index)
        
    return obj

def spectra_objective(model, *args, **kwargs):
    """
    
    Parameters
    ----------
    m : Pyomo ConcreteModel
        This is the current used in parameter fitting

    Returns
    -------
    obj : Objective function for Pyomo models
        This is the concentration based objective function

    """
    obj=0
    # change this to items in the list (D or D_bar)
    # for t in model.meas_times:
    #     for l in model.meas_lambdas:
    
    for index, values in model.D.items():
        obj += _spectra_term(model, index)
        
    return obj

        
def _concentration_term(model, index, **kwargs):
    """
    
    Parameters
    ----------
    m : Pyomo ConcreteModel
        This is the current used in parameter fitting

    index : tuple
        This is the index of the model.C component

    Returns
    -------
    objective_concentration_term : Pyomo expression
        LS concentration term for objective

    """
    custom_sigma = kwargs.get('sigma', None)
    
    if custom_sigma is None:
        sigma = model.sigma[index[1]]
    else:
        sigma = custom_sigma[index[1]]
    
    objective_concentration_term = 0.5*(model.C[index] - model.Z[index]) ** 2 \
        / sigma**2
    
    return objective_concentration_term
    

def _complementary_state_term(model, index):
    """
    
    Parameters
    ----------
    m : Pyomo ConcreteModel
        This is the current used in parameter fitting

    index : tuple
        This is the index of the model.C component

    Returns
    -------
    objective_complementary_state_term : Pyomo expression
        LS complementary state term for objective

    """
    objective_complementary_state_term = 0.5*(model.U[index] - model.X[index]) ** 2 \
        / model.sigma[index[1]]**2
    
    return objective_complementary_state_term

def _spectra_term(model, index, use_sigma=True):
    """
    
    Parameters
    ----------
    m : Pyomo ConcreteModel
        This is the current used in parameter fitting

    index : tuple
        This is the index of the model.C component

    Returns
    -------
    objective_complementary_state_term : Pyomo expression
        LS complementary state term for objective

    """
    objective_complementary_state_term = 0.5*(model.D[index] - model.D_bar[index]) ** 2
    
    if use_sigma:
        objective_complementary_state_term /= model.sigma['device']**2

    return objective_complementary_state_term
    