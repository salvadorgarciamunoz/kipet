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
    
    for t, v in model.C.items():
        obj += _concentration_term(model, t)
        
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
    
    for t, v in model.U.items():
        obj += _complementary_state_term(model, t)
        
    return obj

        
def _concentration_term(model, index):
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
    objective_concentration_term = 0.5*(model.C[index] - model.Z[index]) ** 2 \
        / model.sigma[index[1]]**2
    
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
    