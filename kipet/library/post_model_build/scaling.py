"""
General scaling method for Kipet models
"""
# Standard library imports
import copy

# Third party imports
from pyomo.environ import Param

# Kipet library imports
from kipet.library.core_methods.PyomoSimulator import PyomoSimulator
from kipet.library.common.VisitorClasses import ScalingVisitor, ReplacementVisitor  


def scale_models(models_input, k_vals, name=None):
    """Takes a model or dict of models and iterates through them to update the
    odes and parameter values
    
    """
    models_dict = copy.deepcopy(models_input)
    
    if not isinstance(models_dict, dict):
        key = 'model-1' if name is None else name
        models_dict = {key: models_dict}
    
    d_init = {}
    
    for model in models_dict.values():
        d_init_model = _scale_model(model, k_vals)
        d_init.update(d_init_model)

    return d_init, models_dict

def _scale_model(model, k_vals):
    """Scales an individual model and updates the initial parameter dict
    
    """
    scaled_bounds = {}    

    if not hasattr(model, 'K'):
        add_scaling_parameters(model, k_vals)
        scale_parameters(model)
     
        for k, v in model.P.items():
            lb, ub = v.bounds
            
            lb = lb/model.K[k].value
            ub = ub/model.K[k].value
            
            if ub < 1:
                print('Bounds issue, pushing upper bound higher')
                ub = 1.1
            if lb > 1:
                print('Bounds issue, pushing lower bound lower')
                lb = 0.9
                
            scaled_bounds[k] = lb, ub
                
            model.P[k].setlb(lb)
            model.P[k].setub(ub)
            model.P[k].unfix()
            model.P[k].set_value(1)
            
    parameter_dict = {k: (1, scaled_bounds[k]) for k in k_vals.keys() if k in model.P}
            
    return parameter_dict

def add_scaling_parameters(model, k_vals=None):
    
    print("The model has not been scaled and will now be scaled using K parameters")
    
    if k_vals is None:
    
        model.K = Param(model.parameter_names,                                                                                                                                                            
                    initialize={k: v for k, v in model.P.items()},
                    mutable=True,
                    default=1)
    else:
        model.K = Param(model.parameter_names,                                                                                                                                                            
                    initialize={k: v for k, v in k_vals.items() if k in model.P},
                    mutable=True,
                    default=1)
    
    return None
        
def scale_parameters(model, k_vals=None):
    """If scaling, this multiplies the constants in model.K to each
    parameter in model.P.
    
    I am not sure if this is necessary and will look into its importance.
    """
    if not hasattr(model, 'K'):
        add_scaling_parameters(model, k_vals=k_vals)
        
    scale = {}
    for i in model.P:
        scale[id(model.P[i])] = model.K[i]

    for i in model.Z:
        scale[id(model.Z[i])] = 1
        
    for i in model.dZdt:
        scale[id(model.dZdt[i])] = 1
        
    for i in model.X:
        scale[id(model.X[i])] = 1

    for i in model.dXdt:
        scale[id(model.dXdt[i])] = 1

    for k, v in model.odes.items(): 
        scaled_expr = _scale_expression(v.body, scale)
        model.odes[k] = scaled_expr == 0

def remove_scaling(model, bounds=None):
    
    """You need to reset the bounds on the parameter too"""
    """EP problem: parameter bounds are not respected"""
    
    if not hasattr(model, 'K'):
        raise AttributeError('The model is not scaled')
        
    for param in model.P.keys():   
        change_value = model.K[param].value
        model.P[param].set_value(model.K[param].value)
        
        for k, v in model.odes.items():
            ep_updated_expr = _update_expression(v.body, model.K[param], 1)
            model.odes[k] = ep_updated_expr == 0
        
        del model.K[param]
        
    if bounds is not None:
        for param, bound in bounds.items():
            if param in model.P:
                model.P[param].setlb(bound[0])
                model.P[param].setub(bound[1])
        
    return None
        
def _scale_expression(expr, scale):
    
    visitor = ScalingVisitor(scale)
    return visitor.dfs_postorder_stack(expr)

def _update_expression(expr, replacement_param, change_value):
    """Takes the noparam_infon-estiambale parameter and replaces it with its intitial
    value
    
    Args:
        expr (pyomo constraint expr): the target ode constraint
        
        replacement_param (str): the non-estimable parameter to replace
        
        change_value (float): initial value for the above parameter
        
    Returns:
        new_expr (pyomo constraint expr): updated constraints with the
            desired parameter replaced with a float
    
    """
    visitor = ReplacementVisitor()
    visitor.change_replacement(change_value)
    visitor.change_suspect(id(replacement_param))
    new_expr = visitor.dfs_postorder_stack(expr)
    
    return new_expr

