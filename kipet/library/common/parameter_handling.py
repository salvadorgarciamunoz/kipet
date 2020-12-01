"""
Parameter handling
"""

def check_initial_parameter_values(model_object):
    """Checks the initial parameter values and alters them if they violate the
    bounds.
    
    Args:
        model_object (pyomo ConcreteModel): A pyomo model instance of the current
            problem 
            
    Returns:
        None
        
    """
    
    for k, v in model_object.P.items():
        
        bound_push = 1.05
        
        if model_object.P[k].value >= model_object.P[k].ub:
            model_object.P[k].set_value(model_object.P[k].ub/bound_push)
        
        if model_object.P[k].value <= model_object.P[k].lb:
            model_object.P[k].set_value(model_object.P[k].lb*bound_push)
            
    return None
        
def set_scaled_parameter_bounds(model_object, parameter_set=None, rho=10, scaled=True, original_bounds=None):
    """Set the parameter values (scaled) for a given set of parameters
    
    Args:
        model_object (pyomo ConcreteModel): A pyomo model instance of the current
            problem 
            
        parameter_set (list): list of parameters to be considered, if None all
            parameters in the model are considered
            
        rho (float): ratio for setting the upper and lower bounds
        
        scaled (bool): True if the parameters are scaled
        
        original_bounds (bool): True if the original parameter bounds are to be
            used (instead of rho)
            
    Returns:
        None
    
    """
    if parameter_set is None:
        parameter_set = [p for p in model_object.P]
    
    for k, v in model_object.P.items():
        
        if k in parameter_set:
            
            if not scaled:
                ub = rho*model_object.P[k].value
                lb = 1/rho*model_object.P[k].value
            else:
                ub = rho
                lb = 1/rho
            
            if original_bounds is not None:
                if ub > original_bounds[k][1]:
                    ub = original_bounds[k][1]
                if lb < original_bounds[k][0]:
                    lb = original_bounds[k][0]
                
            model_object.P[k].setlb(lb)
            model_object.P[k].setub(ub)
            model_object.P[k].unfix()
            
            if scaled:
                model_object.P[k].set_value(1)
    
        else:
            model_object.P[k].fix()

    return None