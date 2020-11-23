"""
Parameter handling
"""

def check_initial_parameter_values(model_object):
    
    for k, v in model_object.P.items():
        
        bound_push = 1.05
        
        if model_object.P[k].value >= model_object.P[k].ub:
            model_object.P[k].set_value(model_object.P[k].ub/bound_push)
        
        if model_object.P[k].value <= model_object.P[k].lb:
            model_object.P[k].set_value(model_object.P[k].lb*bound_push)
            
    return None
        
def set_scaled_parameter_bounds(model_object, parameter_set=None, rho=10, scaled=True, original_bounds=None):
    """Set the parameter values (scaled) for a given set of parameters
    
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