"""
Parameter handling
"""
from kipet.top_level.variable_names import VariableNames

__var = VariableNames()
parameter_set_name = __var.model_parameter

def calculate_parameter_averages(model_dict):
    
    p_dict = {}
    c_dict = {}
    
    for key, model in model_dict.items():
        for param, obj in getattr(model, __var.model_parameter).items():
            if param not in p_dict:
                p_dict[param] = obj.value
                c_dict[param] = 1
            else:
                p_dict[param] += obj.value
                c_dict[param] += 1
                
    avg_param = {param: p_dict[param]/c_dict[param] for param in p_dict.keys()}
    
    return avg_param
    
def initialize_parameters(model_dict):
    
    avg_param = calculate_parameter_averages(model_dict)
    
    for key, model in model_dict.items():
        for param, obj in getattr(model, __var.model_parameter).items():
            obj.value = avg_param[param] 
            
    return None

def check_initial_parameter_values(model_object):
    """Checks the initial parameter values and alters them if they violate the
    bounds.
    
    Args:
        model_object (pyomo ConcreteModel): A pyomo model instance of the current
            problem 
            
    Returns:
        None
        
    """
    
    for k, v in getattr(model_object, parameter_set_name).items():
        
        bound_push = 1.05
        
        if v.value >= v.ub:
            v.set_value(param.ub/bound_push)
        
        if v.value <= v.lb:
            v.set_value(v.lb*bound_push)
            
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
    param_set = param = getattr(model_object, parameter_set_name)
    
    if parameter_set is None:
        parameter_set = [p for p in param_set]
    
    for k, v in param_set.items():
        
        if k in parameter_set:
            
            if not scaled:
                ub = rho*v.value
                lb = 1/rho*v.value
            else:
                ub = rho
                lb = 1/rho
            
            if original_bounds is not None:
                if ub > original_bounds[k][1]:
                    ub = original_bounds[k][1]
                if lb < original_bounds[k][0]:
                    lb = original_bounds[k][0]
                
            v.setlb(lb)
            v.setub(ub)
            v.unfix()
            
            if scaled:
                v.set_value(1)
    
        else:
            v.fix()

    return None