"""
Helper functions for making KIPET models
"""
from kipet.library.top_level.variable_names import VariableNames

__var = VariableNames()

def step_fun(model, time_var, time=1e-3, M=1, eta=1e-2, constant=0.5, on=False):
    """This formulates the step functions for KIPET using a sigmoidal function
    
    Args:
        pyomo_var: (Var): The algebraic variable used for the step function
        
        delta (float): Time point for step
        
        M (float): Sigmoidal tuning parameter
        
        eta (float): Tuning parameter
        
        constant (float): Constant added to bring to certain value
        
    Returns:
        step_point (Expression): Returns the pyomo expression for the point
    
    """
    factor = 1 if not on else -1
    time *= factor
    pyomo_var = getattr(model, __var.dosing_variable)[time_var, __var.dosing_component]
    
    step_point = 0.5*((time - factor*pyomo_var) / ((time - factor*pyomo_var)**2 + eta**2/M) ** 0.5) + constant
    
    return step_point

def step_fun_var_time(model, time_var, M=1, eta=1e-2, constant=0.5, on=False):
    """This formulates the step functions for KIPET using a sigmoidal function
    
    Args:
        pyomo_var: (Var): The algebraic variable used for the step function
        
        delta (float): Time point for step
        
        M (float): Sigmoidal tuning parameter
        
        eta (float): Tuning parameter
        
        constant (float): Constant added to bring to certain value
        
    Returns:
        step_point (Expression): Returns the pyomo expression for the point
    
    """
    factor = 1 if not on else -1
    #time_var = getattr(model, __var.step_time_var)[__var.step_time_index]
    time_var_step = factor*model.time_step_change[0] 
    pyomo_var = getattr(model, __var.dosing_variable)[time_var, __var.dosing_component]
    
    step_point = 0.5*((time_var_step - factor*pyomo_var) / ((time_var_step - factor*pyomo_var)**2 + eta**2/M) ** 0.5) + constant
    
    return step_point