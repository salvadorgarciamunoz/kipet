"""
Helper functions for making KIPET models
"""
from kipet.top_level.variable_names import VariableNames

__var = VariableNames()

def step_fun(model, time_var, num='0', coeff=1, time=1e-3, fixed=True, switch=False, M=1, eta=1e-2, constant=0.5):
    """This formulates the step functions for KIPET using a sigmoidal function
    
    Args:
        model: The ConcreteModel using the step function
        
        time_var (float): Time point for step
        
        num (str): The name of the step function
        
        coeff (float): Coefficient for the step size
        
        time (float): The time for the step change (init if not fixed)
        
        fixed (bool): Choose if the time is known and fixed or variable
        
        on (bool): True if turning on, False if turning off, optional string args too
        
        M (float): Sigmoidal tuning parameter
        
        eta (float): Tuning parameter
        
        constant (float): Constant added to bring to certain value
        
    Returns:
        step_point (Expression): Returns the pyomo expression for the point
    
    """
    if switch in ['on', 'turn_on']:
        switch = True
    elif switch in ['off', 'turn_off']:
        switch = False
    
    factor = 1 if not switch else -1
    
    if fixed:
        if time is None:
            raise ValueError('Time cannot be None')
        time_var_step = factor*time
    else: 
        if time is not None:
            model.time_step_change[num].set_value(time)
        time_var_step = factor*model.time_step_change[num]
        
    pyomo_var = getattr(model, __var.dosing_variable)[time_var, __var.dosing_component]
    step_point = coeff*(0.5*((time_var_step - factor*pyomo_var) / ((time_var_step - factor*pyomo_var)**2 + eta**2/M) ** 0.5) + constant)
    
    return step_point