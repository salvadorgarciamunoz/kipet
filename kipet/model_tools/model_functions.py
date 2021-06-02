"""
Helper functions for making KIPET models
"""
from kipet.general_settings.variable_names import VariableNames

__var = VariableNames()


def step_fun(model, time_var, num='0', coeff=1, time=1e-3, fixed=True, switch=False, M=1, eta=1e-2, constant=0.5, bounds=None):
    """This formulates the step functions for KIPET using a sigmoidal function

    :param ConcreteModel model: The ConcreteModel using the step function
    :param float time_var: Time point for step
    :param str num: The name of the step function
    :param float coeff: Coefficient for the step size
    :param float time: The time for the step change (init if not fixed)
    :param bool fixed: Choose if the time is known and fixed or variable
    :param bool/str switch: True if turning on, False if turning off, optional string args too
    :param float M: Sigmoidal tuning parameter
    :param float eta: Tuning parameter
    :param float constant: Constant added to bring to certain value
    :param tuple bounds: Bounds for the time if not fixed

    :return: Returns the pyomo expression for the step function
    :rtype: expression

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
