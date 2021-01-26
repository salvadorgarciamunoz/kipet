"""
Helper functions for making KIPET models
"""
from kipet.library.top_level.variable_names import VariableNames

__var = VariableNames()

def step_fun(model, time_var, time=1e-3, M=1, eta=1e-2, constant=0.5):
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
    pyomo_var = getattr(model, __var.dosing_variable)[time_var, __var.dosing_component]
    
    step_point = 0.5*((time - pyomo_var) / ((time - pyomo_var)**2 + eta**2/M) ** 0.5) + constant
    
    return step_point

#%%

# def step(t, delta=1e-3, M=1e-8, eps=5e-6, constant=0.5):
    
#     v = 0.5*((delta - t) / ((delta - t)**2 + eps**2/M) ** 0.5) + constant
    
#     return v


# import numpy as np
# import matplotlib.pyplot as plt

# t = np.linspace(0, 600)

# ts = 210

# plt.figure()

# results = []
# results2 = []
# results3 =[]

# for i in t:
    
#     fun1 = step(i, delta=-51, M=1, eps=1e-2, constant=0.5)
#     fun2 = step(i, delta=210, M=1, eps=1e-2, constant=0.5)
    
#     factor = 1#7.27609e-05 
#     fun = factor*(fun1) # + fun2)
    
#     results.append(fun)
#     results2.append(fun2)
#     results3.append(fun + fun2)
    
# plt.plot(t, results)
# plt.plot(t, results2)
# plt.plot(t, results3)

# #%%

# t = np.linspace(-0.0, 600)

# ts = 210


# results = []
# for i in t:
    
#     fun1 = step(i, delta=-1, M=1, eps=1e-2, constant=1.5)
#     # fun2 = step(i, delta=210, M=1, eps=1e-2)
    
#     # fun = 7.27609e-05*(fun1 + fun2)
    
#     results.append(fun1)
    
# plt.plot(t, results)

