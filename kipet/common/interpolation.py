"""
This module contains functions for ensuring that variables have congruent
time indicies.

For many users who may have delved into older versions of KIPET, these used
to be contained in the PyomoSimulator and Simulator classes.

@author: kevin
"""

import bisect

import numpy as np
import pandas as pd


def interpolate_trajectory(t, data):
    
    data = pd.DataFrame(data)
    df_interpolated = pd.DataFrame(columns=data.columns)
    
    for col in data.columns:
        
        tr = data[col]
        tr_val = np.zeros((len(t)))
        times = [float(ti) for ti in t]
        indx = tr.index.astype(float)
    
        for i, t_indx in enumerate(times):
    
            if i == 0:
                tr_val[i] = tr.iloc[0]
                continue        
    
            indx_left = bisect.bisect_left(indx[1:], times[i])
            if indx_left == len(tr) - 1:
                tr_val[i] = tr.iloc[indx_left]
                continue
            
            dx = indx[indx_left + 1] - indx[indx_left]
            dy = tr.iloc[indx_left + 1] - tr.iloc[indx_left]
            m = dy/dx
            val = tr.iloc[indx_left] +  (times[i] - indx[indx_left]) * m
            tr_val[i] = val
    
        df_interpolated[col] = tr_val
    
    return df_interpolated

# def initialize_from_trajectory(variable_name, trajectories):
#     """Initializes discretized points with values from trajectories.
#     Args:
#         variable_name (str): Name of the variable in pyomo model
#         trajectories (DataFrame or Series): Indexed in in the same way the pyomo
#         variable is indexed. If the variable is by two sets then the first set is
#         the indices of the data frame, the second set is the columns
#     Returns:
#         None
#     """
#     _print(f'Initialization of Var: {variable_name}')
#     if not self.model.alltime.get_discretization_info():
#         raise RuntimeError('apply discretization first before initializing')
        
#     var = getattr(self.model, variable_name)
#     inner_set, component_set = self.build_sets_new(variable_name, trajectories)
   
#     if inner_set is None and component_set is None:
#         return None

#     for component in component_set:
#         if component in trajectories.columns:
#             single_trajectory = trajectories[component]
#             values = interpolate_trajectory(inner_set, single_trajectory)
#             for i, t in enumerate(inner_set):
#                 if not np.isnan(values[i]):
#                     var[t, component].value = values[i]

#     return None

# C = rm.results_dict['simulator'].Z
# D = rm.spectra.data

# Cn = interpolate_trajectory(list(D.index), C)




# Perhaps use this for S inits?
# def scale_variables_from_trajectory(self, variable_name, trajectories):
#     """Scales discretized variables with maximum value of the trajectory.
#     Note:
#         This method only works with ipopt
#     Args:
#         variable_name (str): Name of the variable in pyomo model
#         trajectories (DataFrame or Series): Indexed in in the same way the pyomo
#         variable is indexed. If the variable is by two sets then the first set is
#         the indices of the data frame, the second set is the columns
#     Returns:
#         None
#     """
#     tol = 1e-5
    
#     var = getattr(self.model, variable_name)
#     inner_set, component_set = self.build_sets_new(variable_name, trajectories)
   
#     if inner_set is None and component_set is None:
#         return None
    
#     for component in component_set:
        
#         if component not in component_set:
#             raise RuntimeError(f'Component {component} is not used in the model')
            
#         nominal_vals = abs(trajectories[component].max())
#         if nominal_vals >= tol:
#             scale = 1.0 / nominal_vals
#             for t in inner_set:
#                 self.model.scaling_factor.set_value(var[t, component], scale)

#     self._ipopt_scaled = True
#     return None