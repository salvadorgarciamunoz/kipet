import bisect
import math

import numpy as np
import scipy

from pyomo.dae import *
from pyomo.environ import *
from kipet.library.ResultsObject import *

# These functions are fundamentaly incorrect
# need to move these two functions to utils
# def find_nearest(array,value):
#     idx = np.searchsorted(array, value, side="left")
#     if idx == len(array) or idx==len(array)-1 or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
#         return idx-1
#     else:
#         return idx

#%%

def interpolate_trajectory(t, tr):
    
    slopes = []
    times = [ti for ti in t]
    tr_val = np.zeros((len(times)))
    
    for i, tp in enumerate(tr.index):
        
        if i == 0:
            continue
        
        x1 = tr.index[i - 1]
        x2 = tr.index[i]

        y1 = tr.loc[x1]
        y2 = tr.loc[x2]
        
        slope = (y2 - y1)/(x2 - x1)
        slopes.append((tp, slope))
        
    def slope_calc(time_value, breakpoints=list(tr.index)[1:], slopes=[s[1] for s in slopes] + [0]):
        i = bisect.bisect_left(breakpoints, time_value)
        return slopes[i]

    val = tr.loc[0]

    counter = 0
    for i, time in enumerate(times):
        del_x = times[i] - times[i - 1]
        val += del_x * slope_calc(time)
        tr_val[i] = val

    return tr_val 

#%%

# def interpolate_linearly(x,x_tuple,y_tuple):
#     # print(f'x: {x}')
#     # print(f'x_tuple: {x_tuple}')
#     # print(f'y_tuple: {y_tuple}')
#     m = (y_tuple[1]-y_tuple[0])/(x_tuple[1]-x_tuple[0])
#     return y_tuple[0]+m*(x-x_tuple[0])

# def interpolate_from_trajectory(t,trajectory):

#     times_traj = np.array(trajectory.index)
#     last_time_idx = len(times_traj)-1
#     idx_near = find_nearest(times_traj,t)
    
#     # print(f'times_traj: {times_traj}')
#     # print(f'last_time_idx: {last_time_idx}')
#     # print(f'idx_near: {idx_near}')
    
#     if idx_near==last_time_idx:
#         # print('if true')
#         t_found = times_traj[idx_near]
#         # print(f't_found: {t_found}')
#         return trajectory[t_found]
#     else:
#         # print('if false')
#         idx_near1 = idx_near+1
#         t_found = times_traj[idx_near]
#         t_found1 = times_traj[idx_near1]
#         val = trajectory[t_found]
#         val1 = trajectory[t_found1]
#         x_tuple = (t_found,t_found1)
#         y_tuple = (val,val1)
#         #print(f't_found: {t_found}')
#         #print(f't_found1: {t_found1}')
#         #print(f'val: {val}')
#         #print(f'val1: {val1}')
        
#         return interpolate_linearly(t,x_tuple,y_tuple)

class Simulator(object):
    """Base simulator class.

    Note:
        This class is not intended to be used directly by users

    Attributes:
        model (model): Casadi or Pyomo model.

    """

    def __init__(self,model):
        self.model = model

        # make model atributes of the model global.
        # Need to be remove, and just get things from the model directly

        self._mixture_components = [name for name in self.model.mixture_components]
        self._complementary_states = [name for name in self.model.complementary_states]
        self._algebraics = [name for name in self.model.algebraics]
        self._meas_times = sorted([t for t in self.model.meas_times])
        self._allmeas_times = sorted([t for t in self.model.allmeas_times])#added for new data structure CS
        self._meas_lambdas = sorted([l for l in self.model.meas_lambdas])
        self._n_meas_times = len(self._meas_times)
        self._n_allmeas_times = len(self._allmeas_times)#added for new data structure CS
        self._n_meas_lambdas = len(self._meas_lambdas)
        self._n_components = len(self._mixture_components)
        self._n_algebraics = len(self._algebraics)
        self._n_complementary_states = len(self._complementary_states)
        self._non_absorbing = None
        self._huplc_absorbing = None #added for additional huplc data CS
        self._known_absorbance = None
        self._known_absorbance_data = None
        
        if hasattr(self.model, 'non_absorbing'):
            self._non_absorbing = [name for name in self.model.non_absorbing]
        if hasattr(self.model, 'known_absorbance'):
            self._known_absorbance = [name for name in self.model.known_absorbance]
        #added for removing nonabs ones from first term in objective CS:
        if hasattr(self.model, 'abs_components'):
            self._abs_components = [name for name in
                                    self.model.abs_components]  # added for removing nonabs ones from first term in objective CS
            self._nabs_components = len(self._abs_components)
        ####
        #added for additional huplc data CS:
        if hasattr(self.model, 'huplc_absorbing'):
            self._huplcmeas_times = sorted([t for t in self.model.huplcmeas_times])  # added for additional huplc data CS
            self._n_huplcmeas_times = len(self._huplcmeas_times)  # added for additional huplc data CS
            self._huplc_absorbing = [name for name in self.model.huplc_absorbing]
        if hasattr(self.model, 'huplcabs_components'):
            self._huplcabs_components = [name for name in
                                        self.model.huplcabs_components]
            self._nhuplcabs_components = len(self._huplcabs_components)
        ####
        if not self._mixture_components:
            raise RuntimeError('The model does not have any mixture components.\
            For simulation add mixture components')

    def apply_discretization(self, transformation, **kwargs):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def initialize_from_trajectory(self,trajectory_dictionary):
        raise NotImplementedError("Simulator abstract method. Call child class")
    
    def run_sim(self,solver,**kwds):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def compute_D_given_SC(self,results,sigma_d=0):
        # this requires results to have S and C computed already
        d_results = []

        if hasattr(self, '_abs_components'):  # added for removing non_abs ones from first term in obj CS
            for i, t in enumerate(self._allmeas_times):
                if t in self._meas_times:
                    for j, l in enumerate(self._meas_lambdas):
                        suma = 0.0
                        for w, k in enumerate(self._abs_components):
                            Cs = results.Cs[k][t]  # just the absorbing ones
                            Ss = results.S[k][l]
                            suma += Cs * Ss
                        if sigma_d:
                            suma += np.random.normal(0.0, sigma_d)
                        d_results.append(suma)

        else:
            for i, t in enumerate(self._allmeas_times):
                if t in self._meas_times:
                    for j, l in enumerate(self._meas_lambdas):
                        suma = 0.0
                        for w, k in enumerate(self._mixture_components):
                            C = results.C[k][t]
                            S = results.S[k][l]
                            suma += C * S
                        if sigma_d:
                            suma += np.random.normal(0.0, sigma_d)
                        d_results.append(suma)

        d_array = np.array(d_results).reshape((self._n_meas_times, self._n_meas_lambdas))
        results.D = pd.DataFrame(data=d_array,
                                 columns=self._meas_lambdas,
                                 index=self._meas_times)