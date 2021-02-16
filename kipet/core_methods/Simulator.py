import bisect
import math

import numpy as np
import scipy

from pyomo.dae import *
from pyomo.environ import *
from kipet.core_methods.ResultsObject import *

def interpolate_trajectory(t, tr):
    
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
    
    return tr_val 

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