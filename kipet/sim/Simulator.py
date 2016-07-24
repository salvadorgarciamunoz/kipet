from pyomo.environ import *
from pyomo.dae import *
from ResultsObject import *
import numpy as np
import math
import scipy

# need to move this two functions to utils
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
        return idx-1
    else:
        return idx

def interpolate_linearly(x,x_tuple,y_tuple):
    m = (y_tuple[1]-y_tuple[0])/(x_tuple[1]-x_tuple[0])
    return y_tuple[0]+m*(x-x_tuple[0])

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
        self._meas_times = sorted([t for t in self.model.meas_times])
        self._meas_lambdas = sorted([l for l in self.model.meas_lambdas]) 
        self._n_meas_times = len(self._meas_times)
        self._n_meas_lambdas = len(self._meas_lambdas)
        self._n_components = len(self._mixture_components)
        self._n_complementary_states = len(self._complementary_states)

        if not self._mixture_components:
            raise RuntimeError('The model does not have any mixture components.\
            For simulation add mixture components')
        
    def apply_discretization(self,transformation,**kwargs):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def initialize_from_trajectory(self,trajectory_dictionary):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def run_sim(self,solver,**kwds):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def compute_D_given_SC(self,results,sigma_d=0):
        # this requires results to have S and C computed already
        d_results = []
        for i,t in enumerate(self._meas_times):
            for j,l in enumerate(self._meas_lambdas):
                suma = 0.0
                for w,k in enumerate(self._mixture_components):
                    C =  results.C[k][t]  
                    S = results.S[k][l]
                    suma+= C*S
                if sigma_d:
                    suma+= np.random.normal(0.0,sigma_d)
                d_results.append(suma)
                    
        d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
        results.D = pd.DataFrame(data=d_array,
                                 columns=self._meas_lambdas,
                                 index=self._meas_times)
