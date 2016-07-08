from pyomo.environ import *
from pyomo.dae import *
from ResultsObject import *
import numpy as np
import math
import scipy

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
    def __init__(self,model):
        self.model = model
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
