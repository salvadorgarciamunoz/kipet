from pyomo.environ import *
from pyomo.dae import *
from ResultsObject import *
import numpy as np
import math

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

    def apply_discretization(self,transformation,**kwargs):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def initialize_from_trajectory(self,trajectory_dictionary):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def run_sim(self,solver,tee=False):
        raise NotImplementedError("Simulator abstract method. Call child class")
        
