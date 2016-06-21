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
        self._meas_times = sorted([t for t in self.model.meas_times])
        self._meas_lambdas = sorted([l for l in self.model.meas_lambdas]) 
        self._n_meas_times = len(self._meas_times)
        self._n_meas_lambdas = len(self._meas_lambdas)
        self._n_components = len(self._mixture_components)
        self._discretized = False
        
    def apply_discretization(self,transformation,**kwargs):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def initialize_from_trajectory(self,trajectory_dictionary):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def run_sim(self,solver,tee=False,solver_opts={}):
        raise NotImplementedError("Simulator abstract method. Call child class")
        
    def _solve_CS_from_D(self,C_dataFrame):
        c_noise_array = np.zeros((self._n_meas_times,self._n_components))
        for i,t in enumerate(self._meas_times):
            for j,k in enumerate(self._mixture_components):
                c_noise_array[i,j] = C_dataFrame[k][t]

        D_data = self.model.D
        if self._n_meas_lambdas:
            # build Dij vector
            D_vector = np.zeros(self._n_meas_times*self._n_meas_lambdas)
            
            row  = []
            col  = []
            data = []    
            for i,t in enumerate(self._meas_times):
                for j,l in enumerate(self._meas_lambdas):
                    for k,c in enumerate(self._mixture_components):
                        row.append(i*self._n_meas_lambdas+j)
                        col.append(j*self._n_components+k)
                        data.append(c_noise_array[i,k])
                    D_vector[i*self._n_meas_lambdas+j] = D_data[t,l]    
                
                        
            Bd = scipy.sparse.coo_matrix((data, (row, col)),
                                         shape=(self._n_meas_times*self._n_meas_lambdas,
                                                self._n_components*self._n_meas_lambdas))
            if self._n_meas_times == self._n_components:
                s_array = scipy.sparse.linalg.spsolve(Bd, D_vector)
            elif self._n_meas_times>self._n_components:
                result_ls = scipy.sparse.linalg.lsqr(Bd, D_vector)
                s_array = result_ls[0]
            else:
                raise RuntimeError('Need n_t_meas >= self._n_components')
            
            s_shaped = s_array.reshape((self._n_meas_lambdas,self._n_components))
        else:
            s_shaped = np.empty((self._n_meas_lambdas,self._n_components))

        return (c_noise_array,s_shaped)
