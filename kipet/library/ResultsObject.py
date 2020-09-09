import datetime
import pandas as pd
import numpy as np
from pyomo.core import *
from pyomo.environ import *

from kipet.library.common.read_write_tools import df_from_pyomo_data

class ResultsObject(object):
    
    def __init__(self):
        """
        A class to store simulation and optimization results.
        """
        # Data series
        self.generated_datetime = datetime.datetime
        self.results_name = None
        self.solver_statistics = {}
        self.Z = None
        self.X = None
        self.Y = None
        self.C = None
        self.S = None
        self.sigma_sq = None
        self.device_variance = None
        self.P = None
        self.dZdt = None
        self.dXdt = None
        
        self.objective = None
        self.parameter_covariance = None

    def __str__(self):
        string = "\nRESULTS\n"
        if self.Z is not None:
            string += "Z:\n {}\n\n".format(self.Z)
        if self.C is not None:
            string += "C:\n {}\n\n".format(self.C)
        if self.S is not None:
            string += "S:\n {}\n\n".format(self.S)
        if self.X is not None:
            string += "X:\n {}\n\n".format(self.X)
        if self.dZdt is not None:
            string += "dZdt:\n {}\n\n".format(self.dZdt)
        if self.dXdt is not None:
            string += "dXdt:\n {}\n\n".format(self.dXdt)
        if self.P is not None:
            string += "P:\n {}\n".format(self.P)
        if self.sigma_sq is not None:
            string += "Sigmas2:\n {}\n".format(self.sigma_sq)

        return string

    def compute_var_norm(self,variable_name,norm_type=np.inf):
        var = getattr(self,variable_name)
        var_array = np.array(var)
        return np.linalg.norm(var_array,norm_type)
    
    def load_from_pyomo_model(self, instance, to_load=None):

        if to_load is None:
            to_load = []

        model_variables = set()
        for block in instance.block_data_objects():
            block_map = block.component_map(Var)
            for name in block_map.keys():
                model_variables.add(name)
                
        user_variables = set(to_load)

        if user_variables:
            variables_to_load = user_variables.intersection(model_variables)
        else:
            variables_to_load = model_variables

        diff = user_variables.difference(model_variables)
        if diff:
            print("WARNING: The following variables are not part of the model:")
            print(diff) 
        
        for block in instance.block_data_objects():
            block_map = block.component_map(Var)
            for name in variables_to_load:
                v = block_map[name]
                if v.dim()==0:
                    setattr(self, name, v.value)
                elif v.dim()==1:
                    setattr(self, name, pd.Series(v.get_values()))
                elif v.dim()==2:
                    d = v.get_values()
                    keys = d.keys()
                    if keys:
                        data_frame = df_from_pyomo_data(v)
                    else:
                        data_frame = pd.DataFrame(data=[],
                                                  columns = [],
                                                  index=[])
                    setattr(self, name, data_frame)        
                else:
                    raise RuntimeError('load_from_pyomo_model function not supported for models with variables with dimension>2')
                
    @property
    def parameters(self):
        for k, v in self.P.items():
            print(k, v)
        
