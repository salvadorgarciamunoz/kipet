import datetime
import pandas as pd
import numpy as np
from pyomo.core import *
from pyomo.environ import *

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
        self.C = None
        self.S = None
        self.sigma_sq = None
        self.device_variance = None
        self.P = None
        self.dZdt = None
        self.dXdt = None

    def load_from_pyomo_model(self,instance,to_load=[]):

        model_variables = set()
        for block in instance.block_data_objects():
            block_map = block.component_map(Var)
            for name in block_map.iterkeys():
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
                    setattr(self,name,v.value)
                elif v.dim()==1:
                    setattr(self,name,pd.Series(v.get_values()))
                elif v.dim()==2:
                    d = v.get_values()
                    keys = d.keys()
                    if keys:
                        split_keys = zip(*keys)
                        first_set = set(split_keys[0])
                        second_set = set(split_keys[1])
                        s_first_set = sorted(first_set)
                        s_second_set = sorted(second_set)
                        m = len(first_set)
                        n = len(second_set)

                        v_values = np.zeros((m,n))
                        for i,w in enumerate(s_first_set):
                            for j,k in enumerate(s_second_set):
                                v_values[i,j] = d[w,k]

                        data_frame = pd.DataFrame(data=v_values,
                                                  columns = s_second_set,
                                                  index=s_first_set)
                    else:
                        data_frame = pd.DataFrame(data=[],
                                                  columns = [],
                                                  index=[])
                    setattr(self,name,data_frame)        
                else:
                    raise RuntimeError('load_from_pyomo_model function not supported for models with variables with dimension>2')
                
