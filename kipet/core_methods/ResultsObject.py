"""
Results from the various KIPET models are stored here
"""
# Standard library imports
import itertools
from pathlib import Path
import time

# Thirdparty library imports
import datetime
import numpy as np
import pandas as pd
from pyomo.core import *
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.environ import *

# Kipet library imports
from kipet.post_model_build.pyomo_model_tools import (
    get_vars, 
    get_vars_block, 
    get_result_vars,
    get_index_sets,
    index_set_info,
    )         
from kipet.common.read_write_tools import df_from_pyomo_data

# This needs deletion
result_vars = ['Z', 'C', 'Cm', 'K', 'S', 'X', 'dZdt', 'dXdt', 'P', 'Pinit', 'sigma_sq', 'estimable_parameters', 'Y', 'UD', 'step']

class ResultsObject(object):
    """Container for all of the results. Includes plotting functions"""
    
    def __init__(self):
        """
        A class to store simulation and optimization results.
        """
        # Data series
        self.generated_datetime = datetime.datetime
        self.results_name = None
        self.solver_statistics = {}

    def __str__(self):
        string = "\nRESULTS\n"
        
        # results_vars = get_results_vars()
        
        for var in result_vars:
            if hasattr(self, var) and getattr(self, var) is not None:
                var_str = var
                if var == 'sigma_sq':
                    var_str = 'Sigmas2'
                string += f'{var_str}:\n {getattr(self, var)}\n\n'
        
        return string
    
    def __repr__(self):
        return self.__str__()

    def compute_var_norm(self, variable_name, norm_type=np.inf):
        var = getattr(self,variable_name)
        var_array = np.array(var)
        return np.linalg.norm(var_array,norm_type)
    
    @staticmethod
    def prepare_data_for_init(model, var):
        """Convert results dict into df for initialization
        This is used for data with dimensions larger than 2
        """
        #try:
        index_sets = get_index_sets(var)
        index_dict = index_set_info(index_sets)
        # print(var)
        # print(index_dict)
        time_set = index_sets[index_dict['cont_set'][0]].name
        component_indecies = index_dict['other_set']
        component_sets = [index_sets[i].name for i in component_indecies]
        index = getattr(model, time_set).value_list
        columns = list(itertools.product(*[getattr(model, comp_list).value_list for comp_list in component_sets]))
        df = pd.DataFrame(data=None, index=index, columns=columns)

        for i in index:
            for j in columns:
                jl = list(j)
                jl.insert(index_dict['cont_set'][0], i)
                df.loc[i,j] = var[tuple(jl)].value
                
        return df
        #except:
         #   return None

    def load_from_pyomo_model(self, instance, to_load=None):
        """Load variables from the pyomo model into various formats"""
        
        variables_to_load = get_vars(instance)
    
        for name in variables_to_load:
    
            if name == 'init_conditions':
                continue
            
            var = getattr(instance, name)
            if var.dim()==0:
                setattr(self, name, var.value)
            elif var.dim()==1:
                setattr(self, name, pd.Series(var.get_values()))
            elif var.dim()==2:
                d = var.get_values()
                keys = d.keys()
                if keys:
                    data_frame = df_from_pyomo_data(var)
                else:
                    data_frame = pd.DataFrame(data=[],
                                              columns = [],
                                              index=[])
                setattr(self, name, data_frame)        
            else:
                data_df = self.prepare_data_for_init(instance, var)
                setattr(self, name, data_df)
                
    @property
    def parameters(self):
        return self.P
            
    @property
    def show_parameters(self):
        print('\nThe estimated parameters are:')
        for k, v in self.P.items():
            print(k, v)
            
    @property
    def variances(self):
        print('\nThe estimated variances are:')
        for k, v in self.sigma_sq.items():
            print(k, v)