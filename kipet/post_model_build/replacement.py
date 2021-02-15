"""
Replacement functions

This module can be used to replace parameters in the models with constant
values followed by deletion of the parameters from the model.
"""
# Standard library imports
import copy
import numpy as np
import pandas as pd

# Third party imports
from pyomo.environ import *

# KIPET library imports
from kipet.common.VisitorClasses import ReplacementVisitor 
#from kipet.top_level.variable_names import VariableNames

class ParameterReplacer():
    
    def __init__(self,
                 models=None,
                 fix_parameters=None,
                 parameter_name='P',
                 expression_names=['odes'],
                 user_fixed=True,
                 ):
      
        #self.__var = VariableNames()
        
        self.models = models if models is not None else []
        self.fix_parameters = fix_parameters
        self.parameter_name = parameter_name
        self.expression_names = expression_names
        self.user_fixed = user_fixed

    def remove_fixed_vars(self):
        """Method to remove the fixed parameters from the ConcreteModel
        """
        for model in self.models:
    
            # Change to use the direct models
            #model = _model.p_model       
            p_var = getattr(model, self.parameter_name)
    
            # Check which parameters are fixed and add them to the set:
            params_to_fix = set([k for k, v in p_var.items() if v.fixed])
            
            # If the user defined parameters are also to be fixed:
            #if self.user_fixed:
            #    params_to_fix.add(set(self.fix_parameters))
        
            # Replace the parameters with numeric values and remove Var from model
            for param in params_to_fix:   
                parameter_to_change = param
                if parameter_to_change in p_var.keys():
                    change_value = p_var[param].value
                
                    for expression_var in self.expression_names:
                        e_var = getattr(model, expression_var)
        
                        for k, v in e_var.items(): 
                            ep_updated_expr = _update_expression(v.body, p_var[param], change_value)
                            e_var[k] = ep_updated_expr == 0
                
                    if hasattr(model, 'parameter_names'):
                        model.parameter_names.remove(param)
                    del p_var[param]
        
        return None
    
def _update_expression(expr, replacement_param, change_value):
    """Takes the non-estiambale parameter and replaces it with its intitial
    value
    
    Args:
        expr (pyomo constraint expr): the target ode constraint
        
        replacement_param (str): the non-estimable parameter to replace
        
        change_value (float): initial value for the above parameter
        
    Returns:
        new_expr (pyomo constraint expr): updated constraints with the
            desired parameter replaced with a float
    
    """
    visitor = ReplacementVisitor()
    visitor.change_replacement(change_value)
    visitor.change_suspect(id(replacement_param))
    new_expr = visitor.dfs_postorder_stack(expr)
    
    return new_expr