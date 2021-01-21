"""
Model tools
"""

from pyomo.core.base.var import Var
from pyomo.core.base.param import Param
from pyomo.dae.diffvar import DerivativeVar

def get_vars(model):
    """Extract the variable information from the Pyomo model"""
    
    vars_list = []
    
    model_Var = list(model.component_map(Var))
    model_dVar = list(model.component_map(DerivativeVar))
    
    vars_list = model_Var + model_dVar

    #print(vars_list)

    return vars_list

def get_vars_block(instance):
    """Alternative method for getting the model varialbes"""
    
    model_variables = set()
    for block in instance.block_data_objects():
        block_map = block.component_map(Var)
        for name in block_map.keys():
            model_variables.add(name)
        
    return model_variables

def get_params(instance):
    
    param_list = list(instance.component_map(Param))
    #print(param_list)
    
    return param_list

def get_result_vars(model):
    """Get the Vars and Params needed for the results object"""
    
    result_vars = get_vars(model)
    result_vars += get_params(model)
    return result_vars