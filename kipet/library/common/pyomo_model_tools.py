"""
Model tools
"""
# Third party imports
from pyomo.core.base.var import Var
from pyomo.core.base.param import Param
from pyomo.core.base.set import BoundsInitializer
from pyomo.core.base.set import SetProduct
from pyomo.dae.contset import ContinuousSet
from pyomo.dae.diffvar import DerivativeVar

def get_vars(model):
    """Extract the variable information from the Pyomo model
    
    """
    vars_list = []
    model_Var = list(model.component_map(Var))
    model_dVar = list(model.component_map(DerivativeVar))
    vars_list = model_Var + model_dVar

    return vars_list

def get_vars_block(instance):
    """Alternative method for getting the model varialbes
    
    """
    model_variables = set()
    for block in instance.block_data_objects():
        block_map = block.component_map(Var)
        for name in block_map.keys():
            model_variables.add(name)
        
    return model_variables

def get_params(instance):
    """Get model params (to delete)
    
    """
    param_list = list(instance.component_map(Param))
    
    return param_list

def get_result_vars(model):
    """Get the Vars and Params needed for the results object
    
    """
    result_vars = get_vars(model)
    result_vars += get_params(model)
    
    return result_vars

def get_index_sets(model_var_obj):
    """Retuns a list of the index sets for the model variable
    
    Args:
        model (ConcreteModel): The pyomo model under investigation
        
        var (str): The name of the variable
        
    Returns:
        
        index_set (list): list of indecies
    
    """
    index_dict = {}
    index_set = []
    index = model_var_obj.index_set()
    if not isinstance(index, SetProduct):
        index_set.append(index)
    elif isinstance(index, SetProduct): # or isinstance(index, SetProduct):
        index_set.extend(index.subsets())
    else:
        return None
    
    return index_set

def index_set_info(index_list):
    """Returns whether index list contains a continuous set and where the
    index is
    
    Args:
        index_list (list): list of indicies produced by get_index_sets
        
    Returns:
        cont_set_info (tuple): (Bool, index of continuous set)
        
    """
    index_dict = {'cont_set': [],
                  'other_set': [],
                  }

    for i, index_set in enumerate(index_list):
        if isinstance(index_set, ContinuousSet):
            index_dict['cont_set'].append(i)
        else:
            index_dict['other_set'].append(i)
    index_dict['other_set'] = tuple(index_dict['other_set'])
        
    return index_dict

def change_continuous_set(cs, new_bounds):
    """Changes the bounds of the continuous set
    
    Args:
        cs (ContinuousSet): continuous set to change bound on
        
        new_bounds (tuple): New lower and upper bounds for cs
        
    Returns:
        None
    
    """
    cs.clear()
    cs._init_domain._set = None
    cs._init_domain._set = BoundsInitializer(new_bounds)
    domain = cs._init_domain(cs.parent_block(), None)
    cs._domain = domain
    domain.parent_component().construct()
    
    for bnd in cs.domain.bounds():
        if bnd is not None and bnd not in cs:
            cs.add(bnd)
    cs._fe = sorted(cs)
    
    return None