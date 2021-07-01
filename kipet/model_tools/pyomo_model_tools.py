"""
This module contains various tools that are applied to Pyomo models. Most are
used to extract and format data.

"""
# Standared library imports
import itertools

# Third party imports
import pandas as pd
from pyomo.core.base.param import Param
from pyomo.core.base.set import BoundsInitializer, SetProduct
from pyomo.core.base.var import Var
from pyomo.dae.contset import ContinuousSet
from pyomo.dae.diffvar import DerivativeVar

# KIPET library imports
from kipet.general_settings.variable_names import VariableNames


def get_vars(model):
    """Extract the variable information from the Pyomo model
    
    :param ConcreteModel model: A Pyomo model object
    
    :return list model_varialbes: A list of the model's variables
    
    """
    vars_list = []
    model_Var = list(model.component_map(Var))
    model_dVar = list(model.component_map(DerivativeVar))
    vars_list = model_Var + model_dVar

    return vars_list


def get_vars_block(model):
    """Alternative method for getting the model varialbes
    
    :param ConcreteModel model: A Pyomo model object
    
    :return set model_varialbes: A list of the model's variables
    
    """
    model_variables = set()
    for block in model.block_data_objects():
        block_map = block.component_map(Var)
        for name in block_map.keys():
            model_variables.add(name)
        
    return model_variables


def get_params(model):
    """Returns a list of the model parameters
    
    :param ConcreteModel model: A Pyomo model object
    
    :return list param_list: A list of the model's parameters
    
    """
    param_list = list(model.component_map(Param))
    
    return param_list


def get_result_vars(model):
    """Get the Vars and Params needed for the results object
    
    :param ConcreteModel model: A Pyomo model object
    
    :return list result_vars: A list of the models parameters and variables
    
    """
    result_vars = get_vars(model)
    result_vars += get_params(model)
    
    return result_vars


def get_index_sets(model_var_obj):
    """Retuns a list of the index sets for the model variable
    
    :param model_var_obj: A Pyomo model variable
        
        
    :return list index_set: A list of the variable's indecies
    
    """
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
    
    :param list index_list: list of indicies produced by get_index_sets
        
    :return tuple cont_set_info: (Bool, index of continuous set)
        
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
    """Changes the bounds of the continuous set.
    
    This is primarily used in the fe_factory class to change the bounds on
    the FEs used in simulation.
    
    :param ContinuousSet cs: Continuous set to change bound on
        
    :param tuple new_bounds: New lower and upper bounds for cs
        
    :return: None
    
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


def convert(var):
    """Load variables from the pyomo model into various formats.
    
    If to_load is None all of the model variables will be considered.
    
    :param instance: Pyomo ConcreteModel instance
    :param name str: The name of the model variable
    
    :return var_data: This is the variable and its type depends on the
        dimensions of the data (float, Series, DataFrame)
    
    """ 
    if var.dim()==0:
        var_data = var.value
    elif var.dim()==1:
        var_data = pd.Series(var.extract_values())
    elif var.dim()==2:
        d = var.extract_values()
        keys = d.keys()
        if keys:
            var_data = _df_from_pyomo_data(var)
        else:
            var_data = pd.DataFrame(data=[],
                                    columns = [],
                                    index=[])   
     
    else:
        var_data = _prepare_data_for_init(var)
        
    return var_data
        
        
def _prepare_data_for_init(var):
        """Convert results dict into DataFrame for initialization.
        
        This is used for data with more than two dimensions. It works by
            grouping the remaining dimensions as tuples in the columns of the
            DataFrame.
            
        :param model: The Pyomo model object
        :param var: The target variable in model
        
        :return pandas.DataFrame df: The extracted variable data as a two
            dimensional DataFrame
            
        """
        if len(var) == 0:
            return None
            
        model = var.model()
        index_sets = get_index_sets(var)
        index_dict = index_set_info(index_sets)
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
    
    
def _df_from_pyomo_data(var):
    """Takes a variable object from a Pyomo model and returns a pandas
    DataFrame instance of the data it contains.
    
    The returned DataFrame has time as the index.
    
    :param var: An instance of a Pyomo variable
    
    :return pandas.DataFrame dfs: A DataFrame containing the data of the 
        variable.
    
    """
    val = []
    ix = []
    for index in var:
        ix.append(index)
        try:
            val_raw = var[index].value
        except:
            val_raw = var[index]
            
        if val_raw is None:
            val_raw = 0
        val.append(val_raw)
    
    a = pd.Series(index=ix, data=val)
    dfs = pd.DataFrame(a)
    index = pd.MultiIndex.from_tuples(dfs.index)
   
    dfs = dfs.reindex(index)
    dfs = dfs.unstack()
    dfs.columns = [v[1] for v in dfs.columns]

    return dfs


def model_info(model):
    """This method provides a dict of model attributes to be used in various
    classes.
    
    .. note ::
        
        This may be moved to the TemplateBuilder since this may be redundant
    
    :param ConcreteModel model: A Pyomo model
    
    :return: Useful model attributes
    :rtype: dict
    
    """
    __var = VariableNames()
    model_attrs = {}
    
    model_attrs['mixture_components'] = list(model.mixture_components)
    model_attrs['complementary_states'] = list(model.complementary_states)
    model_attrs['algebraics'] = list(model.algebraics)
    model_attrs['n_components'] = len(model_attrs['mixture_components'])
    model_attrs['n_algebraics'] = len(model_attrs['algebraics'])
    model_attrs['n_complementary_states'] = len(model_attrs['complementary_states'])
   
    model_attrs['known_absorbance_data'] = None
    
    model_attrs['non_absorbing'] = None
    if hasattr(model, 'non_absorbing'):
        model_attrs['non_absorbing'] = list(model.non_absorbing)
        
    model_attrs['known_absorbance'] = None
    if hasattr(model, 'known_absorbance'):
        model_attrs['known_absorbance'] = list(model.known_absorbance)

    if hasattr(model, 'abs_components'):
        model_attrs['abs_components'] = list(model.abs_components.keys())
        model_attrs['nabs_components'] = len(model_attrs['abs_components'])

    model_attrs['huplc_absorbing'] = None
    if hasattr(model, 'huplc_absorbing'):
        model_attrs['huplc_absorbing'] =  list(model.huplc_absorbing)
    
    if hasattr(model, 'huplcabs_components'):
        model_attrs['huplcabs_components'] = list(model.huplcabs_components)
        model_attrs['nhuplcabs_components'] = len(model_attrs['huplcabs_components'])
    
    model_attrs['ipopt_scaled'] = False
    model_attrs['spectra_given'] = hasattr(model, __var.spectra_data)
    model_attrs['concentration_given'] = hasattr(model, __var.concentration_measured) or hasattr(model, __var.user_defined) or hasattr(model, __var.state)
    model_attrs['conplementary_states_given'] = hasattr(model, __var.state)
    model_attrs['absorption_given'] = hasattr(model, __var.spectra_species)
    model_attrs['huplc_given'] = hasattr(model, 'Chat')
    model_attrs['smoothparam_given'] = hasattr(model, __var.smooth_parameter)
                
    return model_attrs
