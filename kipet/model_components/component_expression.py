"""
Component expression handling and units model development
"""
# KIPET library imports
from kipet.model_tools.pyomo_model_tools import get_index_sets
from kipet.general_settings.variable_names import VariableNames

string_replace_dict = {
    ' ': '_',
    '-': 'n',
    '+': 'p',
    '.': '_',
    }


class ModalVar:
    """Simple class to model a model variable"""
    def __init__(self, name, comp, index, model):
        
        self.name = name
        self.comp = comp
        self.index = index
        self.model = model


class Comp:
    """This class is needed to handle pyomo variables and return a dict of these variables.

    This will not be used by the user directly. This is one of the key classes that allows the user to use
    dummy pyomo variables in building the KIPET models.

    :param ConcreteModel model: The model we are going to use (dummy model)
    :param
    """
    
    var = VariableNames()
    
    def __init__(self, model):
        
        self._model = model
        self._model_vars = self.var.model_vars
        self._rate_vars = self.var.rate_vars
        self.var_dict = {}
        self.var_unit_dict = {}
        self.assign_vars()
        self.assign_rate_vars()
        
    def assign_vars(self):
        """Digs through and assigns the variables as top-level attributes

        :return: None
        """
        list_of_vars = self._model_vars
        
        for mv in list_of_vars:
            if hasattr(self._model, mv):
                mv_obj = getattr(self._model, mv)
                index_sets = get_index_sets(mv_obj)
                
                dim = len(index_sets)
                if dim > 1:
                    indx = 1
                else:
                    indx = 0
                
                comp_set = list(index_sets[indx].keys())
                for comp in comp_set:
                    
                    comp = str(comp)
                    if isinstance(comp, str):
                        if isinstance(comp[0], int):
                            comp.insert(0, 'y')
                        
                        comp_name = comp
                        for k, v in string_replace_dict.items():
                            comp_name.replace(k, v)
                    
                    self.var_dict[comp_name] = ModalVar(comp_name, comp_name, mv, self._model)
                    
                    if dim > 1:
                        setattr(self, comp_name, mv_obj[0, comp])
                    else:
                        setattr(self, comp_name, mv_obj[comp])

        return None

    def assign_rate_vars(self):
        """Assigns the rate variables as top-level attributes

        :return: None

        """
        for mv in self._rate_vars:
            if hasattr(self._model, mv):
                
                mv_obj = getattr(self._model, mv)
                index_sets = get_index_sets(mv_obj)
                comp_set = list(index_sets[-1].keys())
                dim = len(index_sets)
                
                for comp in comp_set:
                    
                    if isinstance(comp, str):
                        comp_name = comp
                        comp_name.replace(' ', '_')
                    elif isinstance(comp, int):
                        comp_name = f'y{comp}'
                    
                    comp_name = f'd{comp_name}dt'
                    self.var_dict[comp_name] = ModalVar(comp_name, str(comp), mv, self._model)
                    if dim > 1:
                        setattr(self, comp_name, mv_obj[0, comp])
                    else:
                        setattr(self, comp_name, mv_obj[comp])

        return None
    
    @property
    def get_var_list(self):
        """Returns a list of variables in the Comp dict

        :return: A list of variables
        :rtype: list

        """
        return list(self.var_dict.keys())
