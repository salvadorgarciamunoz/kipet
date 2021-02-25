"""
Component expression handling and units model development
"""
# Standard library imports

# Third party imports
from pyomo.core.base.PyomoModel import ConcreteModel
from pyomo.core.base.var import Var
from pyomo.environ import units as u

# KIPET library imports
from kipet.post_model_build.pyomo_model_tools import get_index_sets
from kipet.common.VisitorClasses import ReplacementVisitor
from kipet.post_model_build.replacement import _update_expression
from kipet.top_level.variable_names import VariableNames

string_replace_dict = {
    ' ': '_',
    '-': 'n',
    '+': 'p',
    '.': '_',
    }

class ModalVar():
        
    def __init__(self, name, comp, index, model):
        
        self.name = name
        self.comp = comp
        self.index = index
        self.model = model
    
class Comp():
    
    var = VariableNames()
    
    def __init__(self, model, constant_info=None):
        
        self._model = model
        self._model_vars = self.var.model_vars
        self._rate_vars = self.var.rate_vars
        self.constant_info = constant_info
        self.var_dict = {}
        self.var_unit_dict = {}
        self.assign_vars()
        self.assign_rate_vars()
        
    def assign_vars(self):
        """Digs through and assigns the variables as top-level attributes
        
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

    def assign_rate_vars(self):

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

    def _id(self, comp):
        
        id_ = id(getattr(self, comp))
        print(id_)
        return id_
    
    @property
    def get_var_list(self):
        
        return list(self.var_dict.keys())
    
    
# class Comp_Check():
    
#     var = VariableNames()
    
#     def __init__(self, model, param_list):
        
#         self._model = model
#         self._param_list = param_list
#         self.var_dict = {}
#         self.assign_params_for_units_check()
        
#     def assign_params_for_units_check(self):
#         """Digs through and assigns the variables as top-level attributes
        
#         """
#         m = self._model
#         for var in self._param_list:
#             if hasattr(m, var):
#                 var_obj = getattr(m, var)
#                 self.var_dict[var] = var_obj
#                 setattr(self, var, var_obj)
                
# def get_unit_model(element_dict, set_up_model):
#     """Takes a ReactionModel instance and returns a Comp_Check object
#     representing the model's varialbes with units
#     """            
#     unit_model = ConcreteModel()
#     new_params = set()
#     for elem, comp_block in element_dict.items():
#         if len(comp_block) > 0:
#             for comp in comp_block:
                
#                 new_params.add(comp.name)
#                 m_units = comp.units
#                 if m_units is None:
#                     m_units = str(1)
#                 else:
#                     m_units = getattr(u, str(m_units))
                    
#                 print(m_units)
#                 print(type(m_units))
#                 setattr(unit_model, comp.name, Var(initialize=1, units=m_units))
        
#     c_units = Comp_Check(unit_model, list(new_params))
    
#     return c_units