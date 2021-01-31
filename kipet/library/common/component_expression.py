from kipet.library.top_level.variable_names import VariableNames
from kipet.library.common.pyomo_model_tools import get_index_sets
from pyomo.environ import *
from pyomo.environ import units as u
from kipet.library.common.VisitorClasses import ReplacementVisitor

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
        
        #self._r_model = model
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
                comp_set = list(index_sets[-1].keys())
                dim = len(index_sets)
                
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
        
        if self.constant_info is not None:
            for mv in self.constant_info.names:
                mv = 'con_' + mv
                #print(mv)
                if hasattr(self._model, mv):
                    #print(mv)
                    mv_obj = getattr(self._model, mv)
                    #print(mv_obj)
                    comp_name = mv[4:]
                        
                    self.var_dict[comp_name] = ModalVar(comp_name, comp_name, mv, self._model)
                    setattr(self, comp_name, mv_obj[0])
        
                        

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
    
    
class Comp_Check():
    
    var = VariableNames()
    
    def __init__(self, model, param_list):
        
        #self._r_model = model
        self._model = model
        self._param_list = param_list
        self.var_dict = {}
        self.var_unit_dict = {}
        self.assign_params_for_units_check()
        # self.assign_rate_vars()
        
    def assign_params_for_units_check(self):
        """Digs through and assigns the variables as top-level attributes
        
        """
        #print('MAKing the con')
        #print(self._param_list)
        m = self._model
        for var in self._param_list:
            #print(var)
            if hasattr(m, var):
                var_obj = getattr(m, var)
                #print(var_obj)
                self.var_dict[var] = var_obj
        
                setattr(self, var, var_obj)
                
        #print(self.var_dict)
                
def get_unit_model(reaction_model):
            
    m = ConcreteModel()
    __var = VariableNames()
    # Pass the kipet model here
    r = reaction_model
    # r = r1
    old_m = r.set_up_model
    
    model_vars = [__var.concentration_model,
                #self.concentration_model_rate,
                  __var.state_model,
                #self.state_model_rate,
                  __var.model_parameter,
                #self.algebraic,
                  #__var.step_variable,
                  __var.model_constant,
                ]
    
    index_vars = model_vars
    
    index_vars += ['con_' + con for con in r.constants.names]
    
    constant_list = ['con_' + con for con in r.constants.names] 
    #index_vars = ['Z', 'P']
     
    CompBlocks = {'P': r.parameters,
                  'Z': r.components,
                  'X': r.components,
                  'Const': r.constants,
                  }
    
    new_params = set()
    
    for index_var in index_vars:
        
        #print(index_var)
        if not hasattr(old_m, index_var):
            continue
        
        var_obj = getattr(old_m, index_var)
        
        for k, v in var_obj.items():
            
            #print(k)
            if isinstance(k, tuple):
                key = k[1]
            else:
                key = k
                
            if key in new_params:
                continue
            
         
            index_name = f'{index_var}_{key}'
            index_name = key
            
            #print(f'Getting {index_name}')
            #print(index_var)
            #print(model_vars)
            if index_var not in constant_list:
                block = CompBlocks[index_var]
            else:
                block = CompBlocks['Const']
                index_var = index_var[4:]
                key = index_var
                index_name = key
            
            m_units = block[key].units
            units = block[key].units.u
            
            #print(m_units)
            #print(type(units))
            
            if units.dimensionless:
                m_units = str(1)
            else:
                m_units = getattr(u, str(m_units))#.format_babel()
                
            #print(block[key].units.u)
            
            setattr(m, index_name, Var(initialize=1, units=m_units))
            #print(getattr(m, index_name).get_units())        
            new_params.add(key)
    
    param_list = list(new_params)
    
    #print(param_list)
    
    #%%
    cn = Comp_Check(m, param_list)
    return cn

def change_to_unit(expr, c_mod, c_mod_new): #, current_model):
        """Method to remove the fixed parameters from the ConcreteModel
        TODO: move to a new expression class
        """
        #print(expr)
        var_dict = c_mod_new.var_dict
        expr_new = expr
        for model_var, var_obj in var_dict.items():
            #print(model_var)
            old_var = getattr(c_mod, model_var)
            #print(old_var)
            new_var = getattr(c_mod_new, model_var)         
            #print(new_var)
            expr_new = _update_expression(expr_new, old_var, new_var)
            
        #print(expr_new)
        return expr_new
  
    # new_expr = change_to_unit(r1.odes['B'].expression, c, cn, '')

def check_units(key, expr, c_mod, c_mod_new):
    
    expr = change_to_unit(expr.expression, c_mod, c_mod_new)
    print(f'The units for expression {key} are:')
    print(u.get_units(expr))

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