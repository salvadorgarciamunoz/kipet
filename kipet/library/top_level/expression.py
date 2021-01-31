"""
Expression Classes
"""
from kipet.library.common.VisitorClasses import ReplacementVisitor
from pyomo.environ import ConcreteModel, Var, Objective
from pyomo.environ import units as u
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent, check_units_equivalent

class Expression():
    
    
    def __init__(self,
                 name,
                 expression):
        
        self.name = name
        self.expression = expression
        
    def __str__(self):
        
        return self.expression.to_string()
    
    # def units(self):
        
    #     return(u.get_units(self.expression))
    #%%
#     def change_to_unit(expr, c_mod, c_mod_new, var): #, current_model):
#         """Method to remove the fixed parameters from the ConcreteModel
#         TODO: move to a new expression class
#         """
#         print(expr)
#         var_dict = c_mod_new.var_dict
#         expr_new = expr
#         for model_var, var_obj in var_dict.items():
#             print(model_var)
#             old_var = getattr(c_mod, model_var)
#             print(old_var)
#             new_var = getattr(c_mod_new, model_var)         
#             print(new_var)
#             expr_new = _update_expression(expr_new, old_var, new_var)
            
#         print(expr_new)
#         return expr_new
  
#     new_expr = change_to_unit(r1.odes['B'].expression, c, cn, '')


# #%%
#     @staticmethod
#     def _update_expression(expr, replacement_param, change_value):
#         """Takes the non-estiambale parameter and replaces it with its intitial
#         value
        
#         Args:
#             expr (pyomo constraint expr): the target ode constraint
            
#             replacement_param (str): the non-estimable parameter to replace
            
#             change_value (float): initial value for the above parameter
            
#         Returns:
#             new_expr (pyomo constraint expr): updated constraints with the
#                 desired parameter replaced with a float
        
#         """
#         visitor = ReplacementVisitor()
#         visitor.change_replacement(change_value)
#         visitor.change_suspect(id(replacement_param))
#         new_expr = visitor.dfs_postorder_stack(expr)       
#         return new_expr

    
    
#%% 
# ex = Expression('r1', c.k1*c.A)

# from pyomo.environ import ConcreteModel, Var, Objective
# from pyomo.environ import units as u
# from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent, check_units_equivalent
# model = ConcreteModel()
# model.acc = Var(initialize=5.0, units=u.m/u.s**2)
# model.obj = Objective(expr=(model.acc - 9.81*u.m/u.s**2)**2)
# assert_units_consistent(model.obj) # raise exc if units invalid on obj
# assert_units_consistent(model) # raise exc if units invalid anywhere on the model
# assert_units_equivalent(model.obj.expr, u.m**2/u.s**4) # raise exc if units not equivalent
# print(u.get_units(model.obj.expr)) # print the units on the objective
# print(check_units_equivalent(model.acc, u.m/u.s**2))
#%%

