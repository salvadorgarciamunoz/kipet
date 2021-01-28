"""Example 1: ODE Simulation with new KipetModel"""

# Standard library imports
import sys

# Third party imports

# Kipet library imports
from kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('reaction-1')
    
    # Add the model parameters
    r1.add_parameter('k1', 2)
    r1.add_parameter('k2', 0.2)
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=1)
    r1.add_component('B', state='concentration', init=0.0)
    r1.add_component('C', state='concentration', init=0.0)
    
    
    # New way of writing ODEs - only after declaring components, algebraics,
    # and parameters
    c = r1.get_model_vars()
    
    # c now holds of all of the pyomo variables needed to define the equations
    rates = {}
    rates['A'] = -c.k1 * c.A
    rates['B'] = c.k1 * c.A - c.k2 * c.B
    rates['C'] = c.k2 * c.B
    
    r1.add_equations(rates)

    # Add dosing points 
    r1.add_dosing_point('A', 3, 0.3)
    
    # Create the model - simulations require times
    r1.set_times(0, 10)
    r1.create_pyomo_model()
    
    # print(r1.model.odes.display())
    
    r1.settings.collocation.ncp = 3
    r1.settings.collocation.nfe = 50
    # r1.settings.simulator.method = 'fe'
    
    #Simulate with default options
    r1.simulate()
    
    if with_plots:
        r1.results.plot()
    #%%
# from kipet.library.top_level.variable_names import VariableNames
# from kipet.library.common.pyomo_model_tools import get_index_sets
# from kipet.library.common.VisitorClasses import ReplacementVisitor
    
# # class TimeReplacer():
    
# #     def __init__(self,
# #                   models=None,
# #                   fix_parameters=None,
# #                   parameter_name='P',
# #                   expression_names=['odes'],
# #                   user_fixed=True,
# #                   ):
      
# #         #self.__var = VariableNames()
        
# #         self.models = models if models is not None else []
# #         self.fix_parameters = fix_parameters
# #         self.parameter_name = parameter_name
# #         self.expression_names = expression_names
# #         self.user_fixed = user_fixed

# #     def change_time(self):
# #         """Method to remove the fixed parameters from the ConcreteModel
# #         """
# #         expr_orig = c.k1 *c.A
# #         print(expr_orig.to_string())
    
# #         expr_new_time = _update_expression(expr_orig, r1.model.Z[0, 'A'], r1.model.Z[10, 'A'])
# #         print(expr_new_time.to_string())
        
# #         return None
    
# #%%
    
# def _update_expression(expr, replacement_param, change_value):
#     """Takes the non-estiambale parameter and replaces it with its intitial
#     value
    
#     Args:
#         expr (pyomo constraint expr): the target ode constraint
        
#         replacement_param (str): the non-estimable parameter to replace
        
#         change_value (float): initial value for the above parameter
        
#     Returns:
#         new_expr (pyomo constraint expr): updated constraints with the
#             desired parameter replaced with a float
    
#     """
#     visitor = ReplacementVisitor()
#     visitor.change_replacement(change_value)
#     visitor.change_suspect(replacement_param)
#     new_expr = visitor.dfs_postorder_stack(expr)
    
#     return new_expr
# #%%    
    
# class ModalVar():
    
#     def __init__(self, name, comp, index, model):
        
#         self.name = name
#         self.comp = comp
#         self.index = index
#         self.model = model

# class Comp():
    
#     var = VariableNames()
    
#     def __init__(self, model):
        
#         # self._r_model = model
#         self._model = model
#         self._model_vars = self.var.model_vars
#         self._rate_vars = self.var.rate_vars
#         self.var_dict = {}
#         self.assign_vars()
#         self.assign_rate_vars()
        
#     def assign_vars(self):
#         """Digs through and assigns the variables as top-level attributes
        
#         """
#         for mv in self._model_vars:
#             if hasattr(self._model, mv):
#                 mv_obj = getattr(self._model, mv)
#                 index_sets = get_index_sets(mv_obj)
#                 comp_set = list(index_sets[-1].keys())
#                 dim = len(index_sets)
                
#                 for comp in comp_set:
#                     self.var_dict[str(comp)] = ModalVar(str(comp), str(comp), mv, self._model)
#                     if dim > 1:
#                         setattr(self, str(comp), mv_obj[0, comp])
#                     else:
#                         setattr(self, str(comp), mv_obj[comp])

#     def assign_rate_vars(self):

#         for mv in self._rate_vars:
#             if hasattr(self._model, mv):
                
#                 print(mv)
                
#                 mv_obj = getattr(self._model, mv)
#                 index_sets = get_index_sets(mv_obj)
#                 comp_set = list(index_sets[-1].keys())
#                 dim = len(index_sets)
                
#                 print(comp_set)
                
#                 for comp in comp_set:
#                     comp_name = f'd{comp}dt'
                    
#                     self.var_dict[comp_name] = ModalVar(comp_name, str(comp), mv, self._model)
#                     if dim > 1:
#                         setattr(self, comp_name, mv_obj[0, comp])
#                     else:
#                         setattr(self, comp_name, mv_obj[comp])            

#         # @staticmethod       
#         # def convert(m, t, var_str):
#         #     """Attempt to make entering equations easier"""
            
#         #     model_var = getattr(m, var)
#         #     model_name = p_tup[1]         
            
#         #     if p_tup[0] == 'P':
#         #         return model_var[model_name]
#         #     else:
#         #         return model_var[t, model_name]
                                
# c = Comp(r1.s_model)
# print(c.__dict__)
    
#     #%%

# reactions = {}
# reactions['A'] = c.dAdt == -c.k1 * c.A
# reactions['B'] = c.dBdt == c.k1 * c.A - c.k2 * c.B
# reactions['C'] = c.dCdt == c.k2 * c.C
# #%%

# def change_time(expr_orig, var_dict, new_time, model):
#     """Method to remove the fixed parameters from the ConcreteModel
#     """
#     print(expr_orig.to_string())

#     expr_new_time = expr_orig
#     visitor = ReplacementVisitor()

#     for model_var, var_obj in var_dict.items():
         
#         if getattr(model, var_obj.index).dim() == 1:
#             continue
         
#         old_var = getattr(model, var_obj.index)[0, var_obj.comp]
        
#         print(f'ID to change: {id(old_var)}')
#         print(id(old_var))
#         new_var = getattr(model, var_obj.index)[new_time, var_obj.comp]
         
#         # visitor.change_replacement(old_var)
#         # visitor.change_suspect(id(new_var))
#         # expr_new_time = visitor.dfs_postorder_stack(expr_new_time)
         
#          #old_var = f'{var_obj.index}[0,{var_obj.comp}]'
        
#         new_var = 'Z[5,A]'
#         print(f'{old_var}\t\t{new_var}')
         
#         expr_new_time = _update_expression(expr_new_time, id(old_var), new_var)
#         print(expr_new_time.to_string())
    
#     return expr_new_time
    
#     #c = Comp(r1.model)
# rA = c.dAdt == -c.k1 * c.A

# var_to_change = rA.args[1].args[1].to_string()
# print(f'ID to change: {id(var_to_change)}')

# rA_1 = change_time(rA, c.var_dict, 5, r1.model)
# # rB_1 = change_time(rB, c.var_dict, 10, r1.model)

# print(rA_1)
# # print(rB_1)

# c = Comp(r1.s_model)

# rZ_1 = change_time(rZ, c.var_dict, 5, r1.s_model)
# print(f'Result: {rZ_1}')    
    

# #%%

# def change_constraint_var(vs, m, con_name, var, t, k):

#     # This is the constraint you want to change (add dummy for the model_var)
#     model_con_obj = getattr(m, con_name)
        
#     indx = (t, k)
#     # this is the variable that will replace the other
#     model_var_new = getattr(m, var)[indx]
#     vs.change_replacement(model_var_new)
    
#     # Deactivate the constraint
#     # model_con_obj[indx].deactivate()
#     var_expr = model_con_obj[indx].expr
    
#     suspect_var = getattr(c, comp)
#     # suspect_var = c.A #m.Z[0,'A']
    
#     #suspect_var = var_expr.args[0].args[1].args[0].args[1]
#     print(suspect_var)
#     # The correct var is c.A
    
#     vs.change_suspect(id(suspect_var))  #: who to replace
#     e_new = vs.dfs_postorder_stack(var_expr)  #: replace
    
#     print(e_new.to_string())
    
#     model_con_obj[indx].set_value(e_new)
#     print(model_con_obj[indx].expr)
    
#     return None

# vs = ReplacementVisitor()
# m = r1.model
# con_name = f'odes'
# t = 5
# comp = 'A'
# var = 'Z'

# change_constraint_var(vs, m, con_name, var, t, comp)



