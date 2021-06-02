"""
Replacement functions

This module can be used to replace parameters in the models with constant
values followed by deletion of the parameters from the model.
"""
# KIPET library imports
from kipet.model_tools.visitor_classes import ReplacementVisitor


class ParameterReplacer:
    """This class replaces fixed parameters with numerical values

    :param list models: A list of models
    :param str parameter_name: The name of the parameter to change
    :param list expression_names: The expressions in which to change the parameter
    """

    def __init__(self,
                 models=None,
                 #fix_parameters=None,
                 parameter_name='P',
                 expression_names=['odes'],
                 #user_fixed=True,
                 ):

        """Initialization of ParameterReplacer

        Used to parse Pyomo expressions and replace parameters/variables with other values
        """

        self.models = models if models is not None else []
        #self.fix_parameters = fix_parameters
        self.parameter_name = parameter_name
        self.expression_names = expression_names
        #self.user_fixed = user_fixed

    def remove_fixed_vars(self):
        """Method to remove the fixed parameters from the ConcreteModel. This replaces the variables in place.

        :return: None
        """
        for model in self.models:
    
            # Change to use the direct models
            p_var = getattr(model, self.parameter_name)
    
            # Check which parameters are fixed and add them to the set:
            params_to_fix = set([k for k, v in p_var.items() if v.fixed])

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
    """Takes the non-estiambale parameter and replaces it with its initial value

    :param expression expr: The target ODE constraint in the Pyomo model
    :param str replacement_param: The non-estimable parameter to replace
    :param float change_value: The initial value for the above parameter

    :return expression new_expr: The updated constraints with the
        desired parameter replaced with a float
    
    """
    visitor = ReplacementVisitor()
    visitor.change_replacement(change_value)
    visitor.change_suspect(id(replacement_param))
    new_expr = visitor.dfs_postorder_stack(expr)
    
    return new_expr
