"""
General scaling methods for Kipet models

.. warning::

        This class is still under development and may change in future versions!

"""
# Standard library imports
import copy

# Third party imports
from pyomo.environ import Param

from kipet.model_tools.visitor_classes import ReplacementVisitor, ScalingVisitor
# Kipet library imports
from kipet.general_settings.variable_names import VariableNames

__var = VariableNames()


def scale_models(models_input, k_vals, name=None):
    """Takes a model or dict of models and iterates through them to update the
    odes and parameter values

    :param array-like models_input: The models to be updated
    :param dict k_vals: The current parameter values
    :param str name: The name of the model

    :return: The dict of ? and the dict of models
    :rtype: tuple

    """
    models_dict = copy.deepcopy(models_input)
    
    if not isinstance(models_dict, dict):
        key = 'model-1' if name is None else name
        models_dict = {key: models_dict}
    
    d_init = {}
    
    for model in models_dict.values():
        d_init_model = _scale_model(model, k_vals)
        d_init.update(d_init_model)

    return d_init, models_dict


def _scale_model(model, k_vals):
    """Scales an individual model and updates the initial parameter dict

    :param ConcreteModel model: The Pyomo model to be scaled
    :param dict k_vals: The current parameter values

    :return: The scaled parameter dict
    :rtype: dict

    """
    scaled_bounds = {}    

    if not hasattr(model, __var.model_parameter_scaled):
        add_scaling_parameters(model, k_vals)
        scale_parameters(model)
     
        for k, v in getattr(model, __var.model_parameter).items():
            lb, ub = v.bounds
            
            lb = lb/getattr(model, __var.model_parameter_scaled)[k].value
            ub = ub/getattr(model, __var.model_parameter_scaled)[k].value
            
            if ub < 1:
                print('Bounds issue, pushing upper bound higher')
                ub = 1.1
            if lb > 1:
                print('Bounds issue, pushing lower bound lower')
                lb = 0.9
                
            scaled_bounds[k] = lb, ub
                
            model_params = getattr(model, __var.model_parameter)
            model_params[k].setlb(lb)
            model_params[k].setub(ub)
            model_params[k].unfix()
            model_params[k].set_value(1)
            
    parameter_dict = {k: (1, scaled_bounds[k]) for k in k_vals.keys() if k in model_params}
            
    return parameter_dict


def add_scaling_parameters(model, k_vals=None):
    """This will actually set up the model variables in the Pyomo models

    :param ConcreteModel model: A pyomo model
    :param dict k_vals: A dict of parameter values

    :return: None

    """
    print("The model has not been scaled and will now be scaled using K parameters")
    
    if k_vals is None:
    
        setattr(model, __var.model_parameter_scaled, Param(model.parameter_names,                                                                                                                                                            
                    initialize={k: v for k, v in getattr(model, __var.model_parameter).items()},
                    mutable=True,
                    default=1))
    else:
        setattr(model, __var.model_parameter_scaled, Param(model.parameter_names,                                                                                                                                                            
                    initialize={k: v for k, v in k_vals.items() if k in getattr(model, __var.model_parameter)},
                    mutable=True,
                    default=1))
    
    return None


def scale_parameters(model, k_vals=None):
    """If scaling, this multiplies the constants in model.K to each
    parameter in model.P.
    
    I am not sure if this is necessary and will look into its importance.
    """
    if not hasattr(model, __var.model_parameter_scaled):
        add_scaling_parameters(model, k_vals=k_vals)
        
    scale = {}

    for var in __var.model_vars + __var.rate_vars:
        if hasattr(model, var):
            for i in getattr(model, var):
                if var == __var.model_parameter:
                    scale[id(getattr(model, var)[i])] = getattr(model, __var.model_parameter_scaled)[i]
                    continue
                scale[id(getattr(model, var)[i])] = 1

    for k, v in getattr(model, __var.ode_constraints).items():
        scaled_expr = _scale_expression(v.body, scale)
        getattr(model, __var.ode_constraints)[k] = scaled_expr == 0


def remove_scaling(model, bounds=None):
    """This method resets the scaling

    :param ConcreteModel model: A pyomo model
    :param tuple bounds: The bounds to be reset

    :return: None

    """
    if not hasattr(model, __var.model_parameter_scaled):
        raise AttributeError('The model is not scaled')
        
    for param in getattr(model, __var.model_parameter).keys():   
        change_value = getattr(model, __var.model_parameter_scaled)[param].value
        getattr(model, __var.model_parameter)[param].set_value(getattr(model, __var.model_parameter_scaled)[param].value)
        
        for k, v in model.odes.items():
            ep_updated_expr = update_expression(v.body, getattr(model, __var.model_parameter_scaled)[param], 1)
            model.odes[k] = ep_updated_expr == 0
        
        del getattr(model, __var.model_parameter_scaled)[param]
        
    if bounds is not None:
        for param, bound in bounds.items():
            if param in getattr(model, __var.model_parameter):
                getattr(model, __var.model_parameter)[param].setlb(bound[0])
                getattr(model, __var.model_parameter)[param].setub(bound[1])
        
    return None
        
def _scale_expression(expr, scale):
    """Method to set up the scaling

    :param expression expr: The expression to scale
    :param dict scale: The mapping of ids to variables

    :return: The scaled expression
    :rtype: expression

    """
    visitor = ScalingVisitor(scale)
    return visitor.dfs_postorder_stack(expr)


def update_expression(expr, replacement_param, change_value):
    """Takes the non-estiambale parameter and replaces it with its intitial
    value

    .. warning::

        This may not be needed and may be deleted

    :param expression expr: The target ode constraint
    :param str replacement_param: The non-estimable parameter to replace
    :param float change_value: The initial value for the above parameter

    :param expression new_expr: updated constraints with the
        desired parameter replaced with a float

    :return: Updated expression
    :rtype: expression

    """
    visitor = ReplacementVisitor()
    visitor.change_replacement(change_value)
    visitor.change_suspect(id(replacement_param))
    new_expr = visitor.dfs_postorder_stack(expr)
    
    return new_expr
