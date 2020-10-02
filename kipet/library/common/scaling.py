"""
General scaling method for Kipet models
"""

# Third party imports
from pyomo.environ import Param

# Kipet library imports
#from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.common.VisitorClasses import ScalingVisitor     


def scale_models(models_dict, k_vals):
    """Takes a model or dict of models and iterates through them to update the
    odes and parameter values
    
    """
    if not isinstance(models_dict, dict):
        models_dict = {'model-1': models_dict}
    
    d_init = {}
    
    for model in models_dict.values():
        d_init_model = _scale_model(model, k_vals)
        model.K.display()
        model.P.display()
        
        d_init.update(d_init_model)

    return d_init

def _scale_model(model, k_vals):
    """Scales an individual model and updates the initial parameter dict
    
    """
    scaled_bounds = {}    

    if not hasattr(model, 'K'):
        print("The model has not been scaled and will now be scaled using K parameters")
        model.K = Param(model.parameter_names, 
                         initialize={k: v for k, v in k_vals.items() if k in model.P},
                         mutable=True,
                         default=1)
         
        #model_ps = PyomoSimulator(model)
        scale_parameters(model)
     
        for k, v in model.P.items():
            lb, ub = v.bounds
            
            lb = lb/model.K[k].value
            ub = ub/model.K[k].value
            
            if ub < 1:
                print('Bounds issue, pushing upper bound higher')
                ub = 1.1
            if lb > 1:
                print('Bounds issue, pushing lower bound lower')
                lb = 0.9
                
            scaled_bounds[k] = lb, ub
                
            model.P[k].setlb(lb)
            model.P[k].setub(ub)
            model.P[k].unfix()
            model.P[k].set_value(1)
             
    parameter_dict = {k: (1, scaled_bounds[k]) for k in k_vals.keys() if k in model.P}
            
    return parameter_dict
        
def scale_parameters(model):
    """If scaling, this multiplies the constants in model.K to each
    parameter in model.P.
    
    I am not sure if this is necessary and will look into its importance.
    """
    #if self.model.K is not None:
    scale = {}
    for i in model.P:
        scale[id(model.P[i])] = model.K[i]

    for i in model.Z:
        scale[id(model.Z[i])] = 1
        
    for i in model.dZdt:
        scale[id(model.dZdt[i])] = 1
        
    for i in model.X:
        scale[id(model.X[i])] = 1

    for i in model.dXdt:
        scale[id(model.dXdt[i])] = 1

    for k, v in model.odes.items(): 
        scaled_expr = scale_expression(v.body, scale)
        model.odes[k] = scaled_expr == 0
        
def scale_expression(expr, scale):
    
    visitor = ScalingVisitor(scale)
    return visitor.dfs_postorder_stack(expr)