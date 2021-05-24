"""
Convenience class to hold the results from the Pyomo models after simulation
and parameter fitting.

"""
# Standard library imports

# Third party imports
import numpy as np

# Kipet library imports
from kipet.post_model_build.pyomo_model_tools import convert, get_vars

# This needs deletion at some point!
result_vars = ['Z', 'C', 'Cm', 'K', 'S', 'X', 'dZdt', 'dXdt', 'P', 'Pinit', 'sigma_sq', 'estimable_parameters', 'Y', 'UD', 'step']

class ResultsObject(object):
    """Container for all of the results from the Pyomo model"""
    
    def __init__(self):
        """
        A class to store simulation and optimization results.
        
        The ResultsObject instance takes no initial parameters. This will most
        likely change in the near future.
        
        """
        pass

    def __str__(self):
        string = "\nRESULTS\n"
        
        # results_vars = get_results_vars()
        
        for var in result_vars:
            if hasattr(self, var) and getattr(self, var) is not None:
                var_str = var
                if var == 'sigma_sq':
                    var_str = 'Sigmas2'
                string += f'{var_str}:\n {getattr(self, var)}\n\n'
        
        return string
    
    def __repr__(self):
        return self.__str__()

    def compute_var_norm(self, variable_name, norm_type=np.inf):
        var = getattr(self, variable_name)
        var_array = np.array(var)
        return np.linalg.norm(var_array,norm_type)

    def load_from_pyomo_model(self, model, to_load=None):
        """Load variables from the pyomo model into various formats.
        
        This will set the attribute of all the model variables in a specific 
        format depending on the dimensionality of the variable into the 
        ResultsObject.
        
        :param ConcreteModel model: Model of the reaction system
        
        :return: None
        
        """
        variables_to_load = get_vars(model)
    
        for name in variables_to_load:
    
            if name == 'init_conditions':
                continue
            
            var = getattr(model, name)
            var_data = convert(var)
            setattr(self, name, var_data)
                
    @property
    def parameters(self):
        """Returns the parameter dictionary
        
        :return: The parameter values from the model
        :rtype: dict
        
        """
        return self.P
            
    @property
    def show_parameters(self):
        """Displays the parameter values in a conveninent manner
        
        :return: None
        
        """
        print('\nThe estimated parameters are:')
        for k, v in self.P.items():
            print(k, v)
            
    @property
    def variances(self):
        """Displays the variances of the parameters in a conveninent manner
        
        :return: None
        
        """
        print('\nThe estimated variances are:')
        for k, v in self.sigma_sq.items():
            print(k, v)
