"""
Convenience class to hold the results from the Pyomo models after simulation
and parameter fitting.

"""
# Standard library imports

# Third party imports
import numpy as np

# Kipet library imports
from kipet.model_tools.pyomo_model_tools import convert, get_vars

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
        self.parameter_covariance = None

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

    def _confidence_interval_display(self, interval=0.95):
        """
        Function to display calculated confidence intervals

        :param dict variances: The component variances

        :return: None

        """
        margin = 15
        if self.parameter_covariance is not None:
            deviations = self.deviations(interval)
            
            print(f'\n# Parameter Values with confidence: ({int(interval*100)}%)')
            for k, p in self.P.items():
                if k in self.variances:
                    print(f'{k.rjust(margin)} = {p:0.4e} +/- {deviations[k]:0.4e}') 
            if hasattr(self, 'Pinit'):
                for k, p in self.Pinit.items():
                    print(f'{k.rjust(margin)} = {p:0.4e} +/- {deviations[k]:0.4e}') 
            if hasattr(self, 'time_step_change'):
                for k, p in self.time_step_change.items():
                    if k in self.variances:
                        print(f'{k.rjust(margin)} = {p:0.4e} +/- {deviations[k]:0.4e}') 
        else:
            print(f'\n# Parameter Values')
            for k, p in self.P.items():
                print(f'{k.rjust(margin)} = {p:0.4e}')
            if hasattr(self, 'Pinit'):
                for k, p in self.Pinit.items():
                    print(f'{k.rjust(margin)} = {p:0.4e}') 
            if hasattr(self, 'time_step_change'):
                for k, p in self.time_step_change.items():
                    print(f'{k.rjust(margin)} = {p:0.4e}') 
            
        return None

    def deviations(self, interval=0.95):
        """Calculates bounds for the parameters based on STD and condifence
        interval provided
        
        :param float interval: The confidence interval
        
        """
        if not hasattr(self, 'parameter_covariance') or self.parameter_covariance is None:
            return None
        
        import scipy.stats as st
        dev = st.norm.ppf(1-(1-interval)/2)
        
        deviations = {k: dev*v**0.5 for k, v in self.variances.items()}
        
        return deviations

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
                

    def parameters(self, interval=0.95):
        """Returns the parameter dictionary
        
        :return: The parameter values from the model
        :rtype: dict
        
        """
        return self._confidence_interval_display(interval)
            

    def show_parameters(self, interval=0.95):
        """Displays the parameter values in a convenient manner
        
        :return: None
        
        """
        return self._confidence_interval_display(interval)
            
    @property
    def variances(self):
        """Displays the variances of the parameters in a convenient manner
        
        :return: None
        
        """
        if not hasattr(self, 'parameter_covariance') or self.parameter_covariance is None:
            return None
        
        var = dict(zip(self.parameter_covariance.columns, 
                       np.diag(self.parameter_covariance.values)
            ))
        
        return var