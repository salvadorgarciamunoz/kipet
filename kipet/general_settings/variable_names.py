"""
General KIPET Model Varialbe names
"""

class VariableNames(object):
    
    """Names of the KIPET model variables

    This provides a central location to reduce errors in developing KIPET modules by forcing a naming convention.
    It also provides useful collections of variables for use in many other functions.

    This is not really needed by the user and should only be used by developers.
    """
    def __init__(self):
    
        # Strong defaults
        self.volume_name = 'V'
    
        # Variables for KIPET models    
        self.model_parameter = 'P'
        self.model_parameter_scaled = 'K'
        self.concentration_measured = 'Cm'
        self.concentration_spectra = 'C'
        #self.concentration_spectra_abs = 'Cs'
        self.concentration_model = 'Z'
        self.concentration_model_rate = f'd{self.concentration_model}dt'
        self.state = 'U'
        self.state_model = 'X'
        self.state_model_rate = f'd{self.state_model}dt'
        self.spectra_species = 'S'
        self.spectra_data = 'D'
        self.user_defined = 'UD'
        self.huplc_data = 'Dhat'
        self.smooth_parameter = 'Ps'
        self.algebraic = 'Y'
        self.unwanted_contribution = 'g'
        self.ode_constraints = 'odes'
        
        self.dosing_variable = 'Dose'
        self.dosing_component = 'd_var'
        
        self.time_step_change = 'time_step_change'
        self.step_variable = 'step'
        # Debug options
        
        self.model_constant = 'Const'
        self.concentration_init = 'Pinit'
        
        self.DEBUG = False
        
        
    @property
    def optimization_variables(self):
        """These are the independent variables that need to be fixed in 
        simulations
        """
        
        model_vars = [self.model_parameter,
                      self.time_step_change,
                      self.concentration_init
                      ]
        
        return model_vars
    
    @property
    def time_dependent_variables(self):
        """Property to return variables that are time dependent:

        :Defaults:

            ['Z', 'dZdt', 'S', 'C', 'X', 'dXdt', 'U', 'Y']

        :return model_vars: The list of target variables
        :rtype: list

        """
        model_vars = [self.concentration_model,
                      self.concentration_model_rate,
                      self.spectra_species,
                      self.concentration_spectra,
                      self.state_model,
                      self.state_model_rate,
                      self.state,
                      self.algebraic,
                      ]
        
        return model_vars
        
    @property
    def modeled_states(self):
        """Property to return variables that are modeled:

        :Defaults:

            ['Z', 'dZdt', 'X', 'dXdt']

        :return model_vars: The list of target variables
        :rtype: list

        """
        model_vars = [self.concentration_model,
                      self.concentration_model_rate,
                      self.state_model,
                      self.state_model_rate,
                      ]
        
        return model_vars
    
    @property
    def model_vars(self):
        """Property to return component variables:

        :Defaults:

            ['Z', 'X', 'P', 'Y', 'step', 'Const']

        :return model_vars: The list of target variables
        :rtype: list

        """
        model_vars = [self.concentration_model,
                      self.state_model,
                      self.model_parameter,
                      self.algebraic,
                      self.step_variable,
                      self.model_constant,
                      ]
        
        return model_vars
    
    @property
    def rate_vars(self):
        """Property to return rate variables:

        :return model_vars: The list of target variables
        :rtype: list

        """
        model_vars = [
            self.concentration_model_rate,
            self.state_model_rate,
            ]
        
        return model_vars
    
    @property
    def plot_vars(self):
        """Property to return plotted variables

        :return model_vars: The list of target variables
        :rtype: list

        """
        model_vars = [
            self.concentration_model_rate,
            self.state_model_rate,
            self.algebraic,
            self.step_variable,
            self.spectra_species,
            ]
        
        return model_vars
