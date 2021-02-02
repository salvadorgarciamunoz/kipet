"""
General KIPET Model Settings

This should remove all of the fixed variable names across the KIPET software
"""

class VariableNames(object):
    
    """Names of the KIPET model variables"""
    
    def __init__(self):
    
        # Variables for KIPET models    
    
        self.model_parameter = 'P'
        self.model_parameter_scaled = 'K'
        self.concentration_measured = 'Cm'
        self.concentration_spectra = 'C'
        self.concentration_spectra_abs = 'Cs'
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
        
        self.DEBUG = False
        
        
    @property
    def time_dependent_variables(self):
        
        """
        ['Z', 'dZdt', 'S', 'C', 'X', 'dXdt', 'U', 'Y']
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
        """
        ['Z', 'dZdt', 'X', 'dXdt']
        """
        model_vars = [self.concentration_model,
                      self.concentration_model_rate,
                      self.state_model,
                      self.state_model_rate,
                      ]
        
        return model_vars
    
    @property
    def model_vars(self):
        """
        ['Z', 'X', 'P', 'Y', 'step', 'Const']
        """
        model_vars = [self.concentration_model,
                      #self.concentration_model_rate,
                      self.state_model,
                      #self.state_model_rate,
                      self.model_parameter,
                      self.algebraic,
                      self.step_variable,
                      self.model_constant,
                      ]
        
        return model_vars
    
    @property
    def rate_vars(self):
        
        model_vars = [
            self.concentration_model_rate,
            self.state_model_rate,
            ]
        
        return model_vars
    
    @property
    def plot_vars(self):
        
        model_vars = [
            self.concentration_model_rate,
            self.state_model_rate,
            self.algebraic,
            self.step_variable,
            self.spectra_species,
            ]
        
        return model_vars