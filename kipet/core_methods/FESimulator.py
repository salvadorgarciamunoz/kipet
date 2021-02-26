"""
FESimulator
"""
# Standard library imports

# Third party imports
from pyomo.environ import *
from pyomo.dae import *

# KIPET library imports
from kipet.core_methods.fe_factory import fe_initialize
from kipet.core_methods.PyomoSimulator import PyomoSimulator
from kipet.top_level.variable_names import VariableNames

__author__ = 'Michael Short, Kevin McBride'  #: July 2018 - January 2021

class FESimulator(PyomoSimulator):
    
    """This class is just an interface that allows the user of Kipet to easily
    implement the more general fe_factory class designed by David M. Thierry 
    without having to re-write the model to fit those arguments.
    """
    def __init__(self, model):
        """It takes in a standard Kipet/Pyomo model, rewrites it and calls 
        fe_factory. More information on fe_factory is included in that class
        description.
    
        Args:
            model (ConcreteModel): Pyomo model passed from KIPET
        
        """
        super(FESimulator, self).__init__(model)

        # KIPET variables
        self.__var = VariableNames()

        self.p_sim = PyomoSimulator(model)
        self.model = self.p_sim.model
        self.param_dict = {}
        self.param_name = self.__var.model_parameter
        
        # setattr(self.model, 'test_var', Var(self.model.alltime,
        #                                       [0],
        #                                       initialize=1.0))
        # print(self.model.test_var.display())

        self.c_sim = self.model.clone()
        
        # Check all parameters are fixed before simulating
        for param_obj in getattr(self.model, self.__var.model_parameter).values():
            if not param_obj.fixed:
                param_obj.fixed = True
        # Build the parameter dictionary in the format that fe_factory uses    
        model_var = self.__var.model_parameter
        
        for k, v in getattr(self.model, model_var).items():
            self.param_dict[model_var, k] = v.value

        #Build the initial condition dictionary in the format that fe_factory uses
        vars_to_init = [self.__var.concentration_model,
                        self.__var.state_model,
                        ]

        self.ics_ = {}
        for var in vars_to_init:
            self._var_init(var)        
    
        
    def _var_init(self, model_var):
        """Initializes the IC dict for the state variables and modifies it in
        place.
        
        Args:
            model_var (str): The model variable to be used in initializing the
            initial condition dict
            
        Returns:
            None
        """
        if hasattr(self.model, model_var):
            for k, v in getattr(self.model, model_var).items():
                st = self.model.start_time
                if k[0] == st:
                    self.ics_[model_var, k[1]] = v.value
                    
        return None
     
    def call_fe_factory(self, inputs_sub=None, dosing_points=None):    
        """This function applies all the inputs necessary for fe_factory to
        work, using Kipet syntax. Requires external inputs/dosing points to be
        specified with the following arguments.

        Args:
            inputs_sub (dict): dictionary of inputs 
            
            dosing_points (dict): dictionary of the dosing points
            
        Returns:
            None
        
        """
        self.inputs_sub=inputs_sub
        
        init = fe_initialize(self.model, 
                             self.c_sim,
                             init_con="init_conditions_c",
                             param_name=self.param_name,
                             param_values=self.param_dict,
                             inputs_sub=self.inputs_sub)
    
        init.load_initial_conditions(init_cond=self.ics_)

        if dosing_points is not None:
            init.load_discrete_jump(dosing_points)
            
        init.run()

        return None