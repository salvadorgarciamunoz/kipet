"""
This is a wrapper for kipet so that users can more easily access the code
without requiring a plethora of imports

@author: kevin 2020
"""
# Standard library imports
import copy

# Third party imports

# Kipet library imports
import kipet.library.data_tools as data_tools
from kipet.library.EstimationPotential import EstimationPotential
from kipet.library.FESimulator import FESimulator
from kipet.library.ParameterEstimator import ParameterEstimator
from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.TemplateBuilder import TemplateBuilder
from kipet.library.VarianceEstimator import VarianceEstimator

from kipet.library.DataHandler import DataBlock, DataSet
from kipet.library.common.model_components import ParameterBlock, ComponentBlock
from kipet.library.common.read_write_tools import set_directory

DEFAULT_DIR = 'data_sets'

class KipetModel():
    
    """This should consolidate all of the Kipet classes into a single class to
    enable a simpler framework for using the software. 
    
    """
    def __init__(self, *args, **kwargs):
        
        self.model = None
        self.builder = TemplateBuilder()
        self.options = {'solver' : 'ipopt'}
        self.components = ComponentBlock()   
        self.parameters = ParameterBlock()
        self.datasets = DataBlock()
        self.equations = None
        self.constants = None
        self.results = {}
        
    def __repr__(self):
        
        kipet_str = 'KipetTemplate Object:\n\n'
        kipet_str += f'Has ODEs: {hasattr(self, "odes")}\n'
        kipet_str += f'Has Model: {hasattr(self, "model") and getattr(self, "model") is not None}\n'
        kipet_str += '\n'
        kipet_str += f'{self.components}\n'
        kipet_str += f'{self.parameters}\n'
        kipet_str += f'{self.datasets}\n'
        
        return kipet_str
    
    def __str__(self):
        return self.__repr__()
    
    def add_component(self, *args, **kwargs):
        """Add the components to the Kipet instance
        
        Args:
            components (list): list of Component instances
            
        Returns:
            None
            
        """
        self.components.add_component(*args, **kwargs)
        return None
    
    def add_parameter(self, *args, **kwargs):
        """Add the parameters to the Kipet instance
        
        Args:
            parameters (list): list of Parameter instances
            
            factor (float): defaults to 1, the scalar multiple of the parameters
            for simulation purposes
            
        Returns:
            None
            
        """
        self.parameters.add_parameter(*args, **kwargs)
        return None
    
    def add_dataset(self, *args, **kwargs):
        """Add the datasets to the Kipet instance
        
        Args:
            datasets (list): list of Parameter instances
            
            factor (float): defaults to 1, the scalar multiple of the parameters
            for simulation purposes
            
        Returns:
            None
            
        """
        self.datasets.add_dataset(*args, **kwargs)
        return None
    
    def set_directory(self, filename, directory=DEFAULT_DIR):
        """Wrapper for the set_directory method. This replaces the awkward way
        of ensuring the correct directory for the data is used."""

        return set_directory(filename, directory)
    
    def add_equations(self, ode_fun):
        """Wrapper for the set_odes method used in the builder"""
        
        self.odes = ode_fun
        return None
        
    def create_pyomo_model(self, *args, **kwargs):
        """Adds the component, parameter, data, and odes to the TemplateBuilder
        instance and creates the model. The model is stored under self.model
        and there is nothing returned.

        Args:
            None

        Returns:
            None

        """
        if len(self.components) > 0:
            self.builder.add_components(self.components)
        else:
            raise ValueError('The model has no components')
            
        if len(self.parameters) > 0:
            self.builder.add_parameters(self.parameters)
        else:
            raise ValueError('The model has no parameters')
        
        if len(self.datasets) > 0:
            self.builder.input_data(self.datasets)
            self.allow_optimization = True
        elif len(self.datasets) == 0:
            self.allow_optimization = False
        else:
            pass
            
        if hasattr(self, 'odes'):
            self.builder.set_odes_rule(self.odes)
        else:
            raise ValueError('The model requires a set of ODEs')
        
        self.model = self.builder.create_pyomo_model(*args, **kwargs)
        
        return None
    
    def simulate(self, options=None, **kwargs):
        """This should try to handle all of the simulation cases"""
        
        method = kwargs.pop('method', 'dae.collocation')
        ncp = kwargs.pop('ncp', 3)
        nfe = kwargs.pop('nfe', 50)
        scheme = kwargs.pop('scheme', 'LAGRANGE-RADAU')
        
        default_options = {
            'solver' : 'ipopt',
            }
        
        options = options.copy() if options is not None else default_options
        
        if method == 'fe':
            simulation_class = FESimulator
        elif method == 'dae.collocation':
            simulation_class = PyomoSimulator
        
        self.s_model = self.model.clone()
        
        for param in self.s_model.P.values():
            param.fix()
        
        simulator = simulation_class(self.s_model)
        simulator.apply_discretization(method,
                                       ncp = ncp,
                                       nfe = nfe,
                                       scheme = scheme)
        
        if method == 'fe':
            simulator.call_fe_factory()
        
        self.run_simulation(simulator, options)
        
        return None
    
    def run_simulation(self, simulator, options):
        """Runs the simulations, may be combined with the above at a later date
        """
    
        solver = options.pop('solver', 'ipopt')
        solver_options = options.pop('solver_options', [])
    
        self.results['sim'] = simulator.run_sim(solver,
                                          tee=False,
                                          solver_options=solver_options,
                                          )
    
        return None
    
    def reduce_spectra_data_set(self, dropout=4):
        """To reduce the computational burden, this can be used to reduce 
        the amount of spectral data used
        
        """
        A_set = [l for i, l in enumerate(self.model.meas_lambdas) if (i % dropout == 0)]
        return A_set
    
    def create_variance_estimator(self, options=None, **kwargs):
        """This is a wrapper for creating the VarianceEstimator"""
        
        self.create_estimator(options, estimator='v_estimator', **kwargs)
        return None
        
    def create_parameter_estimator(self, options=None, **kwargs):
        """This is a wrapper for creating the ParameterEstiamtor"""
        
        self.create_estimator(options, estimator='p_estimator', **kwargs)
        return None
        
    def create_estimator(self, options, estimator=None, **kwargs):
        """This function handles creating the Estimator object"""
        
        if not self.allow_optimization:
            raise AttributeError('This model is not ready for optimization')
        
        method = kwargs.pop('method', 'dae.collocation')
        ncp = kwargs.pop('ncp', 3)
        nfe = kwargs.pop('nfe', 50)
        scheme = kwargs.pop('scheme', 'LAGRANGE-RADAU')
        
        default_options = {
            'solver' : 'ipopt',
            }
        
        options = options.copy() if options is not None else default_options
        
        if estimator == 'v_estimator':
            Estimator = VarianceEstimator
            est_str = 'VarianceEstimator'
            
        elif estimator == 'p_estimator':
            Estimator = ParameterEstimator
            est_str = 'ParameterEstimator'
            
        else:
            raise ValueError('Keyword argument estimator must be p_estimator or v_estimator.')
        
        if self.model is None:
            raise ValueError(f'Cannot create {est_str} without pyomo model')
        else:
            setattr(self, f'{estimator[0]}_model', self.model.clone())
            setattr(self, estimator, Estimator(getattr(self, f'{estimator[0]}_model')))
            getattr(self, estimator).apply_discretization(method,
                                                  ncp = ncp,
                                                  nfe = nfe,
                                                  scheme = scheme)
        return None
    
    def run_ve_opt(self, *args, **kwargs):
        """Wrapper for run_opt method in VarianceEstimator"""
        
        self._run_opt('v_estimator', *args, **kwargs)
        return None
    
    def run_pe_opt(self, *args, **kwargs):
        """Wrapper for run_opt method in ParameterEstimator"""
        
        self._run_opt('p_estimator', *args, **kwargs)
        return None
    
    def _run_opt(self, estimator, *args, **kwargs):
        """Runs the respective optimization for the estimator"""
        
        if not hasattr(self, estimator):
            raise AttributeError(f'KipetModel has no attribute {estimator}')
            
        self.results[estimator] = getattr(self, estimator).run_opt(*args, **kwargs)
        return self.results[estimator]
    
    def initialize_from_trajectory(self, variable=None, source=None):
        """Wrapper for the initialize_from_trajectory method in
        ParameterEstimator
        
        """
        self._from_trajectory('initialize', variable, source)
        return None
    
    def scale_variables_from_trajectory(self, variable=None, source=None):
        """Wrapper for the scale_varialbes_from_trajectory method in
        ParameterEstimator
        
        """
        self._from_trajectory('scale_variables', variable, source)
        return None
        
    def _from_trajectory(self, category, variable, source):
        """Generic initialization/scaling function"""
        
        method = getattr(self.p_estimator, f'{category}_from_trajectory')
        if variable is None:
            for var in ['Z', 'C', 'S']:
                method(var, getattr(source, var))   
        else:
            method(variable, getattr(source, variable))
        return None
    
    def set_known_absorbing_species(self, *args, **kwargs):
        """Wrapper for set_known_absorbing_species in TemplateBuilder
        
        """
        self.builder.set_known_absorbing_species(*args, **kwargs)    
        return None
    
    @property
    def result(self):
        return self.results['p_estimator']