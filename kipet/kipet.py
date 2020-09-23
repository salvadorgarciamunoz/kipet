"""
This is a wrapper for kipet so that users can more easily access the code
without requiring a plethora of imports

@author: kevin 2020
"""
# Standard library imports
import copy

# Third party imports
import pandas as pd

# Kipet library imports
import kipet.library.data_tools as data_tools
from kipet.library.EstimationPotential import EstimationPotential, reduce_models
from kipet.library.FESimulator import FESimulator
from kipet.library.ParameterEstimator import ParameterEstimator
from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.TemplateBuilder import TemplateBuilder
from kipet.library.VarianceEstimator import VarianceEstimator

from kipet.library.DataHandler import DataBlock, DataSet
from kipet.library.common.model_components import ParameterBlock, ComponentBlock
from kipet.library.common.read_write_tools import set_directory

DEFAULT_DIR = 'data_sets'

class KipetModelBlock():
    
    """This will hold a dict of KipetModel instances
    
    It is not necessary unless many different methods are needed for the 
    underlying KipetModel instances
    
    """
    def __init__(self):
        
        self.models = {}
        
    def __getitem__(self, value):
        
        return self.models[value]
         
    def __str__(self):
        
        block_str = "KipetModelBlock - for multiple KipetModels\n\n"
        
        for name, model in self.models.items():
            block_str += f'{name}\tDatasets: {len(model.datasets)}\n'
        
        return block_str

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        for model, contents in self.models.items():
            yield data
            
    def __len__(self):
        return len(self.models)
    
    def add_model_list(self, model_list):
        """Handles lists of parameters or single parameters added to the model
        """
        for model in model_list:
            self.add_model(model)         
        
        return None
    
    def add_model(self, model):
        
        if isinstance(model, KipetModel):
            self.models[model.name] = model
        else:
            raise ValueError('KipetModelBlock can only add KipetModel instances.')
            
        return None
    
class AttrDict(dict):

    "This class lets you use nested dicts like accessing attributes"
    
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setitem__(self, key, item):
        
        if isinstance(item, dict):
            return super(AttrDict, self).__setitem__(key, AttrDict(item))
        else:
            return dict.__setitem__(self, key, item)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    
class Settings():
    
    """This is a container for all of the options that can be used in Kipet
    Since it can be confusing due to the large number of options, this should
    make it easier for the user to see everything in one spot.
    
    """
    def __init__(self):
        
        self.general = AttrDict()
        self.variance_estimator = AttrDict()
        self.parameter_estimator = AttrDict()
        self.collocation = AttrDict()
        self.solver = AttrDict()
        
        # Initialize to the defaults (can be used at anytime)
        self.reset()
        
    def __str__(self):
        
        m = 25
          
        settings = 'Settings\n\n'
        
        settings += 'General Settings:\n'
        for k, v in self.general.items():
            settings += f'{str(k).rjust(m)} : {v}\n'
        
        settings += '\nCollocation Settings:\n'
        for k, v in self.collocation.items():
            settings += f'{str(k).rjust(m)} : {v}\n'
            
        settings += '\nSimulation Settings:\n'
        for k, v in self.simulator.items():
            settings += f'{str(k).rjust(m)} : {v}\n'
            
        settings += '\nVarianceEstimator Settings:\n'
        for k, v in self.variance_estimator.items():
            settings += f'{str(k).rjust(m)} : {v}\n'
        
        settings += '\nParameterEstimator Settings:\n'
        for k, v in self.parameter_estimator.items():
            settings += f'{str(k).rjust(m)} : {v}\n'
        
        settings += '\nSolver Settings:\n'
        for k, v in self.solver.items():
            settings += f'{str(k).rjust(m)} : {v}\n'
        
        return settings
        
    def __repr__(self):
        return self.__str__()
    
    def reset(self, specific_settings=None):
        """Initializes the settings dicts to their default values"""
        
        general = {'scale_variances': False,
                   # If true, PE is intialized with VE results
                   'initialize_pe' : True,
                    # If true, PE is scaled with VE results
                   'scale_pe' : True,
            }
        
        collocation = {'method': 'dae.collocation',
                       'ncp': 3,
                       'nfe': 50,
                       'scheme': 'LAGRANGE-RADAU',
            }
        
        sim_opt = {'solver': 'ipopt',
                   'tee': False,
                   'solver_opts': AttrDict(),
            }
        
        ve_opt = { 'solver': 'ipopt',
                   'tee': True,
                   'solver_opts': AttrDict(),
                   'tolerance': 1e-5,
                   'max_iter': 15,
                   'method': 'originalchenetal',
                   'use_subset_lambdas': False,
                   'freq_subset_lambdas': 4,
                   'secant_point': 1e-11,
                   'initial_sigmas': 1e-10,
                   'max_device_variance': False,
            }
    
        pe_opt = { 'solver': 'ipopt',
                   'tee': True,
                   'solver_opts': AttrDict(),
                   'covariance': False,
                   'with_d_vars': False,
                   'symbolic_solver_labels': False,
                   'estimability': False,
                   'report_time': False,
                   'model_variance': True,
                   'inputs': None,
                   'inputs_sub': None,
                   'trajectories': None,
                   'fixedtraj': False,
                   'fixedy': False,
                   'yfix': None,
                   'yfixtraj': None,
                   'jump': False,
                   'jump_states': None,
                   'jump_times': None,
                   'feed_times': None,       
                   'unwanted_G': False,
                   'time_variant_G': False,
                   'time_invariant_G': False,
                   'St': dict(),
                   'Z_in': dict(),      
            }
    
        solver = {'nlp_scaling_method': 'gradient-based',
                  'linear_solver': 'ma57',
            }
    
        self.collocation = AttrDict(collocation)
        self.simulator = AttrDict(sim_opt)
        self.general = AttrDict(general)
        self.variance_estimator = AttrDict(ve_opt)
        self.parameter_estimator = AttrDict(pe_opt)
        self.solver = AttrDict(solver)
        
        return None
        
    
class KipetModel():
    
    """This should consolidate all of the Kipet classes into a single class to
    enable a simpler framework for using the software. 
    
    """
    def __init__(self, *args, **kwargs):
        
        self.name = kwargs.get('name', 'Model-1')
        self.model = None
        self.builder = TemplateBuilder()
        self.components = ComponentBlock()   
        self.parameters = ParameterBlock()
        self.datasets = DataBlock()
        self.equations = None
        self.constants = None
        self.results_dict = {}
        self.settings = Settings()
        
        self.odes = None
        self.algs = None
        
    def __repr__(self):
        
        m = 20
        
        kipet_str = f'KipetTemplate Object {self.name}:\n\n'
        kipet_str += f'{"ODEs".rjust(m)} : {hasattr(self, "odes") and getattr(self, "odes") is not None}\n'
        kipet_str += f'{"Algebraics".rjust(m)} : {hasattr(self, "odes") and getattr(self, "odes") is not None}\n'
        kipet_str += f'{"Model".rjust(m)} : {hasattr(self, "model") and getattr(self, "model") is not None}\n'
        kipet_str += '\n'
        
        kipet_str += f'{self.components}\n'
        kipet_str += f'Algebraic Variables:\n{", ".join(self.algebraic_variables)}\n\n'
        kipet_str += f'{self.parameters}\n'
        kipet_str += f'{self.datasets}\n'
        
        return kipet_str
    
    def __str__(self):
        return self.__repr__()
    
    def clone(self, name=None, init=None):
        """Makes a copy of the KipetModel and removes the data. This is done
        to reuse the model, components, and parameters in an easier manner
        
        """
        new_kipet_model = copy.deepcopy(self)
        
        # Reset the datasets
        new_kipet_model.datasets = DataBlock()
        
        # Workaround for missing names
        if name is None:
            new_kipet_model.name = self.name + '_copy'
        else:
            new_kipet_model.name = name
            
        # Workaround for the initializations
        if init is not None:
            if isinstance(init, (list, tuple)):
                for i, comp in enumerate(new_kipet_model.components):
                    print(comp.init)
                    comp.init = init[i]
            elif isinstance(init, dict):
                for k, new_init_val in init.items():
                    new_kipet_model.components[k].init = new_init_val
        else:
            print('Cloned model has the same initial values as original.')
            
        return new_kipet_model
        
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
    
    def add_algebraic_variables(self, *args, **kwargs):
        
        if isinstance(args[0], list):
            self.algebraic_variables = args[0]
        self.builder.add_algebraic_variable(*args, **kwargs)
        return None
    
    def set_directory(self, filename, directory=DEFAULT_DIR):
        """Wrapper for the set_directory method. This replaces the awkward way
        of ensuring the correct directory for the data is used."""

        return set_directory(filename, directory)
    
    def add_equations(self, ode_fun):
        """Wrapper for the set_odes method used in the builder"""
        
        self.odes = ode_fun
        return None
    
    def add_algebraics(self, algebraics):
        """Wrapper for the set_algebraics method used in the builder"""
        
        self.algs = algebraics
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
        scale_parameters = kwargs.pop('scale_parameters', False)
        
        if len(self.components) > 0:
            self.builder.add_components(self.components)
        else:
            raise ValueError('The model has no components')
            
        if len(self.parameters) > 0:
            self.builder.add_parameters(self.parameters)
        else:
            self.allow_optimization = False   
        
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
            
        if hasattr(self, 'algs'):
            self.builder.set_algebraics_rule(self.algs)
        
        self.builder.set_parameter_scaling(scale_parameters)
        #self.builder.add_state_variance(self.components.variances)
        self.model = self.builder.create_pyomo_model(*args, **kwargs)
        
        return None
    
    def simulate(self, options=None, **kwargs):
        """This should try to handle all of the simulation cases"""
    
        self.create_simulator(options, **kwargs)
        self.run_simulation()
        
        return None
    
    def create_simulator(self):
        """This should try to handle all of the simulation cases"""
        
        kwargs = self.settings.collocation
        
        method = kwargs.get('method', 'dae.collocation')
        ncp = kwargs.get('ncp', 3)
        nfe = kwargs.get('nfe', 50)
        scheme = kwargs.get('scheme', 'LAGRANGE-RADAU')
        
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
        
        self.simulator = simulator
        
        return None
        
    def run_simulation(self):
        """Runs the simulations, may be combined with the above at a later date
        
        """
        simulator_options = self.settings.simulator
        self.results = self.simulator.run_sim(**simulator_options)
    
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
        
    def create_estimator(self, estimator=None, **kwargs):
        """This function handles creating the Estimator object"""
        
        if not self.allow_optimization:
            raise AttributeError('This model is not ready for optimization')
        
        method = kwargs.pop('method', 'dae.collocation')
        ncp = kwargs.pop('ncp', 3)
        nfe = kwargs.pop('nfe', 50)
        scheme = kwargs.pop('scheme', 'LAGRANGE-RADAU')
        
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
    
    def _update_related_settings(self):
        
        # Start with what is known
        if self.settings.parameter_estimator['covariance']:
            if self.settings.parameter_estimator['solver'] not in ['k_aug', 'ipopt_sens']:
                raise ValueError('Solver must be k_aug or ipopt_sens for covariance matrix')
        
        # If using sensitivity solvers switch covariance to True
        if self.settings.parameter_estimator['solver'] in ['k_aug', 'ipopt_sens']:
            self.settings.parameter_estimator['covariance'] = True
        
        #Subset of lambdas
        if self.settings.variance_estimator['use_subset_lambdas']:
            self.settings.variance_estimator['subset_lambdas'] = self.reduce_spectra_data_set(self.settings.variance_estimator['freq_subset_lambdas']) 
        
        if self.settings.general.scale_pe:
            self.settings.solver.nlp_scaling_method = 'user-scaling'
    
        if self.settings.variance_estimator.max_device_variance:
            self.settings.parameter_estimator.model_variance = False
    
    def run_opt(self):
        """Run ParameterEstimator but checking for variances - this should
        remove the VarianceEstimator being required to be implemented by the user
        
        """
        if not self.allow_optimization:
            raise ValueError('The model is incomplete for parameter optimization')
            
        # Some settings are required together, this method checks this
        self._update_related_settings()
        
        # Check if all component variances are given; if not run VarianceEstimator
        if not self.components.has_all_variances:
            
            self.create_estimator(estimator='v_estimator', **self.settings.collocation)
            settings_run_ve_opt = self.settings.variance_estimator
            
            if self.settings.variance_estimator.max_device_variance:
                max_device_variance = self.v_estimator.solve_max_device_variance(**settings_run_ve_opt)
            else:
                self.run_ve_opt(**settings_run_ve_opt)
            
        # Create ParameterEstimator
        self.create_estimator(estimator='p_estimator', **self.settings.collocation)
        variances = self.components.variances
        
        # If variance calculated using VarianceEstimator, initialize PE isntance
        if 'v_estimator' in self.results_dict:
            if self.settings.general['initialize_pe']:
                self.initialize_from_trajectory(source=self.results_dict['v_estimator'])
            if self.settings.general['scale_pe']:
                self.scale_variables_from_trajectory(source=self.results_dict['v_estimator'])
            variances = self.results_dict['v_estimator'].sigma_sq
        
        if self.settings.variance_estimator.max_device_variance:
            variances = max_device_variance
        
        if self.settings.general['scale_variances']:
            variances = self._scale_variances(variances)
        
        settings_run_pe_opt = self.settings.parameter_estimator
        settings_run_pe_opt['solver_opts'] = self.settings.solver
        settings_run_pe_opt['variances'] = variances
        
        self.run_pe_opt(**settings_run_pe_opt)
        self.results = self.results_dict['p_estimator']
        
        return self.results
    
    @staticmethod
    def _scale_variances(variances):
        
        max_var = max(variances.values())
        scaled_vars = {comp: var/max_var for comp, var in variances.items()}
        return scaled_vars

    def _run_opt(self, estimator, *args, **kwargs):
        """Runs the respective optimization for the estimator"""
        
        if not hasattr(self, estimator):
            raise AttributeError(f'KipetModel has no attribute {estimator}')
            
        self.results_dict[estimator] = getattr(self, estimator).run_opt(*args, **kwargs)
        return self.results_dict[estimator]
    
    def initialize_from_trajectory(self, variable=None, source=None, obj='p_estimator'):
        """Wrapper for the initialize_from_trajectory method in
        ParameterEstimator
        
        """
        self._from_trajectory('initialize', variable, source, obj)
        return None
    
    def scale_variables_from_trajectory(self, variable=None, source=None, obj='p_estimator'):
        """Wrapper for the scale_varialbes_from_trajectory method in
        ParameterEstimator
        
        """
        self._from_trajectory('scale_variables', variable, source, obj)
        return None
        
    @staticmethod
    def _get_source_data(source, var):
        """Get the correct data from a ResultsObject or a DataFrame"""
        
        if isinstance(source, pd.DataFrame):
            return source
        else:
            return getattr(source, var)
    
    def _from_trajectory(self, category, variable, source, obj):
        """Generic initialization/scaling function"""
        
        estimator = getattr(self, obj)
        method = getattr(estimator, f'{category}_from_trajectory')
        
        if variable is None:
            for var in ['Z', 'C', 'S']:
                method(var, self._get_source_data(source, var))   
        else:
            method(variable, self._get_source_data(source, variable))
        return None
    
    # def fix_from_trajectory(self, variable_name, variable_index, trajectories):
    #     """Wrapper for fix_from_trajectory in PyomoSimulator"""
        
    #     if not hasattr(self, 'simulator'):
    #         raise AttributeError('KipetModel has no simulator')
    #     if not hasattr(self.model, variable_name):
    #         raise AttributeError(f'KipetModel has no algebraic variable {variable_name}')
    #     else:
    #         self.simulator.fix_from_trajectory(self, variable_name, variable_index, trajectories)
                                               
    #     return None
                                               
    def set_known_absorbing_species(self, *args, **kwargs):
        """Wrapper for set_known_absorbing_species in TemplateBuilder
        
        """
        self.builder.set_known_absorbing_species(*args, **kwargs)    
        return None
    
    def reduce_model(self):
        """This calls the reduce_models method in the EstimationPotential
        module to reduce the model based on the reduced hessian parameter
        selection method. 
        """
        parameter_dict = self.parameters.as_dict(bounds=True)

        models_dict_reduced, parameter_data = reduce_models(self.model,
                                                            parameter_dict=parameter_dict,
                                                            )
        self.model = models_dict_reduced['model_1']
        self.using_reduced_model = True
        
        return models_dict_reduced, parameter_data
