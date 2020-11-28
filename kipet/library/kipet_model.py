"""
This is a wrapper for kipet so that users can more easily access the code
without requiring a plethora of imports

Should block be the standard? This could check the number of models in the set
and perform the calculations as singles or MEE

@author: kevin 2020
"""
# Standard library imports
import collections
import copy
import pathlib
import weakref

# Third party imports
import numpy as np
import pandas as pd
from pyomo.environ import Var

# Kipet library imports
import kipet.library.core_methods.data_tools as data_tools
from kipet.library.core_methods.EstimationPotential import (
    reduce_model,
    replace_non_estimable_parameters,
    )
# from kipet.library.core_methods.EstimationPotential_working import (
#     reduce_model,
#    )
from kipet.library.core_methods.FESimulator import FESimulator
from kipet.library.core_methods.MEE_new import MultipleExperimentsEstimator
from kipet.library.core_methods.ParameterEstimator import ParameterEstimator
from kipet.library.core_methods.PyomoSimulator import PyomoSimulator
from kipet.library.core_methods.TemplateBuilder import TemplateBuilder
from kipet.library.core_methods.VarianceEstimator import VarianceEstimator

from kipet.library.common.pre_process_tools import decrease_wavelengths
#from kipet.library.common.read_write_tools import set_directory
from kipet.library.post_model_build.scaling import scale_models

from kipet.library.mixins.TopLevelMixins import WavelengthSelectionMixins

from kipet.library.top_level.datahandler import DataBlock, DataSet
from kipet.library.top_level.helper import DosingPoint
from kipet.library.top_level.model_components import ParameterBlock, ComponentBlock
from kipet.library.top_level.settings import Settings, USER_DEFINED_SETTINGS

DEFAULT_DIR = 'data_sets'

class KipetModel():
    
    """This will hold a dict of ReactionModel instances
    
    It is not necessary unless many different methods are needed for the 
    underlying KipetModel instances
    
    """
    def __init__(self):
        
        self.models = {}
        self.settings = Settings(category='block')
        #self.variances = {}
        self.no_variances_provided = False
        self.results = {}
        
    def __getitem__(self, value):
        
        return self.models[value]
         
    def __str__(self):
        
        block_str = "KipetModel\n\n"
        
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
        
        if isinstance(model, ReactionModel):
            self.models[model.name] = model
        else:
            raise ValueError('KipetModel can only add ReactionModel instances.')
            
        return None
    
    @staticmethod
    def add_noise_to_data(data, noise):
        """Wrapper for adding noise to data after data has been added to
        the specific ReactionModel
        
        """
        return data_tools.add_noise_to_signal(data, noise)       
    
    def new_reaction(self, name, model_to_clone=None, items_not_copied=None):
        
        """New reactions can be added to KIPET using this function
        
        Args:
            name (str): The name of the model/experiment used in all references
                made to it in KIPET, especially in python dicts
            model_to_clone (ReactionModel): You can copy an existing ReactionModel by
                adding it here
            items_not_copied (list): This is a list of strings for the various
                attributes in the ReactionModel that should not be copied
                example: ['datasets'] ==> does not copy the dataset into the 
                new model
                
        Returns:
            self.models[name] (ReactionModel): This is the newly created
                ReactionModel instance
                
        """
        if model_to_clone is None:
        
            self.models[name] = ReactionModel(name=name)
            self.models[name].settings.general.data_directory = self.settings.general.data_directory
        
        else:
            if isinstance(model_to_clone, ReactionModel):
                kwargs = {}
                kwargs['name'] = name
                if items_not_copied is not None:
                    if isinstance(items_not_copied, str):
                        items_not_copied = [items_not_copied]
                    if isinstance(items_not_copied, list):
                        for item in items_not_copied:
                            kwargs[item] = False
                self.models[name] = model_to_clone.clone(**kwargs)             
            else:
                raise ValueError('KipetModel can only add ReactionModel instances.')
        
        return self.models[name]
    
    def add_reaction(self, kipet_model_instance):
    
        if isinstance(model, ReactionModel):
            self.models[model.name] = model
            self.models[model.name].settings.general.data_directory = self.settings.general.data_directory
        else:
            raise ValueError('KipetModel can only add ReactionModel instances.')

        return None
    
    def create_multiple_experiments_estimator(self, *args, **kwargs):
        """A quick wrapper for MEE without big changes
        
        """
        #variances = kwargs.pop('variances', None)
        
        # self.variances_provided = True
        # if variances is None or len(variances) == 0:
        #     self.variances_provided = False
        
        if 'spectral' in self.data_types:
            self.settings.parameter_estimator.spectra_problem = True
        else:
            self.settings.parameter_estimator.spectra_problem = False    
            
        # self.variances = {}

        for name, model in self.models.items():
            model.settings.collocation = self.settings.collocation
            model.populate_template()
            
            for dataset in model.datasets:
                if self.settings.general.freq_wavelength_subset is not None:
                    if model.datasets[dataset.name].category == 'spectral':
                        freq = self.settings.general.freq_wavelength_subset
                        model.datasets[dataset.name].data = decrease_wavelengths(dataset.data, freq)
               
                # if variances is not None:
                #     self.variances[name] = variances
                
        self.mee = MultipleExperimentsEstimator(self.models)
        self.mee.spectra_problem = self.settings.parameter_estimator.spectra_problem
        
    def run_opt(self, *args, **kwargs):
        
        if len(self.models) > 1:
            self.create_multiple_experiments_estimator(*args, **kwargs)
            self.run_multiple_experiments_estimator()
        
        else:
            reaction_model = self.models[list(self.models.keys())[0]]
            results = reaction_model.run_opt()
            self.results[reaction_model.name] = results
            
        return None
        
    def run_multiple_experiments_estimator(self, **kwargs):
        
        """Main function controlling the parameter estimation for multiple
        experiments. It defaults to solving the MEE, but can be selected to
        simply solve each model individually. This is the basis for the KIPET
        individual models too.
        
        """
        run_full_model = kwargs.get('multiple_experiments', True)
        
        #if not self.variances_provided:
        self.calculate_variances()
        #else:
        #    self.mee.variances = self.variances
        self.calculate_parameters()
        
        if run_full_model:
            self.run_full_model()
        
        return None
        
    def calculate_variances(self):
        """Uses the ReactionModel framework to calculate variances instead of 
        repeating this in the MEE
        
        """
        variance_dict = {}
        self.mee.variances = {}
        
        for model in self.models.values():
            
            if len(model.variances) == 0:
                model.create_variance_estimator(**self.settings.collocation)
                model.run_ve_opt()
                variance_dict[model.name] = model.results_dict['v_estimator'].sigma_sq
                self.mee.variances[model.name] = variance_dict[model.name]
            else:
                variance_dict[model.name] = model.variances
                self.mee.variances[model.name] = variance_dict[model.name]
        
        self.results_variances = variance_dict
        self.mee._variance_solved = True
        self.mee.variance_results = variance_dict
        self.mee.opt_model = {k: v.model for k, v in self.models.items()}
        return variance_dict
    
    def calculate_parameters(self):
        """Uses the ReactionModel framework to calculate parameters instead of 
        repeating this in the MEE
        
        """
        parameter_estimator_model_dict = {}
        parameter_dict = {}
        
        for model in self.models.values():
            
            settings_run_pe_opt = model.settings.parameter_estimator
            settings_run_pe_opt['solver_opts'] = model.settings.solver
            settings_run_pe_opt['variances'] = self.results_variances[model.name]
            settings_run_pe_opt['confindence_interval'] = self.settings.parameter_estimator.confidence
            model.create_parameter_estimator(**self.settings.collocation)
            model.run_pe_opt(**settings_run_pe_opt)
            parameter_dict[model.name] = model.results_dict['p_estimator']
            parameter_estimator_model_dict[model.name] = model.p_estimator
        
        self.mee.initialization_model = parameter_estimator_model_dict
        self.mee.confidence_interval = self.settings.parameter_estimator.confidence
        
        list_components = {}
        for name, model in self.models.items():
            list_components[name] = [comp.name for comp in model.components if comp.state == 'concentration']
        self.mee._sublist_components = list_components

        return parameter_dict
    
    def run_full_model(self):
        
        global_params = list(self.global_params)
        list_params_across_blocks = global_params
        list_species_across_blocks = list(self.all_species)
        list_waves_across_blocks = list(self.all_wavelengths)

        results = self.mee.solve_consolidated_model(global_params,
                                 list_params_across_blocks,
                                 list_species_across_blocks,
                                 list_waves_across_blocks,
                                 **self.settings.parameter_estimator)
    
        self.results = results
        for key, results_obj in self.results.items():
            results_obj.file_dir = self.settings.general.charts_directory
        return results
    
    def write_data_file(self, filename, data, directory=None, filetype='csv'):
        """Method to write data to a file using KipetModel
        
        Args:
            filename (str): the name of the file (plus relative directory)
            
            data (pandas DataFrame): The data to be written to the file
            
            directory (str): absolute directory to use instead
            
            filetype (str): the filetype to be used (in case not in file name)
        
        Returns:
            None
        
        """
        _filename = filename
        if directory is None:
            _filename = _set_directory(self, filename)
        else:
            _filename = pathlib.Path(directory).joinpath(filename)
        if not _filename.parent.is_dir():
            _filename.parent.mkdir(exist_ok=True)
        data_tools.write_file(_filename, data)
        
        return None
        
    def read_data_file(self, filename, directory=None):
        """Method to read data file using KipetModel
        
        Args:
            filename (str): the name of the file (plus relative directory)
            
            data (pandas DataFrame): The data to be written to the file
            
            directory (str): absolute directory to use instead
            
        Returns:
            read_data (pandas DataFrame): The data read from the file
        
        """
        _filename = filename
        if directory is None:
            _filename = _set_directory(self, filename)
        else:
            _filename = pathlib.Path(directory).joinpath(filename)
        
        read_data = data_tools.read_file(_filename)
        return read_data
    
    @property
    def all_params(self):
        set_of_all_model_params = set()
        for name, model in self.models.items():
            set_of_all_model_params = set_of_all_model_params.union(model.parameters.names)
        return set_of_all_model_params
    
    @property
    def global_params(self):
        
        parameter_counter = collections.Counter()
        global_params = set()
        for name, model in self.models.items():
            for param in model.parameters.names:
                parameter_counter[param] += 1
        
        for param, num in parameter_counter.items():
            if num > 1:
                global_params.add(param)
            
        return global_params
    
    @property
    def all_wavelengths(self):
        set_of_all_wavelengths = set()
        for name, model in self.models.items():
            set_of_all_wavelengths = set_of_all_wavelengths.union(list(model.model.meas_lambdas))
        return set_of_all_wavelengths
    
    @property
    def all_species(self):
        set_of_all_species = set()
        for name, model in self.models.items():
            set_of_all_species = set_of_all_species.union(model.components.names)
        return set_of_all_species
    
    @property
    def model_list(self):
        return [model.name for model in self.models.values()]
    
    @property
    def data_types(self):
        data_types = set()
        for name, kipet_model in self.models.items():
            for dataset_name, dataset in kipet_model.datasets.datasets.items():
                data_types.add(dataset.category)
        return data_types
    
    @property
    def show_parameters(self):
        for reaction, model in self.models.items():
            print(f'{reaction}')
            model.results.show_parameters
            
    def plot(self, *args, **kwargs):
        for reaction, model in self.models.items():
            file_dir = self.settings.general.charts_directory
            kwargs['file_dir'] = file_dir
            print(kwargs)
            model.results.plot(*args, **kwargs)
        
    def test_version(self):
        print('yes, this is new')
        return None
    
    
class ReactionModel(WavelengthSelectionMixins):
    
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
        self.constants = None
        self.results_dict = {}
        self.settings = Settings(category='model')
        self.algebraic_variables = []
        
        self.variances = {}
        
        self.odes = None
        self.algs = None
        self.custom_objective = None
        
        self.dosing_var = None
        self.dosing_points = None
        self._has_dosing_points = False
        
        self._has_non_absorbing_species = False
        
        self._var_to_fix_from_trajectory = []
        self._var_to_initialize_from_trajectory = []

    def __repr__(self):
        
        m = 20
        
        kipet_str = f'ReactionModel Object {self.name}:\n\n'
        kipet_str += f'{"ODEs".rjust(m)} : {hasattr(self, "odes") and getattr(self, "odes") is not None}\n'
        kipet_str += f'{"Algebraics".rjust(m)} : {hasattr(self, "odes") and getattr(self, "odes") is not None}\n'
        kipet_str += f'{"Model".rjust(m)} : {hasattr(self, "model") and getattr(self, "model") is not None}\n'
        kipet_str += '\n'
        
        kipet_str += f'{self.components}\n'
        kipet_str += f'Algebraic Variables:\n{", ".join([str(av) for av in self.algebraic_variables])}\n\n'
        kipet_str += f'{self.parameters}\n'
        kipet_str += f'{self.datasets}\n'
        
        return kipet_str
    
    def __str__(self):
        return self.__repr__()
    
    def _unwanted_G_initialization(self, *args, **kwargs):
        """Prepare the ParameterEstimator model for unwanted G contributions
        
        """
        self.builder.add_qr_bounds_init(bounds=(0,None),init=1.1)
        self.builder.add_g_bounds_init(bounds=(0,None))
        
        return None
    
    def add_dosing_point(self, component, time, step):
        """Add a dosing point or several (check template for how this is handled)
        
        """
        conversion_dict = {'state': 'X', 
                           'concentration': 'Z',
                           }
        
        if self.dosing_var is None:
            raise AttributeError('ReactionModel needs a designated algebraic variable for dosing')
        
        if component not in self.components.names:
            raise ValueError('Invalid component name')
        
        dosing_point = DosingPoint(component, time, step)

        model_var = conversion_dict[self.components[component].state]
        
        if self.dosing_points is None:
            self.dosing_points = {}
        
        if model_var not in self.dosing_points.keys():
            self.dosing_points[model_var] = [dosing_point]
        else:
            self.dosing_points[model_var].append(dosing_point)
            
        self._has_dosing_points = True
        
    def set_dosing_var(self, var):
        
        """Check when multiple dosing vars are needed"""
        
        # if not isinstance(var, list):
        #     var = [var]
        
        # for _var in var:
        if var not in self.algebraic_variables:
            raise ValueError('Not a valid algebraic variable')
            
        self.dosing_var = var

        return None
    
    def call_fe_factory(self):
        """Somewhat of a wrapper for this simulator method, but better"""

        self.simulator.call_fe_factory({'Y': [self.dosing_var]}, self.dosing_points)
        
        return None
    
    def clone(self, *args, **kwargs):
        """Makes a copy of the ReactionModel and removes the data. This is done
        to reuse the model, components, and parameters in an easier manner
        
        """
        new_kipet_model = copy.deepcopy(self)
        
        name = kwargs.get('name', self.name + '_copy')
        copy_model = kwargs.get('model', True)
        copy_builder = kwargs.get('builder', True)
        copy_components = kwargs.get('components', True)   
        copy_parameters = kwargs.get('parameters', True)
        copy_datasets = kwargs.get('datasets', True)
        copy_constants = kwargs.get('constants', True)
        copy_settings = kwargs.get('settings', True)
        copy_algebraic_variables = kwargs.get('alg_vars', True)
        copy_odes = kwargs.get('odes', True)
        copy_algs = kwargs.get('algs', True)
        
        # Reset the datasets
        
        new_kipet_model.name = name
        
        if not copy_model:
            new_kipet_model.model = None
        
        if not copy_builder:
            new_kipet_model.builder = TemplateBuilder()
            
        if not copy_components:
            new_kipet_model.components = ComponentBlock()
        
        if not copy_parameters:
            new_kipet_model.parameters = ParameterBlock()
            
        if not copy_datasets:
            del new_kipet_model.datasets
            new_kipet_model.datasets = DataBlock()
            
        if not copy_constants:
            new_kipet_model.constants = None
            
        if not copy_algebraic_variables:
            new_kipet_model.algebraic_variables = []
            
        if not copy_settings:
            new_kipet_model.settings = Settings()
            
        if not copy_odes:
            new_kipet_model.odes = None
            
        if not copy_algs:
            new_kipet_model.algs = None
        
        list_of_attr_to_delete = ['p_model', 'v_model', 'p_estimator',
                                  'v_estimator', 'simulator']
        
        for attr in list_of_attr_to_delete:
            if hasattr(new_kipet_model, attr):
                setattr(new_kipet_model, attr, None)
        
        new_kipet_model.results_dict = {}
            
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
        name = kwargs.get('name', None)
        if len(args) > 0:
            name = args[0]
        filename = kwargs.get('file', None)
        data = kwargs.pop('data', None)
        category = kwargs.get('category', None)
        
        # Check if file name is given and add directory (general)
        if filename is not None:
            filename = _set_directory(self, filename)
            kwargs['file'] = filename
            #kwargs['data'] = None
            
            # Read data from file
            dataframe = data_tools.read_file(filename)
        
        elif filename is None and data is not None:
            dataframe = data
        
        else:
            raise ValueError('User must provide filename or dataframe')
        
        # Now we have the dataframe of data - check labels for components
        if category is None:
            self._check_data_category(name, dataframe, **kwargs)    
        else:
            self._add_categorized_dataset(name, dataframe, **kwargs)
        
        return None
    
    def _check_data_category(self, name, data, **kwargs):
        """Checks the category for data entered without a category"""
        
        # if components have already been entered, check them
        if len(self.components) > 0:
            data_labels = []
            
            # The types of data that can be autormated (concentration and state)
            concentration_data_labels = []
            state_data_labels = []

            for col in data.columns:
                if col in self.components.names:
                    if self.components[col].state == 'concentration':
                        concentration_data_labels.append(col)
                    elif self.components[col].state == 'state':
                        state_data_labels.append(col)
                        
            if len(concentration_data_labels) > 0:
                state_data = data.loc[:, concentration_data_labels]
                df_name = name if name is not None else 'C_data'
                self.datasets.add_dataset(df_name, category='concentration', data=state_data)
                
            if len(state_data_labels) > 0:
                state_data = data.loc[:, state_data_labels]
                df_name = name if name is not None else 'U_data'
                self.datasets.add_dataset(df_name, category='state', data=state_data)

        else:
            raise AttributeError('Data must have a cateogory or be matched to component data')
            
        remove_negatives = kwargs.get('remove_negatives', False)
        if remove_negatives:
            self.datasets[df_name].remove_negatives()
            
        return None
    
    def _add_categorized_dataset(self, name, data, **kwargs):
        """Specific function for adding concentration data"""
        
        category = kwargs.get('category', None)

        # General trajectory data
        if category == 'trajectory':
            df_name = name if name is not None else 'Traj_data'
            self.datasets.add_dataset(df_name, category=category, data=data)
        elif category == 'concentration':
            df_name = name if name is not None else 'C_data'
            self.datasets.add_dataset(df_name, category=category, data=data)
        elif category == 'state':
            df_name = name if name is not None else 'U_data'
            self.datasets.add_dataset(df_name, category=category, data=data)
        elif category == 'spectral':
            df_name = name if name is not None else 'D_data'
            self.datasets.add_dataset(df_name, category=category, data=data)
        else:
            df_name = name if name is not None else 'UD_data'
            self.datasets.add_dataset(df_name, category='custom', data=data)
                
        remove_negatives = kwargs.get('remove_negatives', False)
        if remove_negatives:
            self.datasets[df_name].remove_negatives()
        
        return None
    
    def add_algebraic_variables(self, *args, **kwargs):
        
        if isinstance(args[0], list):
            self.algebraic_variables = args[0]
        self.builder.add_algebraic_variable(*args, **kwargs)
        return None
    
    def set_times(self, start_time=None, end_time=None):
        """Add times to model for simulation (overrides data-based times)"""
        
        if start_time is None or end_time is None:
            raise ValueError('Time needs to be a number')
        
        self.settings.general.simulation_times = (start_time, end_time)
        return None
    
    # def set_directory(self, filename, abs_dir=False):
    #     """Wrapper for the set_directory method. This replaces the awkward way
    #     of ensuring the correct directory for the data is used."""

    #     directory = self.settings.general.data_directory
    #     print(f'The current data directory is : {directory}')
    #     file_path = pathlib.Path(directory).joinpath(filename)
    #     print(f'The data file is the following: {file_path}')
        
    #     return file_path
    
    # def write_file(self, filename, data, directory=None, filetype='csv'):
    #     """Method to write data to a file using KipetModel
    #     """
    #     _filename = filename
        
    #     if directory is None:
    #         _filename = self.set_directory(filename)
    #     else:
    #         _filename = pathlib.Path(directory).joinpath(filename)
        
    #     data_tools.write_file(_filename, data, filetype)
        
    #     return None
        
    # def read_data_file(self, filename, directory=None):
    #     """Method to read data file using KipetModel
    #     """
    #     _filename = filename
        
    #     if directory is None:
    #         _filename = self.set_directory(filename)
    #     else:
    #         _filename = pathlib.Path(directory).joinpath(filename)
        
    #     return data_tools.read_file(_filename)
    
    def add_equations(self, ode_fun):
        """Wrapper for the set_odes method used in the builder"""
        
        self.odes = ode_fun
        return None
    
    def add_algebraics(self, algebraics):
        """Wrapper for the set_algebraics method used in the builder"""
        
        self.algs = algebraics
        return None
    
    def add_objective_from_algebraic(self, algebraic_var):
        """Wrapper for the set_algebraics method used in the builder"""
        
        self.custom_objective = algebraic_var
        return None
    
    def populate_template(self, *args, **kwargs):
        
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
            
        if hasattr(self, 'odes') and self.odes is not None:
            self.builder.set_odes_rule(self.odes)
        else:
            raise ValueError('The model requires a set of ODEs')
            
        if hasattr(self, 'algs') and self.algs is not None:
            self.builder.set_algebraics_rule(self.algs)
            
        if hasattr(self, 'custom_objective') and self.custom_objective is not None:
            self.builder.set_objective_rule(self.custom_objective)
        
        self.builder.set_parameter_scaling(self.settings.general.scale_parameters)
        self.builder.add_state_variance(self.components.variances)
        
        if self._has_dosing_points:
            self._add_feed_times()
            
        # It seems this is repetitive - refactor
        self.builder._G_contribution = self.settings.parameter_estimator.G_contribution
        
        if self.settings.parameter_estimator.G_contribution is not None:
            self._unwanted_G_initialization()
        
        start_time, end_time = None, None
        if self.settings.general.simulation_times is not None:
            print(f'times are: {type(self.settings.general.simulation_times)}')
            start_time, end_time = self.settings.general.simulation_times
       
        return start_time, end_time
        
    def create_pyomo_model(self, *args, **kwargs):
        """Adds the component, parameter, data, and odes to the TemplateBuilder
        instance and creates the model. The model is stored under self.model
        and there is nothing returned.

        Args:
            None

        Returns:
            None

        """
        if hasattr(self, 'model'):
            del self.model
            
        start_time, end_time = self.populate_template(*args, **kwargs)
        self.model = self.builder.create_pyomo_model(start_time, end_time)
        
        if self._has_non_absorbing_species:
            self.builder.set_non_absorbing_species(self.model, self.non_abs_list, check=True)    
        
        if hasattr(self,'fixed_params') and len(self.fixed_params) > 0:
            for param in self.fixed_params:
                self.model.P[param].fix()
            
        return None
    
    def _add_feed_times(self):
        
        feed_times = set()
        
        for model_var, dp in self.dosing_points.items():
            for point in dp:
                feed_times.add(point.time)
        
        self.builder.add_feed_times(list(feed_times))
        return None
    
    def _from_trajectories(self, estimator):
        """This handles all of the fixing, initializing, and scaling from 
        trajectory data
        
        """
        if len(self._var_to_fix_from_trajectory) > 0:
            for fix in self._var_to_fix_from_trajectory:
                if isinstance(fix[2], str):
                    if fix[2] in self.datasets.names:
                        fix[2] = self.datasets[fix[2]].data
                getattr(self, estimator).fix_from_trajectory(*fix)
                
        if len(self._var_to_initialize_from_trajectory) > 0:
            for init in self._var_to_initialize_from_trajectory:
                if isinstance(init[1], str):
                    if init[1] in self.datasets.names:
                        init[1] = self.datasets[init[1]].data
                getattr(self, estimator).initialize_from_trajectory(*init)
                
        return None
    
    def simulate(self):
        """This should try to handle all of the simulation cases"""
    
        self.create_simulator()
        self._from_trajectories('simulator')
        self.run_simulation()
        
        return None
    
    def create_simulator(self):
        """This should try to handle all of the simulation cases"""
        
        sim_set_up_options = copy.copy(self.settings.simulator)
        dis_method = sim_set_up_options.pop('method', 'dae.collocation')
        
        if self.dosing_var is not None:
            dis_method = 'fe'
        
        kwargs = self.settings.collocation
        
        method = kwargs.get('method', 'dae.collocation')
        ncp = kwargs.get('ncp', 3)
        nfe = kwargs.get('nfe', 50)
        scheme = kwargs.get('scheme', 'LAGRANGE-RADAU')
        
        if dis_method == 'fe':
            simulation_class = FESimulator
        else:
            simulation_class = PyomoSimulator
        
        if self.model is None:
            self.create_pyomo_model(*self.settings.general.simulation_times)
        
        self.s_model = self.model.clone()
        
        for param in self.s_model.P.values():
            param.fix()
        
        simulator = simulation_class(self.s_model)
        simulator.apply_discretization(method,
                                       ncp = ncp,
                                       nfe = nfe,
                                       scheme = scheme)
        
        if self.dosing_var is not None and hasattr(self.s_model, 'Y'):
            for key in simulator.model.alltime.value:
                simulator.model.Y[key, self.dosing_var].set_value(key)
                simulator.model.Y[key, self.dosing_var].fix()
        
        self.simulator = simulator
        
        return None
        
    def run_simulation(self):
        """Runs the simulations, may be combined with the above at a later date
        
        """
        if self._has_dosing_points:
            self.call_fe_factory()
        
        simulator_options = self.settings.simulator
        simulator_options.pop('method', None)
        self.results = self.simulator.run_sim(**simulator_options)
        self.results.file_dir = self.settings.general.charts_directory
    
        return None
    
    def reduce_spectra_data_set(self, dropout=4):
        """To reduce the computational burden, this can be used to reduce 
        the amount of spectral data used
        
        """
        A_set = [l for i, l in enumerate(self.model.meas_lambdas) if (i % dropout == 0)]
        return A_set
    
    def bound_profile(self, var, bounds):
        """Wrapper for TemplateBuilder bound_profile method"""
        
        self.builder.bound_profile(var=var, bounds=bounds)
        return None
    
    def create_variance_estimator(self, **kwargs):
        """This is a wrapper for creating the VarianceEstimator"""
        if len(kwargs) == 0:
            kwargs = self.settings.collocation
        
        if self.model is None:    
            self.create_pyomo_model()  
        
        self.create_estimator(estimator='v_estimator', **kwargs)
        return None
        
    def create_parameter_estimator(self, **kwargs):
        """This is a wrapper for creating the ParameterEstiamtor"""
        if len(kwargs) == 0:
            kwargs = self.settings.collocation
            
        if self.model is None:    
            self.create_pyomo_model()  
            
        self.create_estimator(estimator='p_estimator', **kwargs)
        return None
        
    def create_estimator(self, estimator=None, **kwargs):
        """This function handles creating the Estimator object"""
        
        # if not self.allow_optimization:
        #     raise AttributeError('This model is not ready for optimization')
        
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
        
        setattr(self, f'{estimator[0]}_model', self.model.clone())
        setattr(self, estimator, Estimator(getattr(self, f'{estimator[0]}_model')))
        getattr(self, estimator).apply_discretization(method,
                                              ncp = ncp,
                                              nfe = nfe,
                                              scheme = scheme)
        return None
    
    # def solve_variance_given_delta(self):
    #     """Wrapper for this VarianceEstimator function"""
    #     variances = self.v_estimator.solve_sigma_given_delta(**self.settings.variance_estimator)
    #     return variances
        
    def run_ve_opt(self, *args, **kwargs):
        """Wrapper for run_opt method in VarianceEstimator"""
        
        kwargs.update(self.settings.variance_estimator)
        
        if kwargs['method'] == 'direct_sigmas':
            worst_case_device_var = self.v_estimator.solve_max_device_variance(**kwargs)
            kwargs['device_range'] = (self.settings.variance_estimator.best_accuracy, worst_case_device_var)
            
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
        if self.settings.variance_estimator['freq_subset_lambdas'] is not None:
            if type(self.settings.variance_estimator['freq_subset_lambdas'], int):
                self.settings.variance_estimator['subset_lambdas' ] = self.reduce_spectra_data_set(self.settings.variance_estimator['freq_subset_lambdas']) 
        
        if self.settings.general.scale_pe and not self.settings.general.no_user_scaling:
            self.settings.solver.nlp_scaling_method = 'user-scaling'
    
        if self.settings.variance_estimator.max_device_variance:
            self.settings.parameter_estimator.model_variance = False
    
    def fix_parameter(self, param_to_fix):
        
        if not hasattr(self, 'fixed_params'):
            self.fixed_params = []
        
        if isinstance(param_to_fix, str):
            param_to_fix = [param_to_fix]
            
        self.fixed_params += [p for p in param_to_fix]
    
    def run_opt(self):
        """Run ParameterEstimator but checking for variances - this should
        remove the VarianceEstimator being required to be implemented by the user
        
        """
        if self.model is None:    
            self.create_pyomo_model()  
        
        if not self.allow_optimization:
            raise ValueError('The model is incomplete for parameter optimization')
            
        # Some settings are required together, this method checks this
        self._update_related_settings()
        
        # Check if all component variances are given; if not run VarianceEstimator
        has_spectral_data = 'spectral' in [d.category for d in self.datasets]
        has_all_variances = self.components.has_all_variances
        variances_with_delta = None
        
        if self.settings.variance_estimator.method == 'direct_sigmas':
            raise ValueError('This variance method is not intended for use in the manner: see Ex_13_direct_sigma_variances.py')
        
        if not has_all_variances and has_spectral_data:
            """If the data is spectral and not all variances are provided, VE needs to be run"""
            
            self.create_estimator(estimator='v_estimator', **self.settings.collocation)
            settings_run_ve_opt = self.settings.variance_estimator
            
            if self.settings.variance_estimator.max_device_variance:
                max_device_variance = self.v_estimator.solve_max_device_variance(**settings_run_ve_opt)
            
            # elif self.settings.variance_estimator.use_delta:
            #     variances_with_delta = self.solve_variance_given_delta()

            else:
                self.run_ve_opt(**settings_run_ve_opt)
                
        elif not has_all_variances and not has_spectral_data:
            for comp in self.components:
                try:
                    comp.variance = self.variances[comp.name]
                except:
                    print(f'No variance information for {comp.name} found, setting equal to unity')
                    comp.variance = 1
                
        # Create ParameterEstimator
        self.create_estimator(estimator='p_estimator', **self.settings.collocation)
        #if self.settings.parameter_estimator.G_contribution is not None:
            #self._unwanted_G_initialization(self.p_model)
        variances = self.components.variances
        self.variances = variances
        
        # If variance calculated using VarianceEstimator, initialize PE isntance
        if 'v_estimator' in self.results_dict:
            if self.settings.general['initialize_pe']:
                self.initialize_from_variance_trajectory()
            if self.settings.general['scale_pe']:
                self.scale_variables_from_variance_trajectory()
            self.variances = self.results_dict['v_estimator'].sigma_sq
        
        elif self.settings.variance_estimator.max_device_variance:
            self.variances = max_device_variance
        
        # elif variances_with_delta is not None: 
        #     variances = variances_with_delta
            
        if self.settings.general['scale_variances']:
            self.variances = self._scale_variances(variances)
        
        settings_run_pe_opt = self.settings.parameter_estimator
        settings_run_pe_opt['solver_opts'] = self.settings.solver
        settings_run_pe_opt['variances'] = self.variances
        
        self.run_pe_opt(**settings_run_pe_opt)
        self.results = self.results_dict['p_estimator']
        self.results.file_dir = self.settings.general.charts_directory
        
        return self.results
    
    @staticmethod
    def _scale_variances(variances):
        
        max_var = max(variances.values())
        scaled_vars = {comp: var/max_var for comp, var in variances.items()}
        return scaled_vars

    def _run_opt(self, estimator, *args, **kwargs):
        """Runs the respective optimization for the estimator"""
        
        if not hasattr(self, estimator):
            raise AttributeError(f'ReactionModel has no attribute {estimator}')
            
        self.results_dict[estimator] = getattr(self, estimator).run_opt(*args, **kwargs)
        return self.results_dict[estimator]
    
    def initialize_from_variance_trajectory(self, variable=None, obj='p_estimator'):
        """Wrapper for the initialize_from_trajectory method in
        ParameterEstimator
        
        """
        source = self.results_dict['v_estimator']
        self._from_trajectory('initialize', variable, source, obj)
        return None
    
    def initialize_from_trajectory(self, variable_name=None, source=None):
        """Wrapper for the initialize_from_trajectory method in
        ParameterEstimator or PyomoSimulator
        
        """
        self._var_to_initialize_from_trajectory.append([variable_name, source])
        return None
    
    def scale_variables_from_variance_trajectory(self, variable=None, obj='p_estimator'):
        """Wrapper for the scale_varialbes_from_trajectory method in
        ParameterEstimator
        
        """
        source = self.results_dict['v_estimator']
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
    
    def fix_from_trajectory(self, variable_name, variable_index, trajectories):
        """Wrapper for fix_from_trajectory in PyomoSimulator. This stores the
        information and then fixes the data after the simulator or estimator
        has been declared
        
        """
        self._var_to_fix_from_trajectory.append([variable_name, variable_index, trajectories])
        return None
                                               
    def set_known_absorbing_species(self, *args, **kwargs):
        """Wrapper for set_known_absorbing_species in TemplateBuilder
        
        """
        self.builder.set_known_absorbing_species(*args, **kwargs)    
        return None
    
    def scale(self):
        """Scale the model"""
        
        parameter_dict = self.parameters.as_dict(bounds=False)    
        scaled_parameter_dict, scaled_models_dict = scale_models(self.model,
                                                                 parameter_dict,
                                                                 name=self.name,
                                                                 )         
        return scaled_parameter_dict, scaled_models_dict
    
    def reduce_model(self, **kwargs):
        """This calls the reduce_models method in the EstimationPotential
        module to reduce the model based on the reduced hessian parameter
        selection method.
        
        Args:
            kwargs:
                replace (bool): defaults to True, option to replace the
                    parameters deemed unestimable from the model with constants
                no_scaling (bool): defaults to True, removes the scaling
                    constants from the model and restores the parameter values
                    and their bounds.
                    
        Returns:
            results (ResultsObject): A standard results object with the reduced
                model results
        
        """
        if self.model is None:
            self.create_pyomo_model()
            
        # settings_rhps['solver_opts'] = self.settings.solver
        kwargs['solver_opts'] = self.settings.solver
        
        parameter_dict = self.parameters.as_dict(bounds=True)
        results, reduced_model = reduce_model(self.model, **kwargs)
        
        results.file_dir = self.settings.general.charts_directory
        
        self.reduced_model = reduced_model
        self.using_reduced_model = True
        self.reduced_model_results = results
        
        return results
    
    # def reduce_model_old(self, **kwargs):
    #     """This calls the reduce_models method in the EstimationPotential
    #     module to reduce the model based on the reduced hessian parameter
    #     selection method.
        
    #     Args:
    #         kwargs:
    #             replace (bool): defaults to True, option to replace the
    #                 parameters deemed unestimable from the model with constants
    #             no_scaling (bool): defaults to True, removes the scaling
    #                 constants from the model and restores the parameter values
    #                 and their bounds.
                    
    #     Returns:
    #         results (ResultsObject): A standard results object with the reduced
    #             model results
        
    #     """
    #     if self.model is None:
    #         self.create_pyomo_model()
        
    #     parameter_dict = self.parameters.as_dict(bounds=True)
        
    #     kwargs['times'] = (self.model.start_time.value, self.model.end_time.value)
        
    #     print(kwargs)
        
    #     reduce_model_old(self, **kwargs)
        
    #     # self.reduced_model = reduced_model
    #     # self.using_reduced_model = True
    #     # self.reduced_model_results = results
        
    #     return None #results
    
    def set_non_absorbing_species(self, non_abs_list):
        """Wrapper for set_non_absorbing_species in TemplateBuilder"""
        
        self._has_non_absorbing_species = True
        self.non_abs_list = non_abs_list
        return None
        
    def add_noise_to_data(self, var, noise, overwrite=False):
        """Wrapper for adding noise to data after data has been added to
        the specific ReactionModel
        
        """
        dataframe = self.datasets[var].data
        if overwrite:
            self.datasets[var].data = dataframe
        return data_tools.add_noise_to_signal(dataframe, noise)    
    
    # def apply_pe_discretization(self, model_object, *args, **kwargs):
    #     """Checks is the model is discretized and discretizes it in the case
    #     that it is not
        
    #     Args:
    #         model (ConcreteModel): A pyomo ConcreteModel
            
    #         ncp (int): number of collocation points used
            
    #         nfe (int): number of finite elements used
            
    #     Returns:
    #         None
            
    #     """
    #     method = kwargs.pop('method', 'dae.collocation')
    #     ncp = kwargs.pop('ncp', 3)
    #     nfe = kwargs.pop('nfe', 50)
    #     scheme = kwargs.pop('scheme', 'LAGRANGE-RADAU')
        
    #     if not model_object.alltime.get_discretization_info():
        
    #         # You need to change this out of an Estimator
    #         model_pe = ParameterEstimator(model_object)
    #         model_pe.apply_discretization(method,
    #                                       ncp=ncp,
    #                                       nfe=nfe,
    #                                       scheme=scheme)
        
    #     return None
        
    # def rule_objective(self, model):
    #     """This function defines the objective function for the estimability
        
    #     This is equation 5 from Chen and Biegler 2020. It has the following
    #     form:
            
    #     .. math::
    #         \min J = \frac{1}{2}(\mathbf{w}_m - \mathbf{w})^T V_{\mathbf{w}}^{-1}(\mathbf{w}_m - \mathbf{w})
            
    #     Originally KIPET was designed to only consider concentration data in
    #     the estimability, but this version now includes complementary states
    #     such as reactor and cooling temperatures. If complementary state data
    #     is included in the model, it is detected and included in the objective
    #     function.
        
    #     Args:
    #         model (pyomo.core.base.PyomoModel.ConcreteModel): This is the pyomo
    #         model instance for the estimability problem.
                
    #     Returns:
    #         obj (pyomo.environ.Objective): This returns the objective function
    #         for the estimability optimization.
        
    #     """
    #     obj = 0
        
    #     from pyomo.environ import Objective
        
    #     print(model.sigma)
    
    #     for k in set(model.mixture_components.value_list) & set(model.measured_data.value_list):
    #         for t, v in model.Cm.items():
    #             obj += 0.5*(model.Cm[t] - model.Z[t]) ** 2 /  1#model.sigma[k]**2
        
    #     for k in set(model.complementary_states.value_list) & set(model.measured_data.value_list):
    #         for t, v in model.U.items():
    #             obj += 0.5*(model.X[t] - model.U[t]) ** 2 / 1#model.sigma[k]**2      
    
    #     model.objective = Objective(expr=obj)
    
    #     return None
    
def _set_directory(model_object, filename, abs_dir=False):
    """Wrapper for the set_directory method. This replaces the awkward way
    of ensuring the correct directory for the data is used.
    
    Args:
        filename (str): the file name to be formatted
        
    Returns:
        file_path (pathlib Path): The absolute path of the given file
    """
    directory = model_object.settings.general.data_directory
    file_path = pathlib.Path(directory).joinpath(filename)
    
    return file_path
        