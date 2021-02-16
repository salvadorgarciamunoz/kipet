"""
Primary means of using KIPET is through the KipetModel class

@author: kevin 2020
"""
# Standard library imports
import collections
import os
import pathlib
import sys

# Third party imports
import pandas as pd

# Kipet library imports
import kipet.core_methods.data_tools as data_tools
from kipet.core_methods.MEE import MultipleExperimentsEstimator
from kipet.nsd_funs.NSD_KIPET import NSD
from kipet.top_level.reaction_model import (
    ReactionModel,
    _set_directory,
    )
from kipet.top_level.settings import (
    Settings, 
    )
from kipet.top_level.unit_base import UnitBase

# from kipet.top_level.element_blocks import (
#     AlgebraicBlock,
#     ComponentBlock,
#     ConstantBlock, 
#     ParameterBlock, 
#     StateBlock,
#     )
from kipet.core_methods.TemplateBuilder import TemplateBuilder

class KipetModel():
    
    """Primary KIPET object that holds ReactionModel instances"""
    
    def __init__(self):
        
        self.models = {}
        self.settings = Settings()
        self.results = {}
        self.global_parameters = None
        self.method = 'mee'
        
        self.use_individual_settings = False
        # ub = UnitBase()
        self.ub = UnitBase()
        self.reset_base_units()    
    
        # self.components = ComponentBlock()   
        # self.parameters = ParameterBlock()
        # self.constants = ConstantBlock()
        # self.algebraics = AlgebraicBlock()
        # self.states = StateBlock()
    
        
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
            
    def __getitem__(self, value):
        return self.models[value]
            
    def __len__(self):
        return len(self.models)

    def reset_base_units(self):
        
        self.ub.TIME_BASE = self.settings.units.time
        self.ub.VOLUME_BASE = self.settings.units.volume

    def add_model_list(self, model_list):
        """Handles lists of parameters or single parameters added to the model
        """
        for model in model_list:
            self.add_model(model)         
        
        return None
    
    def add_model(self, model):
        """Redundant"""
        self.add_reaction(model)  
    #     if isinstance(model, ReactionModel):
    #         self.models[model.name] = model
    #     else:
    #         raise ValueError('KipetModel can only add ReactionModel instances.')
            
    #     return None
    
    def remove_model(self, model):
        
        if isinstance(model, str):
            if model in self.models:
                self.models.pop(model)
        elif isinstance(model, ReactionModel):
            self.models.pop(model.name)
        else:
            print('KipetModel does not have specified model')
        return None    
    
    def new_reaction(self, name, model=None, ignore=None):
        
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
        ignore = ignore if ignore is not None else []
        
        if model is None:
        
            self.models[name] = ReactionModel(name=name, unit_base=self.ub)
#            self.models[name].settings.general.data_directory = self.settings.general.data_directory
            
            assign_list = ['components', 'parameters', 'constants', 'algebraics',
                             'states', 'ub', 'c']
        
            for item in assign_list:
                if item not in ignore and hasattr(self, item):
                    setattr(self.models[name], item, getattr(self, item))
        
        else:
            if isinstance(model, ReactionModel):
                kwargs = {}
                kwargs['name'] = name
                self.models[name] = ReactionModel(name=name, unit_base=self.ub)
                
                assign_list = ['components', 'parameters', 'constants', 'algebraics',
                             'states', 'ub', 'settings', 'c', 'odes_dict']
        
              #  copy_list = []#'c', 'odes_dict']# 'set_up_model']
                for item in assign_list:
                    if item not in ignore and hasattr(model, item):
                        setattr(self.models[name], item, getattr(model, item))
                
                # import copy
                # for item in copy_list:
                #     if item not in ignore and hasattr(model, item):
                #         setattr(self.models[name], item, copy.copy(getattr(model, item)))
                
            else:
                raise ValueError('KipetModel can only add ReactionModel instances.')
        
        return self.models[name]
    
    def add_reaction(self, kipet_model_instance):
    
        if isinstance(kipet_model_instance, ReactionModel):
            kipet_model_instance.unit_base = self.ub
            self.models[kipet_model_instance.name] = kipet_model_instance
            #self.models[kipet_model_instance.name].settings.general.data_directory = self.settings.general.data_directory
        else:
            raise ValueError('KipetModel can only add ReactionModel instances.')

        return None
    
    # def constant(self, *args, **kwargs):
        
    #     kwargs['unit_base'] = self.ub
    #     self.constants.add_element(*args, **kwargs)
    #     return None
        
    # def algebraic(self, *args, **kwargs):
        
    #     kwargs['unit_base'] = self.ub
    #     self.algebraics.add_element(*args, **kwargs)
    #     # if 'step' in kwargs and kwargs['step'] is not None:
    #     #     self.add_step(f's_{args[0]}', time=15, switch='off')
    
    # # def add_step(self, name, *args, **kwargs):
        
    # #     self._has_step_or_dosing = True
    # #     if not hasattr(self, '_step_list'):
    # #         self._step_list = {}
            
    # #     if name not in self._step_list:
    # #         self._step_list[name] = [kwargs]
    # #     else:
    # #         self._step_list[name].append(kwargs)
    
    # def state(self, *args, **kwargs):
    #     """Add the components to the Kipet instance
        
    #     Args:
    #         components (list): list of Component instances
            
    #     Returns:
    #         None
            
    #     """
    #     kwargs['unit_base'] = self.ub
    #     self.states.add_element(*args, **kwargs)
    #     return None
    
    # def component(self, *args, **kwargs):
    #     """Add the components to the Kipet instance
        
    #     Args:
    #         components (list): list of Component instances
            
    #     Returns:
    #         None
            
    #     """
    #     kwargs['unit_base'] = self.ub
    #     self.components.add_element(*args, **kwargs)
    #     return None
    
    # def parameter(self, *args, **kwargs):
    #     """Add the parameters to the Kipet instance
        
    #     Args:
    #         parameters (list): list of Parameter instances
            
    #         factor (float): defaults to 1, the scalar multiple of the parameters
    #         for simulation purposes
            
    #     Returns:
    #         None
            
    #     """
    #     kwargs['unit_base'] = self.ub
    #     self.parameters.add_element(*args, **kwargs)
    #     return None
    
    # def get_model_vars(self):
        
    #     self.create_pyomo_model_vars()
    #     return self.c
    
    # # def load_vars(self):
        
    # #     for var in self.c.get_var_list:
    # #         globals()[var] = getattr(self.c, var)
    
    # def create_pyomo_model_vars(self, *args, **kwargs):
        
    #     self.builder = TemplateBuilder()
    #     setattr(self.builder, 'early_return', True)
            
    #     if len(self.states) > 0:
    #         self.builder.add_model_element(self.states)
        
    #     if len(self.algebraics) > 0:
    #         self.builder.add_model_element(self.algebraics)
        
    #     if len(self.components) > 0:
    #         self.builder.add_model_element(self.components)
            
    #     if len(self.parameters) > 0:
    #         self.builder.add_model_element(self.parameters)
        
    #     if len(self.constants) > 0:
    #         self.builder.add_model_element(self.constants)
        
    #     if hasattr(self, 'c'):
    #         setattr(self.builder, 'c_mod', self.c)
        
    #     self.model = self.builder.create_pyomo_model(0, 1)
    #     self.c = self.builder.c_mod
    #     #self.set_up_model = self.model
    #     self.model = None
            
    #     return None
    
            
    def run_opt(self, *args, **kwargs):
        """Solve a single model or solve multiple models using the MEE
        """
        method = kwargs.get('method', 'mee')
        
        if len(self.models) > 1:
            if method == 'mee':
                self._calculate_parameters()
                self._create_multiple_experiments_estimator(*args, **kwargs)
                self.run_full_model()
            elif method == 'nsd':
                self._calculate_parameters()
                self.mee_nsd(strategy='ipopt')
            else:
                raise ValueError('Not a valid method for optimization')
            
        else:
            reaction_model = self.models[list(self.models.keys())[0]]
            results = reaction_model.run_opt()
            self.results[reaction_model.name] = results
            
        return None
    
    def _create_multiple_experiments_estimator(self, *args, **kwargs):
        """A quick wrapper for MEE without big changes
        
        """
        self.mee = MultipleExperimentsEstimator(self.models)
        self.mee.confidence_interval = self.settings.general.confidence
        
        if 'spectral' in self.data_types:
            self.settings.general.spectra_problem = True
        else:
            self.settings.general.spectra_problem = False    
        
        self.mee.spectra_problem = self.settings.general.spectra_problem
        
    def _calculate_parameters(self):
        """Uses the ReactionModel framework to calculate parameters instead of 
        repeating this in the MEE
        
        """
        for name, model in self.models.items():
            if not model.optimized:
                # Do this for each model separately...
                # for dataset in model.datasets:
                #     if self.settings.general.freq_wavelength_subset is not None:
                #         if model.datasets[dataset.name].category == 'spectral':
                #             freq = self.settings.general.freq_wavelength_subset
                #             model.datasets[dataset.name].data = decrease_wavelengths(dataset.data, freq)
                
                # if not self.use_individual_settings:
                    
                #     model.settings.update(**self.settings)
                
                model.run_opt()
            else:
                print(f'Model {name} has already been optimized')
                
        return None
    
    def run_full_model(self):
        
        # self.settings.parameter_estimator.solver_opts = self.settings.solver
        consolidated_settings = self.settings.general
        consolidated_settings.solver_opts = self.settings.solver
        
        results = self.mee.solve_consolidated_model(self.global_parameters,
                                                    **consolidated_settings)
    
        self.results = results
        for key, results_obj in self.results.items():
#            results_obj.file_dir = self.settings.general.charts_directory
            self.models[key].results = results[key]
            
        return results
    
    def mee_nsd(self, strategy='ipopt'):
        """Performs the NSD on the multiple datasets
        
        Args:
            strategy (str): Method used to control the outer problem
                ipopt, trust-region, newton-step
                
        Returns:
            results
        
        """        
        kwargs = {'kipet': True,
                  'objective_multiplier': 1
                  }
        
        if self.global_parameters is not None:
            global_parameters = self.global_parameters
        else:
            global_parameters = self.all_params
        
        self.nsd = NSD(self.models,
                       strategy=strategy,
                       global_parameters=global_parameters, 
                       kwargs=kwargs)
        
        print(self.nsd.d_init)
        
        results = self.nsd.run_opt()
        
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
        # if directory is None:
        #     _filename = _set_directory(self, filename)
        # else:
        #     _filename = pathlib.Path(directory).joinpath(filename)
        # if not _filename.parent.is_dir():
        #     _filename.parent.mkdir(exist_ok=True)
        calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
        _filename = pathlib.Path(calling_file_name).joinpath(_filename)
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
        # if directory is None:
        #     _filename = _set_directory(self, filename)
        # else:
        #     _filename = pathlib.Path(directory).joinpath(filename)
        
        calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
        _filename = pathlib.Path(calling_file_name).joinpath(_filename)
        read_data = data_tools.read_file(_filename)
        
        return read_data
    
    def plot(self, *args, **kwargs):
        """Plots the modeled concentration profiles"""
        
        for reaction, result in self.results.items():
            description={'title': f'Experiment: {reaction}',
                                  'xaxis': 'Time [s]',
                                  'yaxis': 'Concentration [mol/L]'}
        
            result.plot('Z', description=description)
            
        return None
    
    @staticmethod
    def add_noise_to_data(data, noise):
        """Wrapper for adding noise to data after data has been added to
        the specific ReactionModel
        
        """
        return data_tools.add_noise_to_signal(data, noise)   
    
    # def delete_file(self, filename, directory=None):
    #     """Method to remove files from the directory"""
    #     remove_file(self.settings.general.data_directory, 'ipopt_opt')
    
    @property
    def all_params(self):
        set_of_all_model_params = set()
        for name, model in self.models.items():
            set_of_all_model_params = set_of_all_model_params.union(model.parameters.names)
            
        return set_of_all_model_params
   
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
        
        df_param = pd.DataFrame(data=None, index=self.all_params, columns=self.models.keys())
            
        for reaction, model in self.models.items():
            for param in model.parameters.names:
                df_param.loc[param, reaction] = model.results.P[param]
            
        return df_param
           
    @property
    def get_p_models(self):
        
        return [model.p_model for model in self.models.values()]
    
    
    # Testing placing components into the KM
    
    
    
    
    
    
    