"""
Primary means of using KIPET is through the KipetModel class

@author: kevin 2020
"""
# Standard library imports
import collections
import pathlib

# Third party imports
import pandas as pd

# Kipet library imports
from kipet.library.common.pre_process_tools import decrease_wavelengths
import kipet.library.core_methods.data_tools as data_tools
from kipet.library.core_methods.MEE_3 import MultipleExperimentsEstimator
from kipet.library.nsd_funs.NSD_KIPET import NSD
from kipet.library.top_level.reaction_model import (
    ReactionModel,
    _set_directory,
    )
from kipet.library.top_level.settings import (
    Settings, 
    USER_DEFINED_SETTINGS,
    )

class KipetModel():
    
    """Primary KIPET object that holds ReactionModel instances"""
    
    def __init__(self):
        
        self.models = {}
        self.settings = Settings(category='block')
        self.results = {}
        self.global_parameters = None
        self.method = 'mee'
         
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
        self.mee.confidence_interval = self.settings.parameter_estimator.confidence
        
        if 'spectral' in self.data_types:
            self.settings.parameter_estimator.spectra_problem = True
        else:
            self.settings.parameter_estimator.spectra_problem = False    
        
        self.mee.spectra_problem = self.settings.parameter_estimator.spectra_problem
        
    def _calculate_parameters(self):
        """Uses the ReactionModel framework to calculate parameters instead of 
        repeating this in the MEE
        
        """
        for name, model in self.models.items():
            if not model.optimized:
                for dataset in model.datasets:
                    if self.settings.general.freq_wavelength_subset is not None:
                        if model.datasets[dataset.name].category == 'spectral':
                            freq = self.settings.general.freq_wavelength_subset
                            model.datasets[dataset.name].data = decrease_wavelengths(dataset.data, freq)
                
                model.run_opt()
            else:
                print(f'Model {name} has already been optimized')
                
        return None
    
    def run_full_model(self):
        
        results = self.mee.solve_consolidated_model(self.global_parameters,
                                                    **self.settings.parameter_estimator)
    
        self.results = results
        for key, results_obj in self.results.items():
            results_obj.file_dir = self.settings.general.charts_directory
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
    
    # @property
    # def global_params(self):
        
    #     parameter_counter = collections.Counter()
    #     global_params = set()
    #     for name, model in self.models.items():
    #         for param in model.parameters.names:
    #             parameter_counter[param] += 1
        
    #     for param, num in parameter_counter.items():
    #         if num > 1:
    #             global_params.add(param)
            
    #     return global_params
    
    # @property
    # def all_wavelengths(self):
    #     set_of_all_wavelengths = set()
    #     for name, model in self.models.items():
    #         set_of_all_wavelengths = set_of_all_wavelengths.union(list(model.model.meas_lambdas))
            
    #     return set_of_all_wavelengths
    
    # @property
    # def all_species(self):
    #     set_of_all_species = set()
    #     for name, model in self.models.items():
    #         set_of_all_species = set_of_all_species.union(model.components.names)
            
    #     return set_of_all_species
    
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