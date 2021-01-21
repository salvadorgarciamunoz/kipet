"""
This is a wrapper for kipet so that users can more easily access the code
without requiring a plethora of imports

Should block be the standard? This could check the number of models in the set
and perform the calculations as singles or MEE

@author: kevin 2020
"""
# Standard library imports
import collections
import pathlib

# Third party imports

# Kipet library imports
from kipet.library.top_level.reaction_model import ReactionModel, _set_directory

import kipet.library.core_methods.data_tools as data_tools
from kipet.library.core_methods.MEE_new import MultipleExperimentsEstimator
from kipet.library.common.pre_process_tools import decrease_wavelengths
from kipet.library.top_level.settings import Settings, USER_DEFINED_SETTINGS

from kipet.library.nsd_funs.NSD_TrustRegion_Ipopt import NSD

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
            
            if not model.optimized:
                 
                settings_run_pe_opt = model.settings.parameter_estimator
                settings_run_pe_opt['solver_opts'] = model.settings.solver
                settings_run_pe_opt['variances'] = self.results_variances[model.name]
                settings_run_pe_opt['confindence_interval'] = self.settings.parameter_estimator.confidence
                model.create_parameter_estimator(**self.settings.collocation)
                model.run_pe_opt(**settings_run_pe_opt)
            
            else:
                print('Model has already been optimized')
            
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
    
    @property
    def get_p_models(self):
        
        return [model.p_model for model in self.models.values()]
    
    def mee_nsd(self, strategy='ipopt'):

        kwargs = {'kipet': True,
                  'objective_multiplier': 1
                  }
        
        # Choose the method used to optimize the outer problem
        #strategy = 'ipopt'
        #strategy = 'newton-step'
        #strategy = 'trust-region'
        models = self.get_p_models
        print(models)
        
        nsd = NSD(self.models.values(), kwargs=kwargs)
        
        print(nsd.d_init)
        
        if strategy == 'ipopt':
            # Runs the IPOPT Method
            results = nsd.ipopt_method(scaled=False)
        
        elif strategy == 'trust-region':
            # Runs the Trust-Region Method
            results = nsd.trust_region(scaled=False)
            # Plot the parameter value paths (Trust-Region only)
            nsd.plot_paths()
            
        elif strategy == 'newton-step':
            # Runs the NSD using simple newton steps
            nsd.run_simple_newton_step(alpha=0.1, iterations=15)  
        
        # Plot the results using ReactionModel format
        nsd.plot_results()
        
        return results