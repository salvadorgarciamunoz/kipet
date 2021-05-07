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
from kipet.top_level.reaction_model import ReactionModel
from kipet.top_level.settings import Settings
from kipet.top_level.unit_base import UnitBase


class KipetModel:
    
    """The highest level object in KIPET. All uses of KIPET start with the creation
    of a KipetModel instance.
    
        
    :var dict reaction_models: The dictionary containing ReactionModel instances
    :var Settings settings: A Settings instance holding all modeling options
    :var dict results: A dictionary of ResultsObjects for each ReactionModel instance
    :var list global_parameters: A list of global parameters
    :var UnitBase ub: UnitBase object used to define the base units of the project
    
    :Methods:
    
    - :func:`add_reaction`
    - :func:`add_reaction_list`
    - :func:`remove_reaction`
    - :func:`new_reaction`
    - :func:`run_opt`
    - :func:`read_data_file`
    - :func:`write_data_file`
    - :func:`add_noise_to_data`
    
    :Properties:
    
    - :func:`all_params`
    
    :Example:
        >>> import kipet
        >>> kipet_model = kipet.KipetModel()
    
    """
    def __init__(self):
        """Initialize the KipetModel instance.
        
        The KipetModel object does not require any attributes at initialization.
        
        """
        self.reaction_models = {}
        self.settings = Settings()
        self.results = {}
        self.global_parameters = None
        self.ub = UnitBase()
        
        self.reset_base_units()

    def __str__(self):
        
        block_str = "KipetModel\n\n"

        for name, model in self.reaction_models.items():
            block_str += f"{name}\tDatasets: {len(model.datasets)}\n"

        return block_str

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, value):
        return self.reaction_models[value]

    def __len__(self):
        return len(self.reaction_models)

    def reset_base_units(self):
        """Sets the unit base values to the units declared in the settings
        
        :return: None
        """
        self.ub.TIME_BASE = self.settings.units.time
        self.ub.VOLUME_BASE = self.settings.units.volume
        
        return None


    def add_reaction_list(self, model_list):
        """Method to add a list of ReactionModel instances to the KipetModel
        
        :parameter list model_list: A list of ReactionModel instances
        
        :return: None
        """
        
        for model in model_list:
            self.add_reaction(model)

        return None


    def remove_reaction(self, model):
        """Remove a model instance from the KipetModel model list
        
        :parameter str/ReactionModel model: The name or instance of the reaction model to be removed
    
        :return: None
        """
        if isinstance(model, str):
            if model in self.reaction_models:
                self.reaction_models.pop(model)
        elif isinstance(model, ReactionModel):
            self.reaction_models.pop(model.name)
        else:
            print("KipetModel does not have specified model")
        return None


    def new_reaction(self, name, model=None, ignore=None):

        """Declare new reactions to the KipetModel using this function

        :parameter str name: The name of the model/experiment used in all references
                         made to it in KIPET, especially in python dicts
        :parameter ReactionModel model: Existing ReactionModel to initialize new ReactionModel with
        :parameter list[str] ignore: This is a list of strings for the various
                attributes in the ReactionModel that should not be copied to the new ReactionModel
                
        - **For example**, if you have a ReactionModel instance (r1) and want to create 
          a new instance (r2) with the same settings except for the data, you could use
          the following::

            r2 = kipet_model.new_reaction('r2', model=r1, ignore=['datasets'])


        :return: A new ReactionModel instance
            
        """
        ignore = ignore if ignore is not None else []

        if model is None:

            self.reaction_models[name] = ReactionModel(name=name, unit_base=self.ub)
            
            assign_list = [
                "components",
                "parameters",
                "constants",
                "algebraics",
                "states",
                "ub",
                "c",
            ]

            for item in assign_list:
                if item not in ignore and hasattr(self, item):
                    setattr(self.reaction_models[name], item, getattr(self, item))

        else:
            if isinstance(model, ReactionModel):
                kwargs = {}
                kwargs["name"] = name
                self.reaction_models[name] = ReactionModel(name=name, unit_base=self.ub)

                assign_list = [
                    "components",
                    "parameters",
                    "constants",
                    "algebraics",
                    "states",
                    "ub",
                    "settings",
                    "c",
                    "odes_dict",
                    "algs_dict",
                ]

                for item in assign_list:
                    if item not in ignore and hasattr(model, item):
                        setattr(self.reaction_models[name], item, getattr(model, item))

            else:
                raise ValueError("KipetModel can only add ReactionModel instances.")

        return self.reaction_models[name]
    

    def add_reaction(self, model):
        """Adds a ReactionModel instance to the KipetModel instance
        
        :parameter ReactionModel model: ReactionModel instance
          
        :return: None
        """
        if isinstance(model, ReactionModel):
            model.unit_base = self.ub
            self.reaction_models[model.name] = model
        else:
            raise ValueError("KipetModel can only add ReactionModel instances.")

        return None
    

    def run_opt(self, method='mee'):
        """This method will perform parameter fitting for all ReactionModels in
        the KipetModel models attribute. If more than one ReactionModel instance
        is present, the MultipleExperimentEstiamtor is used to solve for the
        global parameter set.
        
        :parameter str method: Define the method used to solve the multiple
          dataset problem: either **mee** (solves one combined problem simultaneously)
          or **nsd** which solves the problem using a Nested Schur decomposition approach.
        
        .. warning::

            The NSD method is not implemented at this time!
        
        :return: None
        """
        
        if hasattr(self, "mee"):
            del self.mee

        if len(self.reaction_models) > 1:
            if method == "mee":
                self._calculate_parameters()
                self._create_multiple_experiments_estimator()
                self._run_full_model()
            # elif method == 'nsd':
            #     self._calculate_parameters()
            #     self._mee_nsd(strategy='ipopt')
            else:
                raise ValueError("Not a valid method for optimization")

        else:
            reaction_model = self.reaction_models[list(self.reaction_models.keys())[0]]
            results = reaction_model.run_opt()
            self.results[reaction_model.name] = results

        return None

    def _create_multiple_experiments_estimator(self):
        """A quick wrapper for MEE without big changes
        
        """
        
        self.mee = MultipleExperimentsEstimator(self.reaction_models)
        self.mee.confidence_interval = self.settings.general.confidence

        if "spectral" in self.data_types:
            self.settings.general.spectra_problem = True
        else:
            self.settings.general.spectra_problem = False

        self.mee.spectra_problem = self.settings.general.spectra_problem

    def _calculate_parameters(self):
        """Uses the ReactionModel framework to calculate parameters instead of
        repeating this in the MEE

        """
        for name, model in self.reaction_models.items():
            if not model.optimized:
                model.run_opt()
            else:
                print(f"Model {name} has already been optimized")

        return None

    def _run_full_model(self):
        """Runs the MEE after everything has been set up"""
        
        consolidated_settings = self.settings.general
        consolidated_settings.solver_opts = self.settings.solver

        results = self.mee.solve_consolidated_model(
            self.global_parameters, **consolidated_settings
        )

        self.results = results
        for key, results_obj in self.results.items():
            self.reaction_models[key].results = results[key]

        return results

    def _mee_nsd(self, strategy='ipopt'):
        """Performs the NSD on the multiple ReactionModel instances

        :parameter str strategy: Method used to control the outer problem
                ipopt, trust-region, newton-step

        :return results: A dictionary of ResultsObject instances
        :rtype: dict

        """
        kwargs = {'kipet': True,
                  'objective_multiplier': 1
                  }

        if self.global_parameters is not None:
            global_parameters = self.global_parameters
        else:
            global_parameters = self.all_params

        self.nsd = NSD(self.reaction_models,
                        strategy=strategy,
                        global_parameters=global_parameters,
                        kwargs=kwargs)

        #print(self.nsd.d_init)

        results = self.nsd.run_opt()

        self.results = results
        for key, results_obj in self.results.items():
            results_obj.file_dir = self.settings.general.charts_directory

        return results


    @staticmethod
    def write_data_file(filename, data):
        """Method to write data to a file using KipetModel

        Convenient method to save modified data in a format ready to use with ReactionModels.

        :parameter str filename: The name of the file (plus relative directory)

        :parameter pandas.DataFrame data: The pandas DataFrame to be written to the file

        :returns: None
        
        """
        _filename = filename
        calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
        _filename = pathlib.Path(calling_file_name).joinpath(_filename)
        data_tools.write_file(_filename, data)
        
        return None

    @staticmethod
    def read_data_file(filename):
        """Method to read data file using KipetModel
        
        This is useful if you need to modify a datafile before using it with a ReactionModel.

        :parameter str filename: The name of the data file (expected to be in the data
           directory, otherwise use an absolute path).

        :return: The data read from the file
        :rtype: pandas.DataFrame
        """
        _filename = filename
        calling_file_name = os.path.dirname(os.path.realpath(sys.argv[0]))
        _filename = pathlib.Path(calling_file_name).joinpath(_filename)
        read_data = data_tools.read_file(_filename)
        
        return read_data


    @staticmethod
    def add_noise_to_data(data, noise):
        """Wrapper for adding noise to data after data has been added to
        the specific ReactionModel
        
        :parameter pandas.DataFrame data: The dataset to which noise is to be added.
        
        :parameter float noise: The variance of the added noise.
        
        :return: The dataset after noised has been added.
        :rtype: pandas.DataFrame

        """
        noised_data = data_tools.add_noise_to_signal(data, noise)
        
        return noised_data 
    

    @property
    def all_params(self):
        """
        Returns the set of all parameters across all ReactionModel instances.
        
        :return: The set of all unique parameters in all models.
        :rtype: set
        """
        
        set_of_all_model_params = set()
        for name, model in self.reaction_models.items():
            set_of_all_model_params = set_of_all_model_params.union(
                model.parameters.names
            )

        return set_of_all_model_params
