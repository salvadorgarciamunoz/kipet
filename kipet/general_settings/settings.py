"""
Settings for KIPET
"""
# Standard library imports
import ast
from pathlib import Path

# Third party imports
import yaml

# Kipet library imports
from kipet.calculation_tools.helper import AttrDict


class Settings:
    """This is a container for all of the options that can be used in Kipet
    Since it can be confusing due to the large number of options, this should
    make it easier for the user to see everything in one spot.
    
    This class loads the default settings from the settings.yml file located
    in the directory wherever KIPET has been installed. If you really want
    to change this file, go ahead, but make sure that you save a copy of the
    original settings in case something goes wrong.
    
    :Methods:
    
    - :func:`updated_settings`
    
    """

    def __init__(self, category='model'):
        """Settings object initialization that begins each time a
        ReactionModel instance is created.
        
        """
        self._load_settings()
        self._reset_model()

    def __str__(self):

        m = 25

        settings = 'Settings\n\n'

        if hasattr(self, 'general'):
            settings += 'General Settings (general):\n'
            for k, v in self.general.items():
                settings += f'{str(k).rjust(m)} : {v}\n'

        if hasattr(self, 'units'):
            settings += '\nUnit Settings (units):\n'
            for k, v in self.units.items():
                settings += f'{str(k).rjust(m)} : {v}\n'

        if hasattr(self, 'collocation'):
            settings += '\nCollocation Settings (collocation):\n'
            for k, v in self.collocation.items():
                settings += f'{str(k).rjust(m)} : {v}\n'

        if hasattr(self, 'simulator'):
            settings += '\nSimulation Settings (simulator):\n'
            for k, v in self.simulator.items():
                settings += f'{str(k).rjust(m)} : {v}\n'

        if hasattr(self, 'variance_estimator'):
            settings += '\nVarianceEstimator Settings (variance_estimator):\n'
            for k, v in self.variance_estimator.items():
                settings += f'{str(k).rjust(m)} : {v}\n'

        if hasattr(self, 'parameter_estimator'):
            settings += '\nParameterEstimator Settings (parameter_estimator):\n'
            for k, v in self.parameter_estimator.items():
                settings += f'{str(k).rjust(m)} : {v}\n'

        if hasattr(self, 'solver'):
            settings += '\nSolver Settings (solver):\n'
            for k, v in self.solver.items():
                settings += f'{str(k).rjust(m)} : {v}\n'

        return settings

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _get_file_path():
        """Finds the path to the settings.yml file by referencing the directory
        of the current file
        
        """

        current_dir = Path(__file__).parent
        settings_file = (current_dir / '../general_settings/settings.yml').resolve()

        return settings_file

    def _load_settings(self):
        """Loads the settings and places the data into the correct dict
        structure
        
        """
        settings_file = self._get_file_path()

        with open(settings_file) as file:
            self.cfg = yaml.load(file, Loader=yaml.FullLoader)

            for sub_dict in self.cfg.values():
                for key, value in sub_dict.items():
                    if isinstance(value, bool):
                        continue
                    elif value in ['True', 'False', 'None']:
                        sub_dict[key] = ast.literal_eval(value)
                    elif _is_number(value):
                        if float(value) == int(float(value)):
                            sub_dict[key] = int(float(value))
                        else:
                            sub_dict[key] = float(value)
                    else:
                        None

        return None

    def _reset_model(self):
        """Initializes the settings dicts to their default values
        
        """
        self.collocation = AttrDict(self.cfg['collocation'])
        self.simulator = AttrDict(self.cfg['simulator'])
        self.simulator.update({'solver_opts': AttrDict()})
        self.general = AttrDict(self.cfg['general'])
        self.variance_estimator = AttrDict(self.cfg['variance_estimator'])
        self.variance_estimator.update({'solver_opts': AttrDict()})
        self.parameter_estimator = AttrDict(self.cfg['parameter_estimator'])
        self.parameter_estimator.update({'solver_opts': AttrDict()})
        self.solver = AttrDict(self.cfg['solver'])
        #self.units = AttrDict(self.cfg['units'])

        return None

    def update_settings(self, category, item, value):
        """Sets the default settings to some new value
        
        This allows the user to make permanent changes to the settings file.
        
        .. warning::
            
            Careful! This may result in KIPET not working properly if you make
            a change that is incompatible!
        
        :param str category: The category containing the value to change
        :param str item: The name of the setting to change
        :param str value: The new value for the setting
        
        :return: None
        
        """
        settings_file = self._get_file_path()

        self.cfg[category][item] = value
        with open(settings_file, 'w') as yaml_file:
            yaml_file.write(yaml.dump(self.cfg, default_flow_style=False))

        print(f'Updated default settings:\n\n{category}:\n\t{item}: {value}')
        return None

    @property
    def as_dicts(self):
        
        full_name = {
            'general' : 'General Settings',
            'collocation' : 'Collocation Settings',
            'simulator' : 'Simulation Settings',
            'variance_estimator' : 'Variance Estimator Settings',
            'parameter_estimator' : 'Parameter Estimator Settings',
            'solver' : 'Solver Settings',
        }
        
        keys = ['collocation', 'simulator', 'general', 'variance_estimator',
                'parameter_estimator', 'solver']
        
        nested_dict = {}
        for key in keys:
            nested_dict[key] = (full_name[key], getattr(self, key))
            
        return nested_dict
        


def _is_number(s):
    """ Returns True if the input is a float (number), else False

    :param str s: String that will be checked as to whether it is a number
    
    :return: True or False, depending on whether a number is detected.
    :rtype: bool
    
    """
    try:
        float(s)
        return True
    finally:
        return False
