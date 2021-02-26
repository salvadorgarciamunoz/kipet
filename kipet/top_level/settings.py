"""
Settings for KIPET
"""
# Standard library imports
import ast
import os
from pathlib import Path
import yaml

# Third party imports
import numpy as np

# Kipet library imports
from kipet.top_level.helper import AttrDict


class Settings():
    
    """This is a container for all of the options that can be used in Kipet
    Since it can be confusing due to the large number of options, this should
    make it easier for the user to see everything in one spot.
    
    """
    def __init__(self, category='model'):
        
        self._load_settings()
        
        # Initialize to the defaults (can be used at anytime)
        #if category == 'model':
        self.reset_model()
        # else:
        #     self.reset_block()
        
    def __str__(self):
        
        m = 25
          
        settings = 'Settings\n\n'
        
        if hasattr(self, 'general'):
            settings += 'General Settings:\n'
            for k, v in self.general.items():
                settings += f'{str(k).rjust(m)} : {v}\n'
        
        if hasattr(self, 'units'):
            settings += '\nUnit Settings:\n'
            for k, v in self.units.items():
                settings += f'{str(k).rjust(m)} : {v}\n'
        
        if hasattr(self, 'collocation'):
            settings += '\nCollocation Settings:\n'
            for k, v in self.collocation.items():
                settings += f'{str(k).rjust(m)} : {v}\n'
            
        if hasattr(self, 'simulator'):
            settings += '\nSimulation Settings:\n'
            for k, v in self.simulator.items():
                settings += f'{str(k).rjust(m)} : {v}\n'
            
        if hasattr(self, 'variance_estimator'):
            settings += '\nVarianceEstimator Settings:\n'
            for k, v in self.variance_estimator.items():
                settings += f'{str(k).rjust(m)} : {v}\n'
    
        if hasattr(self, 'parameter_estimator'):
            settings += '\nParameterEstimator Settings:\n'
            for k, v in self.parameter_estimator.items():
                settings += f'{str(k).rjust(m)} : {v}\n'
        
        if hasattr(self, 'solver'):
            settings += '\nSolver Settings:\n'
            for k, v in self.solver.items():
                settings += f'{str(k).rjust(m)} : {v}\n'
        
        return settings
        
    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def _get_file_path():
        
        current_dir = Path(__file__).parent
        settings_file = (current_dir / '../settings.yml').resolve()
        
        return settings_file
    
    def _load_settings(self):
        
        settings_file = self._get_file_path()
        
        with open(settings_file) as file:
            self.cfg = yaml.load(file, Loader=yaml.FullLoader)
    
            for sub_dict in self.cfg.values():
                for key, value in sub_dict.items():
                    if isinstance(value, bool):
                        continue
                    elif value in ['True', 'False', 'None']:
                        sub_dict[key] = ast.literal_eval(value)
                    elif is_number(value):
                        if float(value) == int(float(value)):
                            sub_dict[key] = int(float(value))
                        else:
                            sub_dict[key] = float(value)
                    else:
                        None
        
        return None
    
    def reset_model(self, specific_settings=None):
        """Initializes the settings dicts to their default values"""
        # Check which settings are absolutely necessary (really basic settings
        # and sepearate from the problem specific attributes)
        
        self.collocation = AttrDict(self.cfg['collocation'])
        self.simulator = AttrDict(self.cfg['simulator'])
        self.simulator.update({'solver_opts': AttrDict()})
        self.general = AttrDict(self.cfg['general'])
        self.variance_estimator = AttrDict(self.cfg['variance_estimator'])
        self.variance_estimator.update({'solver_opts': AttrDict()})
        self.parameter_estimator = AttrDict(self.cfg['parameter_estimator'])
        self.parameter_estimator.update({'solver_opts': AttrDict()})
        self.solver = AttrDict(self.cfg['solver'])
        self.units = AttrDict(self.cfg['units'])
        
        return None
    
    
    def update_settings(self, category, item, value):
        
        """Sets the default settings to some new value
        
        Don't break KIPET!
        """
        settings_file = self._get_file_path()
        
        self.cfg[category][item] = value
        with open(settings_file, 'w') as yaml_file:
            yaml_file.write( yaml.dump(self.cfg, default_flow_style=False))
    
        print(f'Updated default settings:\n\n{category}:\n\t{item}: {value}')
        return None
    
def is_number(s):
    """ Returns True if the input is a float (number), else False

    """
    try:
        float(s)
        return True
    except:
        return False
    
    
    
    
