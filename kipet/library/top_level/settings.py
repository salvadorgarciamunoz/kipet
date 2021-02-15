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
from kipet.library.top_level.helper import AttrDict


class Settings():
    
    """This is a container for all of the options that can be used in Kipet
    Since it can be confusing due to the large number of options, this should
    make it easier for the user to see everything in one spot.
    
    """
    def __init__(self, category='model'):
        
        self._load_settings()
        
        # Initialize to the defaults (can be used at anytime)
        if category == 'model':
            self.reset_model()
        else:
            self.reset_block()
        
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
    
    def _load_settings(self):
        
        current_dir = Path(__file__).parent
        settings_file = (current_dir / '../../settings.yml').resolve()

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
        
        # general = {
        #     'scale_variances': USER_DEFINED_SETTINGS.get('SCALE_VARIANCES', False),
        #     'initialize_pe': USER_DEFINED_SETTINGS.get('INITIALIZE_PARAMETER_ESTIMATOR', True),
        #     'scale_pe': USER_DEFINED_SETTINGS.get('SCALE_PARAMETER_ESTIMATOR', True),
        #     'scale_parameters': USER_DEFINED_SETTINGS.get('SCALE_PARAMETERS', False),
        #     'simulation_times': USER_DEFINED_SETTINGS.get('SIMULATION_TIMES', None),
        #     'no_user_scaling': USER_DEFINED_SETTINGS.get('NO_USER_SCALING', True),
        #     'data_directory': USER_DEFINED_SETTINGS['DATA_DIRECTORY'],
        #     'charts_directory': USER_DEFINED_SETTINGS['CHART_DIRECTORY'],
        #     }

        # collocation = {
        #     'method': USER_DEFINED_SETTINGS.get('COLLOCATION_METHOD', 'dae.collocation'),
        #     'ncp': USER_DEFINED_SETTINGS.get('COLLOCATION_POINTS', 3),
        #     'nfe': USER_DEFINED_SETTINGS.get('FINITE_ELEMENTS', 50),
        #     'scheme': USER_DEFINED_SETTINGS.get('DISCRETIZATION_SCHEME', 'LAGRANGE-RADAU'),
        #     }
        
        # sim_opt = {
        #     'solver': USER_DEFINED_SETTINGS.get('SIMULATION_SOLVER', 'ipopt'),
        #     'method': USER_DEFINED_SETTINGS.get('COLLOCATION_METHOD', 'dae.collocation'),
        #     'tee':  USER_DEFINED_SETTINGS.get('DISPLAY_SOLVER_OUTPUT_SIMULATION', False),
        #     'solver_opts': AttrDict(),
        #     }
        
        # ve_opt = { 
        #     'solver': USER_DEFINED_SETTINGS.get('VARIANCE_ESTIMATOR_SOLVER', 'ipopt'),
        #     'tee': USER_DEFINED_SETTINGS.get('DISPLAY_SOLVER_OUTPUT_VARIANCE', True),
        #     'tolerance': USER_DEFINED_SETTINGS.get('VARIANCE_SOLVER_TOLERANCE', 1e-5),
        #     'max_iter': USER_DEFINED_SETTINGS.get('VARIANCE_MAX_ITER', 15),
        #     'method': USER_DEFINED_SETTINGS.get('VARIANCE_ESTIMATOR_METHOD', 'originalchenetal'),
        #     'freq_subset_lambdas': USER_DEFINED_SETTINGS.get('WAVELENGTH_SUBSET_FREQ', None),
        #     'secant_point': USER_DEFINED_SETTINGS.get('SECANT_POINT', 1e-11),
        #     'initial_sigmas': USER_DEFINED_SETTINGS.get('INITIAL_SIGMAS', 1e-10),
        #     'max_device_variance': USER_DEFINED_SETTINGS.get('MAX_DEVICE_VARIANCE', False),
        #     'use_delta': USER_DEFINED_SETTINGS.get('USE_DELTA', False),
        #     'delta': USER_DEFINED_SETTINGS.get('DELTA', 1e-7),
        #     'individual_species' : USER_DEFINED_SETTINGS.get('INDIVIDUAL_SPECIES', False),
        #     'fixed_device_variance': USER_DEFINED_SETTINGS.get('FIXED_DEVICE_VARIANCE', None),
        #     'device_range': USER_DEFINED_SETTINGS.get('DEVICE_RANGE', None),
        #     'best_accuracy': USER_DEFINED_SETTINGS.get('BEST_ACCURACY', None),
        #     'num_points': USER_DEFINED_SETTINGS.get('NUM_POINTS', None),
        #     'with_plots': USER_DEFINED_SETTINGS.get('SHOW_PLOTS', False),
        #     'solver_opts': AttrDict(),
        #     }
   
        # pe_opt = { 
        #     'solver': USER_DEFINED_SETTINGS.get('PARAMETER_ESTIMATOR_SOLVER', 'ipopt'),
        #     'tee': USER_DEFINED_SETTINGS.get('DISPLAY_SOLVER_OUTPUT_PARAMETER', True),
        #     'covariance': USER_DEFINED_SETTINGS.get('PARAMETER_COVARIANCE', False),
        #     'with_d_vars': USER_DEFINED_SETTINGS.get('PARAMETER_D_VARS', False),
        #     'symbolic_solver_labels': USER_DEFINED_SETTINGS.get('PARAMETER_SYMBOLIC_LABELS', False),
        #     'estimability': USER_DEFINED_SETTINGS.get('PARAMETER_ESTIMABILITY', False),
        #     'report_time': USER_DEFINED_SETTINGS.get('PARAMETER_REPORT_TIME', False),
        #     'model_variance': USER_DEFINED_SETTINGS.get('PARAMETER_MODEL_VARIANCE', True),
        #     'confidence': USER_DEFINED_SETTINGS.get('PARAMETER_CONFIDENCE', None),
        #     'solver_opts': AttrDict(),
        #     'sim_init': False,
            
        #     'inputs': None,
        #     'inputs_sub': None,
        #     'trajectories': None,
        #     'fixedtraj': False,
        #     'fixedy': False,
        #     'yfix': None,
        #     'yfixtraj': None,
        #     'jump': False,
        #     'jump_states': None,
        #     'jump_times': None,
        #     'feed_times': None,       
        #    # 'G_contribution': None,
        #    # 'St': dict(),
        #    # 'Z_in': dict(),
        #     }
    
        # solver = {
        #     'nlp_scaling_method': USER_DEFINED_SETTINGS.get('NLP_SCALING_METHOD', 'gradient-based'),
        #     'linear_solver': USER_DEFINED_SETTINGS.get('LINEAR_SOLVER', 'ma57'),
        #     }
    
        self.collocation = AttrDict(self.cfg['collocation'])
        self.simulator = AttrDict(self.cfg['simulator'])
        self.general = AttrDict(self.cfg['general'])
        self.variance_estimator = AttrDict(self.cfg['variance_estimator'])
        self.parameter_estimator = AttrDict(self.cfg['parameter_estimator'])
        self.solver = AttrDict(self.cfg['solver'])
        
        return None
    
    def reset_block(self, specific_settings=None):
        """Initializes the settings dicts to their default values"""
        
        # general = {
        #     'scale_variances': USER_DEFINED_SETTINGS.get('SCALE_VARIANCES', False),
        #     'scale_parameters': USER_DEFINED_SETTINGS.get('SCALE_PARAMETERS', False),
        #     'confidence' : None,
        #     'data_directory': USER_DEFINED_SETTINGS['DATA_DIRECTORY'],
        #     'charts_directory': USER_DEFINED_SETTINGS['CHART_DIRECTORY'],
        #     }
        
        # units = {
        #     'concentration': 'M',
        #     'time' : 'h',
        #     'volume' : 'L',
        #     }
        
        # solver = {
        #     'nlp_scaling_method': USER_DEFINED_SETTINGS.get('NLP_SCALING_METHOD', 'gradient-based'),
        #     'linear_solver': USER_DEFINED_SETTINGS.get('LINEAR_SOLVER', 'ma57'),
        #     'solver': 'ipopt',
        #     }
    
        self.general = AttrDict(self.cfg['general'])
        self.units = AttrDict(self.cfg['units'])
        self.solver = AttrDict(self.cfg['solver'])
        
        return None
    
def is_number(s):
    """ Returns True if the input is a float (number), else False

    """
    try:
        float(s)
        return True
    except:
        return False
    
# def read_settings_txt(settings_file='settings.txt'):
#     """This method reads the settings.txt file in the kipet directory. You can
#     change the default settings file if you have several custom settings you
#     would like to use instead. In this way, the default settings can more
#     easily be controlled. It is very important that the correct directory for
#     the installation is used for setting the DATA_DIRECTORY
    
#     Args:
#         settings_file (str): The relatve path to the settings file
        
#     Returns:
#         user_fixed_settings (dict): The loaded settings as a python dict
    
#     """
#     abs_file_path = Path(os.path.abspath(os.path.dirname(__file__)))
#     #print(abs_file_path)
    
#     settings_file_abs = Path(__file__).parents[2].joinpath(settings_file)
#     #print(settings_file_abs)
    
#     user_fixed_settings = {}
#     with settings_file_abs.open() as f: 
#         lines = f.readlines()
        
#     for line in lines:
#         line = line.rstrip('\n')
#         line = line.lstrip('\t').lstrip(' ')
#         if line == '' or line[0] == '#':
#             continue
#         else:
#             setting = line.split('=')
#             str_to_eval = setting[1].rstrip(' ').lstrip(' ')
#             if str_to_eval in ['True', 'False', 'None']:
#                 setting[1] = ast.literal_eval(setting[1])
#             elif is_number(str_to_eval):
#                 setting[1] = float(setting[1])
#             else:
#                 None
#             user_fixed_settings[setting[0]] = setting[1]
#     return user_fixed_settings

# USER_DEFINED_SETTINGS = read_settings_txt()
# # print(USER_DEFINED_SETTINGS)

