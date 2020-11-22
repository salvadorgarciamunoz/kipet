"""
Settings for KIPET
"""
# Standard library imports
import ast
import os
from pathlib import Path

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
        
        self.general = AttrDict()
        self.collocation = AttrDict()
        self.solver = AttrDict()
        
        #if category == 'model':
        self.variance_estimator = AttrDict()
        self.parameter_estimator = AttrDict()

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
    
    def reset_model(self, specific_settings=None):
        """Initializes the settings dicts to their default values"""
        
        general = {'scale_variances': USER_DEFINED_SETTINGS['SCALE_VARIANCES'],
                   'initialize_pe': USER_DEFINED_SETTINGS['INITIALIZE_PARAMETER_ESTIMATOR'],
                   'scale_pe': USER_DEFINED_SETTINGS['SCALE_PARAMETER_ESTIMATOR'],
                   'scale_parameters': USER_DEFINED_SETTINGS['SCALE_PARAMETERS'],
                   'simulation_times': USER_DEFINED_SETTINGS['SIMULATION_TIMES'],
                   'no_user_scaling': USER_DEFINED_SETTINGS['NO_USER_SCALING'],
                   'data_directory': USER_DEFINED_SETTINGS['DATA_DIRECTORY'],
                   'charts_directory': USER_DEFINED_SETTINGS['CHART_DIRECTORY'],
            }
        
        collocation = {'method': USER_DEFINED_SETTINGS['COLLOCATION_METHOD'],
                       'ncp': USER_DEFINED_SETTINGS['COLLOCATION_POINTS'],
                       'nfe': USER_DEFINED_SETTINGS['FINITE_ELEMENTS'],
                       'scheme': USER_DEFINED_SETTINGS['DISCRETIZATION_SCHEME'],
            }
        
        sim_opt = {'solver': 'ipopt',
                   'method': 'dae.collocation',
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
                   'use_delta': False,
                   'delta': 1e-07,
                   'individual_species' : False,
                   'fixed_device_variance': None,
                   'device_range': None,
                   'best_accuracy': None,
                   'num_points': None,
                   'with_plots': False,
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
                   'G_contribution': None,
                   'St': dict(),
                   'Z_in': dict(),
                   'confidence': None,
            }
    
        solver = {'nlp_scaling_method': USER_DEFINED_SETTINGS['NLP_SCALING_METHOD'],
                  'linear_solver': USER_DEFINED_SETTINGS['LINEAR_SOLVER'],
            }
    
        self.collocation = AttrDict(collocation)
        self.simulator = AttrDict(sim_opt)
        self.general = AttrDict(general)
        self.variance_estimator = AttrDict(ve_opt)
        self.parameter_estimator = AttrDict(pe_opt)
        self.solver = AttrDict(solver)
        
        return None
    
    def reset_block(self, specific_settings=None):
        """Initializes the settings dicts to their default values"""
        
        general = {'scale_variances': False,
                   # If true, PE is intialized with VE results
                   'initialize_pe' : True,
                    # If true, PE is scaled with VE results
                   'scale_pe' : True,
                   'scale_parameters' : False,
                   'simulation_times': None,
                   'no_user_scaling': False,
                   'use_wavelength_subset': True,
                   'freq_wavelength_subset': 2,
                   'data_directory': USER_DEFINED_SETTINGS['DATA_DIRECTORY'],
                   'charts_directory': USER_DEFINED_SETTINGS['CHART_DIRECTORY'],
            }
        
        collocation = {'method': USER_DEFINED_SETTINGS['COLLOCATION_METHOD'],
                       'ncp': USER_DEFINED_SETTINGS['COLLOCATION_POINTS'],
                       'nfe': USER_DEFINED_SETTINGS['FINITE_ELEMENTS'],
                       'scheme': USER_DEFINED_SETTINGS['DISCRETIZATION_SCHEME'],
            }
        
        # This should only be in model and not in block:
    
        v_estimator = {'solver': 'ipopt',
                        'solver_opts': AttrDict(),
                        'tee': False,
                        'norm_order': np.inf,
                        'max_iter': 400,
                        'tol': 5e-05,
                        'subset_lambdas': None,
                        'lsq_ipopt': False,
                        'init_C': None,
                        'start_time': {},
                        'end_time': {},
                        }
    
        
        p_estimator = {'solver': 'ipopt',
                        'solver_opts': AttrDict(),
                        'tee': False,
                        'subset_lambdas': None,
                        'start_time': {},
                        'end_time': {},
                        'sigma_sq': {},
                        'spectra_problem': True,
                        'scaled_variance': False,
                        'confidence': None,
                        }
    
        solver = {'nlp_scaling_method': USER_DEFINED_SETTINGS['NLP_SCALING_METHOD'],
                  'linear_solver': USER_DEFINED_SETTINGS['LINEAR_SOLVER'],
            }
    
        self.collocation = AttrDict(collocation)
        self.general = AttrDict(general)
        self.parameter_estimator = AttrDict(p_estimator)
        self.variance_estimator = AttrDict(v_estimator)
        self.solver = AttrDict(solver)
        
        return None
    
def is_number(s):
    """ Returns the input as a float if it is a number else it returns an Error

    """
    try:
        float(s)
        return True
    except:
        return False
    
def read_settings_txt(settings_file='settings.txt'):
    """This method reads the settings.txt file in the kipet directory. You can
    change the default settings file if you have several custom settings you
    would like to use instead. In this way, the default settings can more
    easily be controlled. It is very important that the correct directory for
    the installation is used for setting the DATA_DIRECTORY
    
    Args:
        settings_file (str): The relatve path to the settings file
        
    Returns:
        user_fixed_settings (dict): The loaded settings as a python dict
    
    """
    abs_file_path = Path(os.path.abspath(os.path.dirname(__file__)))
    #print(abs_file_path)
    
    settings_file_abs = Path(__file__).parents[2].joinpath(settings_file)
    #print(settings_file_abs)
    
    user_fixed_settings = {}
    with settings_file_abs.open() as f: 
        lines = f.readlines()
        
    for line in lines:
        line = line.rstrip('\n')
        line = line.lstrip('\t').lstrip(' ')
        if line == '' or line[0] == '#':
            continue
        else:
            setting = line.split('=')
            str_to_eval = setting[1].rstrip(' ').lstrip(' ')
            if str_to_eval in ['True', 'False', 'None']:
                setting[1] = ast.literal_eval(setting[1])
            elif is_number(str_to_eval):
                setting[1] = float(setting[1])
            else:
                None
            user_fixed_settings[setting[0]] = setting[1]
    return user_fixed_settings

USER_DEFINED_SETTINGS = read_settings_txt()
# print(USER_DEFINED_SETTINGS)

