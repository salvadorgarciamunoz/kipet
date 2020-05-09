import copy
import inspect
import itertools
import logging
import numbers
import six
import sys
import warnings

import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.dae import *

try:
    if sys.version_info.major > 3:
        import importlib
        importlib.util.find_spec("casadi")
    else:
        import imp
        imp.find_module('casadi')

    from kipet.library.CasadiModel import CasadiModel
    from kipet.library.CasadiModel import KipetCasadiStruct
    found_casadi = True

except ImportError:
    found_casadi = False

logger = logging.getLogger('ModelBuilderLogger')
    

class KineticParameter():
    """A simple class for holding kinetic parameter data"""

    def __init__(self, name, bounds=None, init=None, uncertainty=None):

        self.name = name
        self.bounds = bounds
        self.init = init
        self.uncertainty = uncertainty
        
    def __str__(self):
        return f'KineticParameter: {self.name}, bounds={self.bounds}, init={self.init}, variance={self.uncertainty}'

    def __repr__(self):
        return f'KineticParameter: {self.name}, bounds={self.bounds}, init={self.init}, variance={self.uncertainty}'


class Component():
    """A simple class for holding component information"""
    
    def __init__(self, name, init, sigma=1, state='concentration'):
    
        self.name = name
        self.init = init
        self.sigma = sigma
        self.state = state

    def __str__(self):
        return f'Component: {self.name}, init={self.init}, sigma={self.sigma}'
    
    def __repr__(self):
        return f'Component: {self.name}, init={self.init}, sigma={self.sigma}'

class TemplateBuilder(object):
    """Helper class for creation of models.

    Attributes:
        _component_names (set): container with names of components.

        _parameters (dict): map of parameter names to corresponding values

        _parameters_bounds (dict): map of parameter names to bounds tuple

        _init_conditions (dict): map of component/state name to its initial condition

        _spectal_data (DataFrame, optional): DataFrame with time indices and wavelength columns

        _absorption_data (DataFrame, optional): DataFrame with wavelength indices and component names columns

        _odes (Function): function specified by user to return dictionary with ODE expressions

        _meas_times (set, optional): container of measurement times

        _huplcmeas_times (set, optional): container of  H/UPLC measurement times

        _allmeas_times (set, optional): container of all measurement times

        _feed_times (set, optional): container of feed times

        _complementary_states (set,optional): container with additional states

    """

    def __init__(self, **kwargs):
        """Template builder constructor

        Args:
            **kwargs: Arbitrary keyword arguments.
            concentrations (dictionary): map of component name to initial condition

            parameters (dictionary): map of parameter name to its corresponding value

            extra_states (dictionary): map of state name to initial condition

        """
        self._component_names = set()
        self._parameters = dict()
        self._parameters_init = dict()  # added for parameter initial guess CS
        self._parameters_bounds = dict()
        self._smoothparameters = dict()  # added for mutable parameters CS
        self._smoothparameters_mutable = dict() #added for mutable parameters CS

        #added for initial condition parameter estimates: CS
        self._initextraparams = dict()
        self._initextraparams_init = dict()  # added for parameter initial guess CS
        self._initextraparams_bounds = dict()

        self._y_bounds = dict()  # added for additional optional bounds CS
        self._prof_bounds = list()  # added for additional optional bounds MS
        self._init_conditions = dict()
        self._spectral_data = None
        self._concentration_data = None
        self._complementary_states_data = None # added for complementary state data (Est.) KM
        self._huplc_data = None #added for additional data CS
        self._smoothparam_data = None  # added for additional smoothing parameter data CS
        self._absorption_data = None
        self._odes = None
        self._meas_times = set()
        self._huplcmeas_times = set() #added for additional data CS
        self._allmeas_times = set() #added for new data structure CS
        self._m_lambdas = None
        self._complementary_states = set()
        self._algebraics = dict()
        self._algebraic_constraints = None
        self._known_absorbance = None
        self._is_known_abs_set = False
        self._warmstart = False  # add for warmstart CS
        self._known_absorbance_data = None
        self._non_absorbing = None
        self._is_non_abs_set = False
        self._huplc_absorbing = None #for add additional data CS
        self._is_huplc_abs_set = False #for add additional data CS
        self._vol = None #For add additional data CS
        self._feed_times = set()  # For inclusion of discrete feeds CS
        self._is_D_deriv = False
        self._is_C_deriv = False
        self._is_U_deriv = False
        self._is_Dhat_deriv = False
        self._state_sigmas = None # Need to put sigmas into the pyomo model as params
        self._model_constants = None # Used in EstimaationPotential
        self._scale_parameters = False # Should be True for EstimationPotential (automatic)
        self._times = None
        self._all_state_data = list()

  #New for estimate initial conditions of complementary states CS:
        self._estim_init = False
        self._allinitcomponents=dict()
        self._initextra_est_list = None
        
        # bounds and init for unwanted contributions KH.L
        self._qr_bounds = None
        self._qr_init = None
        self._g_bounds = None
        self._g_init = None

        components = kwargs.pop('concentrations', dict())
        if isinstance(components, dict):
            for k, v in components.items():
                self._component_names.add(k)
                self._init_conditions[k] = v
        else:
            raise RuntimeError('concentrations must be an dictionary component_name:init_condition')

        parameters = kwargs.pop('parameters', dict())
        if isinstance(parameters, dict):
            for k, v in parameters.items():
                self._parameters[k] = v
        elif isinstance(parameters, list):
            for k in parameters:
                self._parameters[k] = None
        else:
            raise RuntimeError('parameters must be a dictionary parameter_name:value or a list with parameter_names')

        extra_states = kwargs.pop('extra_states', dict())
        if isinstance(extra_states, dict):
            for k, v in extra_states.items():
                self._complementary_states.add(k)
                self._init_conditions[k] = v
        else:
            raise RuntimeError('Extra states must be an dictionary state_name:init_condition')

        algebraics = kwargs.pop('algebraics', dict())
        if isinstance(algebraics, dict):
            for k, v in algebraics.items():
                self._algebraics.add(k)
        elif isinstance(algebraics, set()):
            for k in algebraics:
                self._algebraics.add(k)
        elif isinstance(algebraics, list):
            for k in algebraics:
                self._algebraics[k] = None

        else:
            raise RuntimeError('concentrations must be an dictionary component_name:init_condition')
        
         #For initial condition parameter estimates:
        initextraparams = kwargs.pop('initextraparams', dict())
        if isinstance(initextraparams, dict):
            for k, v in initextraparams.items():
                self._initextraparams[k] = v
        elif isinstance(initextraparams, list):
            for k in initextraparams:
                self._initextraparams[k] = None
        else:
            raise RuntimeError('initextraparams must be a dictionary species_name:value or a list with species_names')


    def set_parameter_scaling(self, use_scaling: bool):
        """Makes an option to use the scaling method implemented for estimability
        
        Args:
            use_scaling (bool): Defaults to False, for using scaled parameters
            
        Returns:
            None
        
        """
        self._scale_parameters = use_scaling
    
        return None
        
    def set_model_times(self, times):
        """Add the times to the builder template.
        
        Args:
            times (iterable): start and end times for the pyomo model
        
        Returns:
            None
            
        """
        self._times = times
        
    def add_state_variance(self, sigma_dict):
        """Provide a variance for the measured states
        
        Args:
            sigma (dict): A dictionary of the measured states with known 
            variance. Provide the value of sigma (standard deviation).
            
        Returns:
            None
        """
        self._state_sigmas = sigma_dict
        # perhaps make this more secure later on and account for different input types
        return None
    
    def add_model_constants(self, constant_dict):
        """Add constants to the model (nominal parameters that are changed in
        the estimability calculations)
        
        Args:
            constant_dict (dict): A dict containing the nominal parameter
            values.
            
        Returns:
            None
        """
        if isinstance(constant_dict, dict):
            self._model_constants = constant_dict
        else:
            raise TypeError('Model constants must be given as a dict')
            
        return None

    def add_parameter(self, *args, **kwds):
        """Add a kinetic parameter(s) to the model.

        Note:
            Plan to change this method add parameters as PYOMO variables

            This method tries to mimic a template implementation. Depending
            on the argument type it will behave differently

        Args:
            param1 (str): Parameter name. Creates a variable parameter

            param1 (list): Parameter names. Creates a list of variable parameters

            param1 (dict): Map parameter name(s) to value(s). Creates a fixed parameter(s)

        Returns:
            None

        """
        bounds = kwds.pop('bounds', None)
        init = kwds.pop('init', None)
        
        if len(args) == 1:
            name = args[0]
            if isinstance(name, six.string_types):
                self._parameters[name] = None
                if bounds is not None:
                    self._parameters_bounds[name] = bounds
                if init is not None:
                    self._parameters_init[name] = init
            elif isinstance(name, list) or isinstance(name, set):
                if bounds is not None:
                    if len(bounds) != len(name):
                        raise RuntimeError('the list of bounds must be equal to the list of parameters')
                for i, n in enumerate(name):
                    self._parameters[n] = None
                    if bounds is not None:
                        self._parameters_bounds[n] = bounds[i]
                    if init is not None:
                        self._parameters_init[n] = init[i]
            elif isinstance(name, dict):
                if bounds is not None:
                    if len(bounds) != len(name):
                        raise RuntimeError('the list of bounds must be equal to the list of parameters')
                for k, v in name.items():
                    self._parameters[k] = v
                    if bounds is not None:
                        self._parameters_bounds[k] = bounds[k]
                    if init is not None:
                        self._parameters_init[k] = init[k]
            else:
                raise RuntimeError('Kinetic parameter data not supported. Try str')
        elif len(args) == 2:
            first = args[0]
            second = args[1]
            if isinstance(first, six.string_types):
                self._parameters[first] = second
                if bounds is not None:
                    self._parameters_bounds[first] = bounds
                if init is not None:
                    self._parameters_init[first] = init
            else:
                raise RuntimeError('Parameter argument not supported. Try str,val')
        else:
            raise RuntimeError('Parameter argument not supported. Try str,val')


    def add_smoothparameter(self, *args, **kwds): #added for smoothingparameter (CS)
        """Add a kinetic parameter(s) to the model.

        Note:
            Plan to change this method add parameters as PYOMO variables

            This method tries to mimic a template implementation. Depending
            on the argument type it will behave differently

        Args:
            param1 (boolean): Mutable value. Creates a variable mutable

        Returns:
            None

        """
        # bounds = kwds.pop('bounds', None)
        # init = kwds.pop('init', None)
        mutable = kwds.pop('mutable', False)

        if len(args) == 2:
            first = args[0]
            second = args[1]
            if isinstance(first, six.string_types):
                self._smoothparameters[first] = second
                if mutable is not False and second is pd.DataFrame:
                    self._smoothparameters_mutable[first] = mutable #added for mutable parameters CS
            else:
                raise RuntimeError('Parameter argument not supported. Try pandas.Dataframe and mutable=True')
        else:
            raise RuntimeError('Parameter argument not supported. Try pandas.Dataframe and mutable=True')

    def _add_state_variable(self, *args, data_type=None):
        
        built_in_data_types = {
            'concentration' : ('component_names', 'Mixture component'),
            'complementary_states' : ('complementary_states', 'Complementary state'),
           }
        
        if len(args) == 1:
            input = args[0]
            if isinstance(input, dict):
                for key, val in input.items():
                    if not isinstance(val, numbers.Number):
                        raise RuntimeError('The init condition must be a number. Try str, float')
                    getattr(self, f'_{built_in_data_types[data_type][0]}').add(key)
                    self._init_conditions[key] = val
            else:
                raise RuntimeError(f'{built_in_data_types[data_type][1]} data not supported. Try dict[str]=float')
        elif len(args) == 2:
            name = args[0]
            init_condition = args[1]

            if not isinstance(init_condition, numbers.Number):
                raise RuntimeError('The second argument must be a number. Try str, float')

            if isinstance(name, six.string_types):
                getattr(self, f'_{built_in_data_types[data_type][0]}').add(name)
                self._init_conditions[name] = init_condition
            else:
                raise RuntimeError(f'{built_in_data_types[data_type][1]} data not supported. Try str, float')
        else:
            raise RuntimeError(f'{built_in_data_types[data_type][1]} data not supported. Try str, float')

        return None


    def add_mixture_component(self, *args):
        """Add a component (reactive or product) to the model.

        This is a wrapper for adding concentration state variables to the
        template.

        Note:
            This method tries to mimic a template implmenetation. Depending
            on the argument type it will behave differently.

        Args:
            param1 (str): component name.

            param2 (float): initial concentration condition.

            param1 (dict): Map component name(s) to initial concentrations value(s).

        Returns:
            None

        """
        self._add_state_variable(*args, data_type='concentration')
        
        return None
    
    def add_complementary_state_variable(self, *args):
        """Add an extra state variable to the model.

        This is a wrapper for adding complementary state variables to the
        template.

        Note:
            This method tries to mimic a template implmenetation. Depending
            on the argument type it will behave differently

            Planning on changing this method to add variables in a pyomo fashion

        Args:
            param1 (str): variable name

            param2 (float): initial condition

            param1 (dict): Map component name(s) to initial condition value(s)

        Returns:
            None

        """
        self._add_state_variable(*args, data_type='complementary_states')
        
        return None
    

    def _add_state_data(self, data, data_type, label=None):
        """Generic method for adding data (concentration or complementary 
        state data) - uses the measured data attribute to process
        
        Args:
            data (DataFrame): DataFrame with measurement times as
                              indices and concentrations as columns.
                              
            data_type (str): The name of the attribute for where the data is to
                be stored.
                
            label (str): The label used to descibe the data in the pyomo model
                index.

        Returns:
            None

        """
        built_in_data_types = {
            'concentration' : 'C',
            'complementary_states' : 'U',
            'spectral' : 'D',
            'huplc' : 'Dhat',
            'smoothparam' : 'Ps',
            }
        
        state_data = ['C', 'U']
        deriv_data = ['C', 'U', 'D', 'Dhat']
        
        if label is None:
            try:
                label = built_in_data_types[data_type]
            except:
                raise ValueError("You need to provide a label for custom data types")
        
        if isinstance(data, pd.DataFrame):
            dfc = pd.DataFrame(index=self._feed_times, columns=data.columns)
            for t in self._feed_times:
                if t not in data.index:
                    dfc.loc[t] = [0.0 for n in range(len(data.columns))]
                    
            dfallc = data.append(dfc)
            dfallc.sort_index(inplace=True)
            dfallc.index = dfallc.index.to_series().apply(
                lambda x: np.round(x, 6))

            count = 0
            for j in dfallc.index:
                if count >= 1 and count < len(dfallc.index):
                    if dfallc.index[count] == dfallc.index[count - 1]:
                        dfallc = dfallc.dropna()
                        
                count += 1

            setattr(self, f'_{data_type}_data', dfallc)
            if label in state_data:
                self._all_state_data += list(data.columns)
        else:
            raise RuntimeError(f'{data_type.capitalize} data format not supported. Try pandas.DataFrame')
        
        if label in deriv_data:
            C = np.array(dfallc)
            for t in range(len(dfallc.index)):
                for l in range(len(dfallc.columns)):
                    if C[t, l] >= 0:
                        pass
                    else:
                        setattr(self, f'_is_{label}_deriv', True)
                        #self._is_C_deriv = True
            if getattr(self, f'_is_{label}_deriv') == True:
                print(
                    "Warning! Since {label}-matrix contains negative values Kipet is assuming a derivative of {label} has been inputted")

        return None
        
    def add_concentration_data(self, data):
        """Add concentration data as a wrapper to _add_state_data

        Args:
            data (DataFrame): DataFrame with measurement times as
                              indices and concentrations as columns.

        Returns:
            None

        """
        self._add_state_data(data,
                             data_type='concentration')
        
        return None
        
    def add_complementary_states_data(self, data):
        """Add complementary state data as a wrapper to _add_state_data

        Args:
            data (DataFrame): DataFrame with measurement times as
                              indices and complmentary states as columns.

        Returns:
            None

        """
        self._add_state_data(data,
                             data_type='complementary_states')
                           
        return None
    
    def add_spectral_data(self, data):
        """Add spectral data as a wrapper to _add_state_data

        Args:
            data (DataFrame): DataFrame with measurement times as
                              indices and wavelengths as columns.

        Returns:
            None

        """
        self._add_state_data(data,
                             data_type='spectral')
        
    def add_huplc_data(self, data): #added for the inclusion of h/uplc data CS
        """Add HPLC or UPLC data as a wrapper to _add_state_data

                Args:
                    data (DataFrame): DataFrame with measurement times as
                                      indices and wavelengths as columns.

                Returns:
                    None
        """
        self._add_state_data(data,
                             data_type='huplc')

    def add_smoothparam_data(self, data): #added for mutable smoothing parameter option CS
        """Add smoothing parameters as a wrapper to _add_state_data

                Args:
                    data (DataFrame): DataFrame with measurement times as
                                      indices and wavelengths as columns.

                Returns:
                    None
        """
        self._add_state_data(data,
                             data_type='smoothparam')


    def add_absorption_data(self, data):
        """Add absorption data

        Args:
            data (DataFrame): DataFrame with wavelengths as
                              indices and muxture components as columns.

        Returns:
            None

        """
        if isinstance(data, pd.DataFrame):
            self._absorption_data = data
        else:
            raise RuntimeError('Spectral data format not supported. Try pandas.DataFrame')

    # For inclusion of discrete jumps
    def add_feed_times(self, times):
        """Add measurement times to the model

        Args:
            times (array_like): feeding points

        Returns:
            None

        """
        for t in times:
            t = round(t,
                      6)  # for added ones when generating data otherwise too many digits due to different data types CS
            self._feed_times.add(t)
            self._meas_times.add(t)  # added here to avoid double addition CS

    # For inclusion of discrete jumps
    def add_measurement_times(self, times):
        """Add measurement times to the model

        Args:
            times (array_like): measurement points

        Returns:
            None

        """
        for t in times:
            t = round(t,6)  # for added ones when generating data otherwise too many digits due to different data types CS
            self._meas_times.add(t)

    # Why is this function here?
    def add_huplcmeasurement_times(self, times): #added for additional huplc times that are on a different time scale CS
        """Add H/UPLC measurement times to the model

        Args:
            times (array_like): measurement points

        Returns:
            None

        """
        for t in times:
            t = round(t,6)  # for added ones when generating data otherwise too many digits due to different data types CS
            self._huplcmeas_times.add(t)

            
    def add_algebraic_variable(self, *args, **kwds):
        """Add an algebraic variable to the model

            This method tries to mimic a template implmenetation. Depending
            on the argument type it will behave differently

        Args:
            param1 (str): Variable name. Creates a variable parameter

            param1 (list): Variable names. Creates a list of variable parameters

        Returns:
            None

        """
        bounds = kwds.pop('bounds', None)

        if len(args) == 1:
            name = args[0]
            if isinstance(name, six.string_types):
                self._algebraics[name] = None
                if bounds is not None:
                    self._y_bounds[name] = bounds
            elif isinstance(name, list) or isinstance(name, set):
                if bounds is not None:
                    if len(bounds) != len(name):
                        raise RuntimeError('the list of bounds must be equal to the list of parameters')
                for i, n in enumerate(name):
                    self._algebraics[n] = None
                    if bounds is not None:
                        self._y_bounds[n] = bounds[i]
            else:
                raise RuntimeError('To add an algebraic please pass name')

    # read bounds and initialize for qr and g (unwanted contri variables) from users KH.L
    def add_qr_bounds_init(self, **kwds):
        bounds = kwds.pop('bounds', None)
        init = kwds.pop('init', None)
        
        self._qr_bounds = bounds
        self._qr_init = init
        
    def add_g_bounds_init(self, **kwds):
        bounds = kwds.pop('bounds', None)
        init = kwds.pop('init', None)
        
        self._g_bounds = bounds
        self._g_init = init
    
    def set_odes_rule(self, rule):
        """Set the ode expressions.

        Defines the ordinary differential equations that define the dynamics of the model

        Args:
            rule (function): Python function that returns a dictionary

        Returns:
            None

        """
        inspector = inspect.getargspec(rule)
        if len(inspector.args) != 2:
            raise RuntimeError('The rule should have two inputs')
        self._odes = rule

    def set_algebraics_rule(self, rule):
        """Set the algebraic expressions.

        Defines the algebraic equations for the system

        Args:
            rule (function): Python function that returns a list of tuples

        Returns:
            None

        """
        inspector = inspect.getargspec(rule)
        if len(inspector.args) != 2:
            raise RuntimeError('The rule should have two inputs')
        self._algebraic_constraints = rule

    def bound_profile(self, var, bounds, comp=None, profile_range=None):
        """function that allows the user to bound a certain profile to some value

        Args:
            var (pyomo variable object): the pyomo variable that we will bound
            comp (str, optional): The component that bound applies to
            profile_range (tuple,optional): the range within the set to be bounded
            bounds (tuple): the values to bound the profile to

        Returns:
            None

        """
        if not isinstance(var, str):
            raise RuntimeError('var argument needs to be type string')

        if var not in ['C', 'U', 'S']:
            raise RuntimeError('var argument needs to be either C, U, or S')

        if comp is not None:
            if not isinstance(comp, str):
                raise RuntimeError('comp argument needs to be type string')
            if comp not in self._component_names:
                raise RuntimeError('comp needs to be one of the components')

        if profile_range is not None:
            if not isinstance(profile_range, tuple):
                raise RuntimeError('profile_range needs to be a tuple')
                if profile_range[0] > profile_range[1]:
                    raise RuntimeError('profile_range[0] must be greater than profile_range[1]')

        if not isinstance(bounds, tuple):
            raise RuntimeError('bounds needs to be a tuple')

        self._prof_bounds.append([var, comp, profile_range, bounds])

    def _validate_data(self, model, start_time, end_time):
        """Verify all inputs to the model make sense.

        This method is not suppose to be used by users. Only for developers use

        Args:
            model (pyomo or casadi model): Model

            start_time (float): initial time considered in the model

            end_time (float): final time considered in the model

        Returns:
            None

        """
        if not self._component_names:
            warnings.warn('The Model does not have any mixture components')
        else:
            if self._odes:
                dummy_balances = self._odes(model, start_time)
                if len(self._component_names) + len(self._complementary_states) != len(dummy_balances):
                    print(
                        'WARNING: The number of ODEs is not the same as the number of state variables.\n If this is the desired behavior, some odes must be added after the model is created.')

            else:
                print(
                    'WARNING: differential expressions not specified. Must be specified by user after creating the model')

        if self._algebraics:
            if self._algebraic_constraints:
                dummy_balances = self._algebraic_constraints(model, start_time)
                if len(self._algebraics) != len(dummy_balances):
                    print(
                        'WARNING: The number of algebraic equations is not the same as the number of algebraic variables.\n If this is the desired behavior, some algebraics must be added after the model is created.')
            else:
                print(
                    'WARNING: algebraic expressions not specified. Must be specified by user after creating the model')

        if self._absorption_data is not None:
            if not self._meas_times:
                raise RuntimeError('Need to add measurement times')

    def create_pyomo_model(self, start_time=None, end_time=None, parameter_normalization=False):
        """Create a pyomo model.

        This method is the core method for further simulation or optimization studies

        Args:
            start_time (float): initial time considered in the model

            end_time (float): final time considered in the model

        Returns:
            Pyomo ConcreteModel

        """
        if self._times is not None:
            if start_time is None:
                start_time = self._times[0]
            if end_time is None:
                end_time = self._times[1]
        else:
            if start_time is None:
                raise ValueError('A start time must be provided')
            if end_time is None:
                raise ValueError('An end time must be provided')
        
        # Model
        pyomo_model = ConcreteModel()

        # Sets
        pyomo_model.mixture_components = Set(initialize=self._component_names)
        pyomo_model.parameter_names = Set(initialize=self._parameters.keys())
        pyomo_model.complementary_states = Set(initialize=self._complementary_states)
        pyomo_model.states = pyomo_model.mixture_components | pyomo_model.complementary_states
        pyomo_model.algebraics = Set(initialize=self._algebraics.keys())

        # New Set for actual data inputs
        pyomo_model.measured_data = Set(initialize=self._all_state_data)
        # Make constants that are equal to the initial guess and set params to 1
        
        list_times = self._meas_times
        m_times = sorted(list_times)
        list_feedtimes = self._feed_times  # For inclusion of discrete feeds CS
        feed_times = sorted(list_feedtimes)  # For inclusion of discrete feeds CS
        m_lambdas = list()
        m_alltimes = m_times
        conc_times = list()

        if self._smoothparam_data is not None:#added for optional smoothing parameter values (mutable) read from file CS
            pyomo_model.smoothparameter_names = Set(initialize=self._smoothparameters.keys()) #added for mutable parameters
            pyomo_model.smooth_param_datatimes = Set(initialize=sorted(self._smoothparam_data.index))
            help_dict=dict()
            for k in self._smoothparam_data.index:
                for j in self._smoothparam_data.columns:
                    help_dict[k, j]=float(self._smoothparam_data[j][k])
            pyomo_model.smooth_param_data = Param(self._smoothparam_data.index, sorted(self._smoothparam_data.columns), initialize=help_dict)

        if self._spectral_data is not None and self._absorption_data is not None:
            raise RuntimeError('Either add absorption data or spectral data but not both')

        # I don't know why m_times is changed for each of these - what if there are more than one?

        if self._spectral_data is not None and self._huplc_data is None:
            list_times = list_times.union(set(self._spectral_data.index))
            list_lambdas = list(self._spectral_data.columns)
            m_times = sorted(list_times)
            m_lambdas = sorted(list_lambdas)
            m_alltimes=m_times

        if self._absorption_data is not None:
            if not self._meas_times:
                raise RuntimeError('Need to add measurement times')
            list_times = list(self._meas_times)
            list_lambdas = list(self._absorption_data.index)
            m_times = sorted(list_times)
            m_lambdas = sorted(list_lambdas)

        if self._concentration_data is not None:
            list_times = list_times.union(set(self._concentration_data.index))
            list_concs = list(self._concentration_data.columns)
            m_times = sorted(list_times)
            conc_times = sorted(list_times)
            m_alltimes = sorted(list_times)#has to be changed for including huplc data with conc data!
            m_concs = sorted(list_concs)

        # New complementary state data KM - This should only be T
        if self._complementary_states_data is not None:
            list_times = list_times.union(set(self._complementary_states_data.index))
            list_comps = list(self._complementary_states_data.columns)
            m_times = sorted(list_times)
            m_alltimes = sorted(list_times)
            m_comps = sorted(list_comps)


        #For inclusion of h/uplc data:
        if self._huplc_data is not None and self._spectral_data is None: #added for additional H/UPLC data (CS)
            list_huplctimes = self._huplcmeas_times
            list_huplctimes = list_huplctimes.union(set(self._huplc_data.index))
            list_conDhats = list(self._huplc_data.columns)
            m_huplctimes = sorted(list_huplctimes)

        if self._huplc_data is not None and self._spectral_data is not None:
            list_times = list_times.union(set(self._spectral_data.index))
            list_lambdas = list(self._spectral_data.columns)
            m_times = sorted(list_times)
            m_lambdas = sorted(list_lambdas)
            list_huplctimes = self._huplcmeas_times
            list_huplctimes = list_huplctimes.union(set(self._huplc_data.index))
            list_conDhats = list(self._huplc_data.columns)
            m_huplctimes = sorted(list_huplctimes)
            list_alltimes = list_times.union(list_huplctimes)
            m_alltimes = sorted(list_alltimes)

        #added for optional smoothing CS:
        if self._smoothparam_data is not None:
            if self._huplc_data is not None and self._spectral_data is not None:
                list_alltimes = list_times.union(list_huplctimes)
                m_alltimes = sorted(list_alltimes)
                list_alltimessmooth = list_alltimes.union(set(self._smoothparam_data.index))
                m_allsmoothtimes = sorted(list_alltimessmooth)
            else:
                list_timessmooth = list_times.union(set(self._smoothparam_data.index))
                m_alltimes = m_times
                m_allsmoothtimes = sorted(list_timessmooth)

        if self._huplc_data is not None:
            if m_huplctimes:
                if m_huplctimes[0] < start_time:
                    raise RuntimeError(
                        'Measurement time {0} not within ({1},{2})'.format(m_huplctimes[0], start_time, end_time))
                if m_huplctimes[-1] > end_time:
                    raise RuntimeError(
                        'Measurement time {0} not within ({1},{2})'.format(m_huplctimes[-1], start_time, end_time))

        if self._smoothparam_data is not None:
            if m_allsmoothtimes:
                if m_allsmoothtimes[0] < start_time:
                    raise RuntimeError(
                        'Measurement time {0} not within ({1},{2})'.format(m_allsmoothtimes[0], start_time, end_time))
                if m_allsmoothtimes[-1] > end_time:
                    raise RuntimeError(
                        'Measurement time {0} not within ({1},{2})'.format(m_allsmoothtimes[-1], start_time, end_time))

            pyomo_model.allsmooth_times = Set(initialize=m_allsmoothtimes, ordered=True)

        # Add given state standard deviations to the pyomo model
        if self._state_sigmas is not None:
            
            state_sigmas = {k: v for k, v in self._state_sigmas.items() if k in pyomo_model.measured_data}
            pyomo_model.sigma = Param(pyomo_model.measured_data, initialize=state_sigmas)
        else:
            pyomo_model.sigma = Param(pyomo_model.measured_data, initialize=1)
        
        if self._scale_parameters:
            pyomo_model.K = Param(pyomo_model.parameter_names, 
                                  initialize=self._parameters_init,
                                  mutable=True,
                                  default=1)

        if m_alltimes:
            if m_alltimes[0] < start_time:
                raise RuntimeError('Measurement time {0} not within ({1},{2})'.format(m_alltimes[0], start_time, end_time))
            if m_alltimes[-1] > end_time:
                raise RuntimeError(
                    'Measurement time {0} not within ({1},{2})'.format(m_alltimes[-1], start_time, end_time))

        self._m_lambdas = m_lambdas

        # For inclusion of discrete feeds CS
        if self._feed_times is not None:
            list_feedtimes = list(self._feed_times)
            feed_times = sorted(list_feedtimes)

        if self._huplc_data is not None:#added for addition huplc data CS
            pyomo_model.huplcmeas_times = Set(initialize=m_huplctimes, ordered=True)
            pyomo_model.huplctime = ContinuousSet(initialize=pyomo_model.huplcmeas_times,
                                                  bounds=(start_time, end_time))
            
        # all of these times are confusing - must it be this way?
        pyomo_model.allmeas_times = Set(initialize=m_alltimes, ordered=True) #add for new data structure CS
        pyomo_model.meas_times = Set(initialize=m_times, ordered=True)
        pyomo_model.feed_times = Set(initialize=feed_times, ordered=True)  # For inclusion of discrete feeds CS
        pyomo_model.meas_lambdas = Set(initialize=m_lambdas, ordered=True)
        
        pyomo_model.alltime = ContinuousSet(initialize=pyomo_model.allmeas_times,
                                         bounds=(start_time, end_time)) #add for new data structure CS

        # Parameters
        
        pyomo_model.init_conditions = Param(pyomo_model.states,
                                            initialize=self._init_conditions,
                                            mutable=True)
        pyomo_model.start_time = Param(initialize=start_time)
        pyomo_model.end_time = Param(initialize=end_time)

        ######################################################################
        # Variables
        ######################################################################
        
        # Declaration and initialization of predicted concentrations and states
        
        model_pred_var_name = {
                'Z' : pyomo_model.mixture_components,
                'X' : pyomo_model.complementary_states,
                    }
        
        for var, model_set in model_pred_var_name.items():
        
            setattr(pyomo_model, var, Var(pyomo_model.alltime,
                                          model_set,
                                          # bounds=(0.0,None),
                                          initialize=1) 
                    )    
        
            for time, comp in getattr(pyomo_model, var):
                if time == pyomo_model.start_time.value:
                    getattr(pyomo_model, var)[time, comp].value = self._init_conditions[comp]
                   
            setattr(pyomo_model, f'd{var}dt', DerivativeVar(getattr(pyomo_model, var),
                                                            wrt=pyomo_model.alltime)
                    )
                   
        # Variables of provided data - set as fixed variables complementary to above
        
        fixed_var_name = {
                'C' : self._concentration_data,
                'U' : self._complementary_states_data,
                    }
        
        for var, data in fixed_var_name.items():
            c_dict = dict()
            if getattr(self, f'_is_{var}_deriv') == True:
                c_bounds = (None, None)
            else:
                c_bounds = (0.0, None)
    
            if data is not None:    
                for i, row in data.iterrows():
                    c_dict.update({(i, col): float(row[col]) for col in data.columns})
                
                setattr(pyomo_model, f'{var}_indx', Set(initialize=c_dict.keys(), ordered=True))
                setattr(pyomo_model, var, Var(getattr(pyomo_model, f'{var}_indx'),
                                              bounds=c_bounds,
                                              initialize=c_dict,
                                              )
                        )
                
                for k, v in getattr(pyomo_model, var).items():
                    getattr(pyomo_model, var)[k].fixed = True
            
            else:
                setattr(pyomo_model, var, Var(pyomo_model.allmeas_times,
                                pyomo_model.mixture_components,
                                bounds=c_bounds,
                                initialize=1))
    
                for time, comp in getattr(pyomo_model, var):
                    if time == pyomo_model.start_time.value:
                        print(f'initial values: {time}, {comp}')
                        getattr(pyomo_model, var)[time, comp].value = self._init_conditions[comp]

        # End intialization for C and U
        
        
        ######################################################################
        # Parameters
        ######################################################################
    
        p_dict = dict()
        for param, init_value in self._parameters.items():
            if init_value is not None and init_value is not pd.DataFrame:
                p_dict[param] = init_value

            # added for option of providing initial guesses CS:
            elif param in self._parameters_init.keys():
                p_dict[param] = self._parameters_init[param]
                #for param, init_value in self._parameters_init.items():
                    #p_dict[p] = init_value
            else:
                for param, bounds in self._parameters_bounds.items():
                    lb = bounds[0]
                    ub = bounds[1]
                    p_dict[param] = (ub - lb) / 2 + lb

        if self._scale_parameters:
            pyomo_model.P = Var(pyomo_model.parameter_names,
                            bounds = (0.1, 10),
                            initialize=1)

        else:
            pyomo_model.P = Var(pyomo_model.parameter_names,
                            # bounds = (0.0,None),
                            initialize=p_dict)

        # set bounds P
        for k, v in self._parameters_bounds.items():
            lb = v[0]
            ub = v[1]
            pyomo_model.P[k].setlb(lb)
            pyomo_model.P[k].setub(ub)

        if self._estim_init==True:#added for the estimation of initial conditions which have to be complementary state vars CS
            pyomo_model.initparameter_names = Set(initialize=self._initextraest)
            # pyomo_model.initparameter_names.pprint()
            pyomo_model.add_component('initextraparams', Set(initialize=self._initextraest))
            pinit_dict = dict()
            for p, v in self._initextraparams.items():
                if v is not None and v is not pd.DataFrame:
                    pinit_dict[p] = v

                # added for option of providing initial guesses CS:
                elif p in self._initextraparams_init.keys():
                    for p, l in self._initextraparams_init.items():
                        pinit_dict[p] = l
                else:
                    for p, s in self._initextraparams_bounds.items():
                        lb = s[0]
                        ub = s[1]
                        pinit_dict[p] = (ub - lb) / 2
            pyomo_model.Pinit = Var(pyomo_model.initparameter_names,initialize=pinit_dict)

            for k, v in self._initextraparams_bounds.items():
                lb = v[0]
                ub = v[1]
                pyomo_model.Pinit[k].setlb(lb)
                pyomo_model.Pinit[k].setub(ub)
            for k in pyomo_model.initparameter_names:
                pyomo_model.Pinit[k].fixed = True #Just added for bound etc functionalities, variable is init_conditions
                # but Pinit is set to the value after solution of the parameter estimation problem CS

        #for optional smoothing parameters (CS):
        if isinstance(self._smoothparameters, dict) and self._smoothparam_data is not None:
            ps_dict = dict()
            for k in self._smoothparam_data.columns:
                for c in pyomo_model.allsmooth_times:
                    ps_dict[c, k] = float(self._smoothparam_data[k][c])

            ps_dict2 = dict()
            for p in self._smoothparameters.keys():
                for t in pyomo_model.alltime:
                    if t in pyomo_model.allsmooth_times:
                        ps_dict2[t, p] = float(ps_dict[t, p])
                    else:
                        ps_dict2[t, p] = 0.0

            pyomo_model.Ps = Param(pyomo_model.alltime, pyomo_model.smoothparameter_names, initialize=ps_dict2, mutable=True, default=20.)#here just set to some value that is noc

        
        
        pyomo_model.Y = Var(pyomo_model.alltime,
                            pyomo_model.algebraics,
                            initialize=1.0)

        # Add optional bounds for algebraic variables (CS):
        for t in pyomo_model.alltime:
            for k, v in self._y_bounds.items():
                lb = v[0]
                ub = v[1]
                pyomo_model.Y[t, k].setlb(lb)
                pyomo_model.Y[t, k].setub(ub)

        # Can this be handled as above?

        if self._absorption_data is not None:
            s_dict = dict()
            for k in self._absorption_data.columns:
                for l in self._absorption_data.index:
                    s_dict[l, k] = float(self._absorption_data[k][l])
        else:
            s_dict = 1.0

        if self._is_D_deriv == True:
            s_bounds = (None, None)
        else:
            s_bounds = (0.0, None)

        # Fixes parameters that were given numeric values
        for p, v in self._parameters.items():
            if v is not None:
                pyomo_model.P[p].value = v
                pyomo_model.P[p].fixed = True

        # spectral data
        if self._spectral_data is not None: #changed for new data structure CS
            s_data_dict = dict()
            for t in pyomo_model.meas_times:
                for l in pyomo_model.meas_lambdas:
                    if t in pyomo_model.meas_times:
                        s_data_dict[t, l] = float(self._spectral_data[l][t])
                    else:
                        s_data_dict[t, l] = float('nan') #missing time points are set to nan to filter out later

            pyomo_model.D = Param(pyomo_model.meas_times,
                                  pyomo_model.meas_lambdas,
                                  initialize=s_data_dict)
        
        # unwanted contributions: create variables qr and g KH.L
        if self._qr_bounds is not None:
            qr_bounds = self._qr_bounds
        elif self._qr_bounds is None:
            qr_bounds = None
            
        if self._qr_init is not None:
            qr_init = self._qr_init
        elif self._qr_init is None:
            qr_init = 1.0
            
        pyomo_model.qr = Var(pyomo_model.alltime, bounds=qr_bounds, initialize=qr_init)
        
        
        if self._g_bounds is not None:
            g_bounds = self._g_bounds
        elif self._g_bounds is None:
            g_bounds = None
            
        if self._g_init is not None:
            g_init = self._g_init
        elif self._g_init is None:
            g_init = 0.1
            
        pyomo_model.g = Var(pyomo_model.meas_lambdas, bounds=g_bounds, initialize=g_init)
        
        ####### Added for new add data CS:
        if self._huplc_data is not None and self._is_huplc_abs_set:
            self.set_huplc_absorbing_species(pyomo_model, self._huplc_absorbing, self._vol, self._solid_spec, check=False)

            Dhat_dict = dict()
            for k in self._huplc_data.columns:
                for c in self._huplc_data.index:
                    Dhat_dict[c, k] = float(self._huplc_data[k][c])

        if self._is_Dhat_deriv == True:
            Dhat_bounds = (None, None)
        else:
            Dhat_bounds = (0.0, None)
            

        ###########

        # validate the model before writing constraints
        self._validate_data(pyomo_model, start_time, end_time)

        # add ode contraints to pyomo model
        def rule_init_conditions(m, k):
            st = start_time
            if k in m.mixture_components:
                return m.Z[st, k] == self._init_conditions[k]
            else:
                return m.X[st, k] == self._init_conditions[k]

        pyomo_model.init_conditions_c = \
            Constraint(pyomo_model.states, rule=rule_init_conditions)

        # the generation of the constraints is not efficient but not critical
        if self._odes:
            
            def rule_odes(m, t, k):
                exprs = self._odes(m, t)
            
                ### Test Area End ###
                if t == m.start_time.value:
                    return Constraint.Skip
                else:
                    if k in m.mixture_components:
                        if k in exprs.keys():
                            return m.dZdt[t, k] == exprs[k]
                        else:
                            return Constraint.Skip
                    else:
                        if k in exprs.keys():
                            return m.dXdt[t, k] == exprs[k]
                        else:
                            return Constraint.Skip

            pyomo_model.odes = Constraint(pyomo_model.alltime,
                                          pyomo_model.states,
                                          rule=rule_odes)

        # the generation of the constraints is not efficient but not critical
        if self._algebraic_constraints:
            n_alg_eqns = len(self._algebraic_constraints(pyomo_model, start_time))

            def rule_algebraics(m, t, k):
                alg_const = self._algebraic_constraints(m, t)[k]
                return alg_const == 0.0

            pyomo_model.algebraic_consts = Constraint(pyomo_model.alltime,
                                                      range(n_alg_eqns),
                                                      rule=rule_algebraics)
        #######################
        ###Distinguish between S with non_absorbing components and S with no non_absorbing components (non_absorbing components excluded from S) (CS):
        if self._is_non_abs_set:  #: in case of a second call after non_absorbing has been declared
            self.set_non_absorbing_species(pyomo_model, self._non_absorbing, check=False)
            pyomo_model.S = Var(pyomo_model.meas_lambdas,
                                pyomo_model.abs_components,
                                bounds=s_bounds,
                                initialize=s_dict)
        else:
            pyomo_model.S = Var(pyomo_model.meas_lambdas,
                                pyomo_model.mixture_components,
                                bounds=s_bounds,
                                initialize=s_dict)
        ######################

        if self._absorption_data is not None:
            for l in pyomo_model.meas_lambdas:
                for k in pyomo_model.mixture_components:
                    pyomo_model.S[l, k].fixed = True

        
        # Iterate throught the component variables and apply the bounds to
        # the speicifc time period provided
        for bound_set in self._prof_bounds:
            # Why would I set bounds on only S, U, or C - these are fixed, right?
            var = bound_set[0]
            component_name = bound_set[1]
            if bound_set[2] is not None:
                bound_time_start = bound_set[2][0]
                bound_time_end = bound_set[2][1]
            upper_bound = bound_set[3][1]
            lower_bound = bound_set[3][0]
            # if var == 'C':
            #     var = 'Z'
            #print(f'for {var}')
            
            for time, comp in getattr(pyomo_model, var):
                if component_name == comp or component_name is None:
                    if bound_set[2] is not None:
                        if time >= bound_time_start and time < bound_time_end:
                            getattr(pyomo_model, var)[time, comp].setlb(lower_bound)
                            getattr(pyomo_model, var)[time, comp].setub(upper_bound)
                    else:
                        getattr(pyomo_model, var)[time, comp].setlb(lower_bound)
                        getattr(pyomo_model, var)[time, comp].setub(upper_bound)

        ### Original replaced by above
        # This section provides bounds if user used bound_profile (MS)
        # for i in self._prof_bounds:
        #     if i[0] == 'C':
        #         for t, c in pyomo_model.C:
        #             if i[1] == c:
        #                 if i[2]:
        #                     if t >= i[2][0] and t < i[2][1]:
        #                         pyomo_model.C[t, c].setlb(i[3][0])
        #                         pyomo_model.C[t, c].setub(i[3][1])
        #                 else:
        #                     pyomo_model.C[t, c].setlb(i[3][0])
        #                     pyomo_model.C[t, c].setub(i[3][1])

        #             elif i[1] == None:
        #                 if i[2]:
        #                     if t >= i[2][0] and t < i[2][1]:
        #                         pyomo_model.C[t, c].setlb(i[3][0])
        #                         pyomo_model.C[t, c].setub(i[3][1])
        #                 else:
        #                     pyomo_model.C[t, c].setlb(i[3][0])
        #                     pyomo_model.C[t, c].setub(i[3][1])
      
        #: in case of a second call after known_absorbing has been declared
        if self._is_known_abs_set:  
            self.set_known_absorbing_species(pyomo_model,
                                             self._known_absorbance,
                                             self._known_absorbance_data,
                                             check=False
                                             )
        
        if self._estim_init:  #: in case of a second call after known_absorbing has been declared
            self.set_estinit_extra_species(pyomo_model,
                                           self._initextra_est_list,
                                           check=False)


        return pyomo_model

    def create_casadi_model(self, start_time, end_time):
        """Create a casadi model.

        Casadi models are for simulation purposes mainly

        Args:
            start_time (float): initial time considered in the model

            end_time (float): final time considered in the model

        Returns:
            CasadiModel

        """
        if found_casadi:
            # Model
            casadi_model = CasadiModel()

            # Sets
            casadi_model.mixture_components = copy.deepcopy(self._component_names)
            casadi_model.parameter_names = self._parameters.keys()
            casadi_model.complementary_states = copy.deepcopy(self._complementary_states)
            casadi_model.states = casadi_model.mixture_components.union(casadi_model.complementary_states)
            casadi_model.algebraics = self._algebraics

            m_times = list()
            m_lambdas = list()
            if self._spectral_data is not None and self._absorption_data is not None:
                raise RuntimeError('Either add absorption data or spectral data but not both')

            if self._spectral_data is not None:
                list_times = list(self._spectral_data.index)
                list_lambdas = list(self._spectral_data.columns)
                m_times = sorted(list_times)
                m_lambdas = sorted(list_lambdas)
            if self._absorption_data is not None:
                if not self._meas_times:
                    raise RuntimeError('Need to add measumerement times')
                list_times = list(self._meas_times)
                list_lambdas = list(self._absorption_data.index)
                m_times = sorted(list_times)
                m_lambdas = sorted(list_lambdas)

            if m_times:
                if m_times[0] < start_time:
                    raise RuntimeError(
                        'Measurement time {0} not within ({1},{2})'.format(m_times[0], start_time, end_time))
                if m_times[-1] > end_time:
                    raise RuntimeError(
                        'Measurement time {0} not within ({1},{2})'.format(m_times[-1], start_time, end_time))

            casadi_model.meas_times = m_times
            casadi_model.allmeas_times = m_times
            casadi_model.meas_lambdas = m_lambdas

            # Variables
            casadi_model.Z = KipetCasadiStruct('Z', list(casadi_model.mixture_components), dummy_index=True)
            casadi_model.X = KipetCasadiStruct('X', list(casadi_model.complementary_states), dummy_index=True)
            casadi_model.Y = KipetCasadiStruct('Y', list(casadi_model.algebraics), dummy_index=True)
            casadi_model.P = KipetCasadiStruct('P', list(casadi_model.parameter_names))
            casadi_model.C = KipetCasadiStruct('C', list(casadi_model.allmeas_times))
            casadi_model.S = KipetCasadiStruct('S', list(casadi_model.meas_lambdas))

            if self._parameters_bounds:
                warnings.warn('Casadi_model do not take bounds on parameters. This is ignored in the integration')

            # Parameters
            casadi_model.init_conditions = self._init_conditions
            casadi_model.start_time = start_time
            casadi_model.end_time = end_time

            if self._absorption_data is not None:
                for l in casadi_model.meas_lambdas:
                    for k in casadi_model.mixture_components:
                        casadi_model.S[l, k] = float(self._absorption_data[k][l])

            if self._spectral_data is not None:
                casadi_model.D = dict()
                for t in casadi_model.meas_times:
                    for l in casadi_model.meas_lambdas:
                        casadi_model.D[t, l] = float(self._spectral_data[l][t])

            # Fixes parameters that were given numeric values
            for p, v in self._parameters.items():
                if v is not None:
                    casadi_model.P[p] = v
            # validate the model before writing constraints
            self._validate_data(casadi_model, start_time, end_time)
            # ignores the time indes t=0
            if self._odes:
                casadi_model.odes = self._odes(casadi_model, 0)
                for k in casadi_model.states:
                    if not k in casadi_model.odes.keys():
                        raise RuntimeError('Missing ode expresion for component {}'.format(k))

            if self._algebraic_constraints:
                alg_const = self._algebraic_constraints(casadi_model, 0)
                for c in alg_const:
                    casadi_model.alg_exprs.append(c)
            return casadi_model
        else:
            raise RuntimeError('Install casadi to create casadi models')

    @property
    def num_parameters(self):
        return len(self._parameters)

    @property
    def num_mixture_components(self):
        return len(self._component_names)

    @property
    def num_complementary_states(self):
        return len(self._complementary_states)

    @property
    def num_algebraics(self):
        return len(self._algebraics)

    @property
    def measurement_times(self):
        return self._meas_times

    @property #added with additional data CS
    def huplcmeasurement_times(self):
        return self._huplcmeas_times

    @property
    def feed_times(self):
        return self._feed_times  # added for feeding points CS

    @num_parameters.setter
    def num_parameters(self):
        raise RuntimeError('Not supported')

    @num_mixture_components.setter
    def num_mixture_components(self):
        raise RuntimeError('Not supported')

    @num_complementary_states.setter
    def num_complementary_states(self):
        raise RuntimeError('Not supported')

    @num_algebraics.setter
    def num_algebraics(self):
        raise RuntimeError('Not supported')

    @measurement_times.setter
    def measurement_times(self):
        raise RuntimeError('Not supported')

    @huplcmeasurement_times.setter
    def huplcmeasurement_times(self):
        raise RuntimeError('Not supported')

    @feed_times.setter
    def feed_times(self):
        raise RuntimeError('Not supported')  # added for feeding points CS

    def has_spectral_data(self):
        return self._spectral_data is not None

    def has_adsorption_data(self):
        return self._absorption_data is not None

    def has_concentration_data(self):
        return self._concentration_data is not None


    def optional_warmstart(self, model):
        if hasattr(model, 'dual') and hasattr(model, 'ipopt_zL_out') and hasattr(model, 'ipopt_zU_out') and hasattr(
                model, 'ipopt_zL_in') and hasattr(model, 'ipopt_zU_in'):
            print('warmstart model components already set up before.')
        else:
            model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

    def set_estinit_extra_species(self, model, initextra_est_list, check=True):#added for the estimation of initial conditions which have to be complementary state vars CS
        # type: (ConcreteModel, list, bool) -> None
        """Sets the non absorbing component of the model.

        Args:
            initextra_est_list: List of to be estimated initial conditions of complementary state variables.
            model: The corresponding model.
            check: Safeguard against setting this up twice.
        """
        if hasattr(model, 'estim_init'):
            print("To be estimated initial conditions of complementary state variables were already set up before.")
            return

        if (self._initextra_est_list and check):
            raise RuntimeError('To be estimated initial conditions of complementary state variables have been already set up.')

        self._initextraest = initextra_est_list
        self._estim_init = True
        model.estim_init=Param(initialize=self._estim_init)

    def add_init_extra(self, *args, **kwds):#added for the estimation of initial conditions which have to be complementary state vars CS
        """Add a kinetic parameter(s) to the model.

        Note:
            Plan to change this method add parameters as PYOMO variables

            This method tries to mimic a template implementation. Depending
            on the argument type it will behave differently

        Args:
            param1 (str): Species name. Creates a variable species

            param1 (list): Species names. Creates a list of variable species

            param1 (dict): Map species name(s) to value(s). Creates a fixed parameter(s)

        Returns:
            None

        """
        bounds = kwds.pop('bounds', None)
        init = kwds.pop('init', None)
        self._estim_init=True

        if len(args) == 1:
            name = args[0]
            if isinstance(name, six.string_types):
                self._initextraparams[name] = None
                if bounds is not None:
                    self._initextraparams_bounds[name] = bounds
                if init is not None:
                    self._initextraparams_init[name] = init
            elif isinstance(name, list) or isinstance(name, set):
                if bounds is not None:
                    if len(bounds) != len(name):
                        raise RuntimeError('the list of bounds must be equal to the list of species')
                for i, n in enumerate(name):
                    self._initextraparams[n] = None
                    if bounds is not None:
                        self._initextraparams_bounds[n] = bounds[i]
                    if init is not None:
                        self._initextraparams_init[n] = init[i]
            elif isinstance(name, dict):
                if bounds is not None:
                    if len(bounds) != len(name):
                        raise RuntimeError('the list of bounds must be equal to the list of species')
                for k, v in name.items():
                    self._initextraparams[k] = v
                    if bounds is not None:
                        self._initextraparams_bounds[k] = bounds[k]
                        print(bounds[k])
                    if init is not None:
                        self._initextraparams_init[k] = init[k]
            else:
                raise RuntimeError('Species data not supported. Try str')
        elif len(args) == 2:
            first = args[0]
            second = args[1]
            if isinstance(first, six.string_types):
                self._initextraparams[first] = second
                if bounds is not None:
                    self._initextraparams_bounds[first] = bounds
                if init is not None:
                    self._initextraparams_init[first] = init
            else:
                raise RuntimeError('Species argument not supported. Try str,val')
        else:
            raise RuntimeError('Species argument not supported. Try str,val')

    def set_non_absorbing_species(self, model, non_abs_list, check=True):
        # type: (ConcreteModel, list, bool) -> None
        """Sets the non absorbing component of the model.

        Args:
            non_abs_list: List of non absorbing components.
            model: The corresponding model.
            check: Safeguard against setting this up twice.
        """
        if hasattr(model, 'non_absorbing'):
            print("non-absorbing species were already set up before.")
            return

        if (self._is_non_abs_set and check):
            raise RuntimeError('Non absorbing species have been already set up.')

        self._is_non_abs_set = True
        self._non_absorbing = non_abs_list
        model.add_component('non_absorbing', Set(initialize=self._non_absorbing))

        C = getattr(model, 'C')
        Z = getattr(model, 'Z')

        times = getattr(model, 'meas_times')
        alltimes = getattr(model, 'allmeas_times')
        allcomps = getattr(model, 'mixture_components')

        model.add_component('fixed_C', ConstraintList())
        new_con = getattr(model, 'fixed_C')

        #############################
        # Exclude non absorbing species from S matrix and create subset Cs of C (CS):
        model.add_component('abs_components_names', Set())
        abscompsnames = [name for name in set(sorted(set(allcomps) - set(self._non_absorbing)))]
        model.add_component('abs_components', Set(initialize=abscompsnames))
        abscomps = getattr(model, 'abs_components')

        model.add_component('Cs', Var(times, abscompsnames))
        Cs = getattr(model, 'Cs')
        model.add_component('matchCsC', ConstraintList())
        matchCsC_con = getattr(model, 'matchCsC')

        for time in alltimes:
            for component in allcomps:
                new_con.add(C[time, component] == Z[time, component])
            for componenta in abscomps:
                if time in times:
                    matchCsC_con.add(Cs[time, componenta] == C[time, componenta])
        ##########################

    def set_known_absorbing_species(self, model, known_abs_list, absorbance_data, check=True):
        # type: (ConcreteModel, list, dataframe, bool) -> None
        """Sets the known absorbance profiles for specific components of the model.

        Args:
            knon_abs_list: List of known species absorbance components.
            model: The corresponding model.
            absorbance_data: the dataframe containing the known component spectra
            check: Safeguard against setting this up twice.
        """
        if hasattr(model, 'known_absorbance'):
            print("species with known absorbance were already set up before.")
            return

        if (self._is_non_abs_set and check):
            raise RuntimeError('Species with known absorbance were already set up before.')

        self._is_known_abs_set = True
        self._known_absorbance = known_abs_list
        self._known_absorbance_data = absorbance_data
        model.add_component('known_absorbance', Set(initialize=self._known_absorbance))
        S = getattr(model, 'S')
        C = getattr(model, 'C')
        Z = getattr(model, 'Z')
        lambdas = getattr(model, 'meas_lambdas')
        model.known_absorbance_data = self._known_absorbance_data
        for component in self._known_absorbance:
            for l in lambdas:
                S[l, component].set_value(self._known_absorbance_data[component][l])
                S[l, component].fix()
      #########################

        #added for additional huplc data CS
    def set_huplc_absorbing_species(self, model, huplc_abs_list, vol=None, solid_spec=None, check=True):
        # type: (ConcreteModel, list, bool) -> None
        """Sets the non absorbing component of the model.

        Args:
            huplc_abs_list: List of huplc absorbing components.
            vol: reactor volume
            solid_spec: solid species observed with huplc data
            model: The corresponding model.
            check: Safeguard against setting this up twice.
        """
        if hasattr(model, 'huplc_absorbing'):
            print("huplc-absorbing species were already set up before.")
            return

        if (self._is_huplc_abs_set and check):
            raise RuntimeError('huplc-absorbing species have been already set up.')


        self._is_huplc_abs_set = True
        self._vol=vol
        model.add_component('vol', Param(initialize=self._vol))


        self._huplc_absorbing = huplc_abs_list
        model.add_component('huplc_absorbing', Set(initialize=self._huplc_absorbing))
        alltime = getattr(model, 'alltime')
        times = getattr(model, 'huplcmeas_times')
        alltimes = getattr(model, 'allmeas_times')
        timesh = getattr(model, 'huplctime')
        alltime=getattr(model, 'alltime')
        #########
        model.add_component('huplcabs_components_names', Set())
        huplcabscompsnames = [name for name in set(sorted(self._huplc_absorbing))]
        model.add_component('huplcabs_components', Set(initialize=huplcabscompsnames))
        huplcabscomps = getattr(model, 'huplcabs_components')

        self._solid_spec=solid_spec
        if solid_spec is not None:
            solid_spec_arg1keys = [key for key in set(sorted(self._solid_spec.keys()))]
            model.add_component('solid_spec_arg1', Set(initialize=solid_spec_arg1keys))
            solid_spec_arg2keys = [key for key in set(sorted(self._solid_spec.values()))]
            model.add_component('solid_spec_arg2', Set(initialize=solid_spec_arg2keys))
            model.add_component('solid_spec', Var(model.solid_spec_arg1, model.solid_spec_arg2, initialize=self._solid_spec))

        C = getattr(model, 'C')
        Z = getattr(model, 'Z')

        Dhat_dict=dict()
        for k in self._huplc_data.columns:
            for c in self._huplc_data.index:
                Dhat_dict[c, k] = float(self._huplc_data[k][c])


        model.add_component('fixed_Dhat', ConstraintList())
        new_con = getattr(model, 'fixed_Dhat')

        model.add_component('Dhat', Var(times, huplcabscompsnames,initialize=Dhat_dict))
        Dhat = getattr(model, 'Dhat')
        model.add_component('Chat', Var(timesh, huplcabscompsnames,initialize=1))

        Chat = getattr(model, 'Chat')

        model.add_component('matchChatZ', ConstraintList())
        matchChatZ_con = getattr(model, 'matchChatZ')
