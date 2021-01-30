"""TemplateBuilder - handles inputs for generating the Pyomo model"""

# Standard library imports
import copy
import inspect
import itertools
import logging
import numbers
import six
import sys
import warnings

# Third party imports
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.dae import *
from pyomo.environ import units as u

# KIPET library imports
from kipet.library.post_model_build.scaling import scale_parameters
from kipet.library.core_methods.PyomoSimulator import PyomoSimulator
from kipet.library.top_level.variable_names import VariableNames
from kipet.library.common.pyomo_model_tools import get_index_sets
from kipet.library.common.VisitorClasses import ReplacementVisitor

logger = logging.getLogger('ModelBuilderLogger')


string_replace_dict = {
    ' ': '_',
    '-': 'n',
    '+': 'p',
    '.': '_',
    }

class ModalVar():
        
    def __init__(self, name, comp, index, model):
        
        self.name = name
        self.comp = comp
        self.index = index
        self.model = model
    
class Comp():
    
    var = VariableNames()
    
    def __init__(self, model):
        
        #self._r_model = model
        self._model = model
        self._model_vars = self.var.model_vars
        self._rate_vars = self.var.rate_vars
        self.var_dict = {}
        self.assign_vars()
        self.assign_rate_vars()
        
    def assign_vars(self):
        """Digs through and assigns the variables as top-level attributes
        
        """
        for mv in self._model_vars:
            if hasattr(self._model, mv):
                mv_obj = getattr(self._model, mv)
                index_sets = get_index_sets(mv_obj)
                comp_set = list(index_sets[-1].keys())
                dim = len(index_sets)
                
                for comp in comp_set:
                    
                    comp = str(comp)
                    if isinstance(comp, str):
                        if isinstance(comp[0], int):
                            comp.insert(0, 'y')
                        
                        comp_name = comp
                        for k, v in string_replace_dict.items():
                            comp_name.replace(k, v)
                    
                    self.var_dict[comp_name] = ModalVar(comp_name, comp_name, mv, self._model)
                    if dim > 1:
                        setattr(self, comp_name, mv_obj[0, comp])
                    else:
                        setattr(self, comp_name, mv_obj[comp])

    def assign_rate_vars(self):

        for mv in self._rate_vars:
            if hasattr(self._model, mv):
                
                mv_obj = getattr(self._model, mv)
                index_sets = get_index_sets(mv_obj)
                comp_set = list(index_sets[-1].keys())
                dim = len(index_sets)
                
                for comp in comp_set:
                    
                    if isinstance(comp, str):
                        comp_name = comp
                        comp_name.replace(' ', '_')
                    elif isinstance(comp, int):
                        comp_name = f'y{comp}'
                    
                    comp_name = f'd{comp_name}dt'
                    
                    self.var_dict[comp_name] = ModalVar(comp_name, str(comp), mv, self._model)
                    if dim > 1:
                        setattr(self, comp_name, mv_obj[0, comp])
                    else:
                        setattr(self, comp_name, mv_obj[comp])  

    def _id(self, comp):
        
        id_ = id(getattr(self, comp))
        print(id_)
        return id_
    
    @property
    def get_var_list(self):
        
        return list(self.var_dict.keys())
    

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
    __var = VariableNames()

    def __init__(self, **kwargs):
        """Template builder constructor

        Args:
            **kwargs: Arbitrary keyword arguments.
            concentrations (dictionary): map of component name to initial condition

            parameters (dictionary): map of parameter name to its corresponding value

            extra_states (dictionary): map of state name to initial condition

        """
        self._component_names = set()
        self._component_units = dict()
        self._parameters = dict()
        self._parameters_init = dict()  # added for parameter initial guess CS
        self._parameters_bounds = dict()
        self._parameters_fixed = dict()
        self._parameters_units = dict()
        self._constants = dict()
        self._constants_units = dict()
        
        self.u = u
        
        self._smoothparameters = dict()  # added for mutable parameters CS
        self._smoothparameters_mutable = dict() #added for mutable parameters CS

        #added for initial condition parameter estimates: CS
        self._initextraparams = dict()
        self._initextraparams_init = dict()  # added for parameter initial guess CS
        self._initextraparams_bounds = dict()

        # TODO: Put all data into it's own structure (attrs?)

        self._y_bounds = dict()  # added for additional optional bounds CS
        self._y_init = dict()
        self._prof_bounds = list()  # added for additional optional bounds MS
        self._init_conditions = dict()
        self._spectral_data = None
        self._concentration_data = None
        self._complementary_states_data = None # added for complementary state data (Est.) KM
        self._custom_data = None
        self._custom_objective = None
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
        # self._estim_init = False
        # self._allinitcomponents=dict()
        # self._initextra_est_list = None
        
        # bounds and init for unwanted contributions KH.L
        self._G_contribution = None
        
        self._qr_bounds = None
        self._qr_init = None
        self._g_bounds = None
        self._g_init = None

        self._add_dosing_var = False

        
    def add_state_variance(self, sigma_dict):
        """Provide a variance for the measured states
        
        Args:
            sigma (dict): A dictionary of the measured states with known 
            variance. Provide the value of sigma (standard deviation).
            
        Returns:
            None
        """
        self._state_sigmas = sigma_dict #{'A': 0.0001, 'T': 0.0625}
        # perhaps make this more secure later on and account for different input types
        return None

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
        
    """
    Adding in the model elements
    """  
    
    def add_constants(self, ConstantBlockObject):
        
        self.template_constant_data = ConstantBlockObject
        
        for constant in ConstantBlockObject:
            self.add_constant(constant)
    
    def add_parameters(self, ParameterBlockObject):
        
        self.template_parameter_data = ParameterBlockObject
        
        # for param in ParameterBlockObject:
        #     self.add_parameter(param)

    def add_components(self, ComponentBlockObject):
        """Add model components to template using the ComponentBlock object
        
        Args:
            ComponentBlockObject (ComponentBlock): Model components
        """
        sigma_dict = {}
        known_dict = ComponentBlockObject.known_values
        
        self.template_component_data = ComponentBlockObject
        
        for component in ComponentBlockObject:       
            if component.state == 'trajectory':
                continue       
            self._add_state_variable(component)
            
            sigma_dict[component.name] = component.variance
            
        self._state_sigmas = sigma_dict
        self._init_known = known_dict
  
    
    def add_constant(self, ModelConstantObject):
        """Add a kinetic parameter(s) to the model.

        Args:
            model_constant (ModelConstant): constant object

        Returns:
            None
        """
        const = ModelConstantObject
        self._constants[const.name] = const.value
        # if const.units is not None:
        self._constants_units[const.name] = self.u._pint_registry(const.units)

    # def add_parameter(self, ModelParameterObject):
    #     """Add a kinetic parameter(s) to the model.

    #     Args:
    #         model_constant (ModelConstant): constant object

    #     Returns:
    #         None
    #     """
    #     param = ModelParameterObject
        
    #     # All of this can be removed and the parameter object used instead
    #     # Can this be done with all cases?
    #     self._parameters[param.name] = None
    #     self._parameters_bounds[param.name] = param.bounds
    #     self._parameters_init[param.name] = param.value
    #     self._parameters_fixed[param.name] = param.fixed
    #     self._parameters_units[param.name] = self.u._pint_registry(param.units)

    def add_smoothparameter(self, *args, **kwds):
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

    def _add_state_variable(self, component, data_type=None):
        
        built_in_data_types = {
            'concentration' : ('component_names', 'Mixture component'),
            'complementary_states' : ('complementary_states', 'Complementary state'),
           }
        
        print(component)
        self._init_conditions[component.name] = component.value
        getattr(self, f'_{built_in_data_types[component.state][0]}').add(component.name)
        self._component_units[component.name] = self.u._pint_registry(component.units)
        
        # Put these checks into the component object class
        # if len(args) == 1:
        #     input = args[0]
        #     if isinstance(input, dict):
        #         for key, val in input.items():
        #             if not isinstance(val, numbers.Number):
        #                 raise RuntimeError('The init condition must be a number. Try str, float')
        #             getattr(self, f'_{built_in_data_types[data_type][0]}').add(key)
        #             self._init_conditions[key] = val
        #     else:
        #         raise RuntimeError(f'{built_in_data_types[data_type][1]} data not supported. Try dict[str]=float')
        
        # elif len(args) == 2:
        #     name = args[0]
        #     init_condition = args[1]

        #     if not isinstance(init_condition, numbers.Number):
        #         raise RuntimeError('The second argument must be a number. Try str, float')

        #     if isinstance(name, six.string_types):
        #         getattr(self, f'_{built_in_data_types[data_type][0]}').add(name)
        #         self._init_conditions[name] = init_condition
        #     else:
        #         raise RuntimeError(f'{built_in_data_types[data_type][1]} data not supported. Try str, float')
                
        # elif len(args) == 3:
        #     name = args[0]
        #     init_condition = args[1]
        #     units = args[2]

        #     if not isinstance(init_condition, numbers.Number):
        #         raise RuntimeError('The second argument must be a number. Try str, float')

        #     if isinstance(name, six.string_types):
        #         getattr(self, f'_{built_in_data_types[data_type][0]}').add(name)
        #         self._init_conditions[name] = init_condition
        #     else:
        #         raise RuntimeError(f'{built_in_data_types[data_type][1]} data not supported. Try str, float')
            
        #     # from pyomo.environ import units as u
        #     # self.u = u
        #     self._component_units[name] = self.u._pint_registry(units)
             
        # else:
        #     raise RuntimeError(f'{built_in_data_types[data_type][1]} data not supported. Try str, float')

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
    
    def input_data(self, DataBlockObject):
        """This should take a DataContainer and input the data, but for now
        it will take a DataBlock list
        
        """
        for db in DataBlockObject:
            if db.category == 'state':     
                self.add_data(db.data, 'complementary_states')
            elif db.category == 'trajectory':
                continue
            else:
                self.add_data(db.data, db.category)
        
        self.datablock = DataBlockObject
        
        return None
    
    def clear_data(self):
        
        self._spectral_data = None
        self._concentration_data = None
        self._complementary_states_data = None # added for complementary state data (Est.) KM
        self._huplc_data = None #added for additional data CS
        self._smoothparam_data = None  # added for additional smoothing parameter data CS
        self._absorption_data = None
        
        return None
    
    def add_data(self, data, data_type=None, label=None):
        
        if data_type is not None:
            self._add_state_data(data, data_type, label=label)
        
        else:
            self.add_experimental_data(data)

        return None

    def _add_state_data(self, data, data_type, label=None, overwrite=True):
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
            'concentration' : self.__var.concentration_measured,
            'complementary_states' : self.__var.state,
            'spectral' : self.__var.spectra_data,
            'huplc' : self.__var.huplc_data,
            'smoothparam' : self.__var.smooth_parameter,
            'custom' : self.__var.user_defined,
            }
        
        state_data = [
            self.__var.concentration_spectra,
            self.__var.state,
            self.__var.concentration_measured,
            self.__var.user_defined,
            ]
            
        deriv_data = [
            self.__var.concentration_spectra,
            self.__var.state,
            self.__var.spectra_data,
            self.__var.huplc_data,
            self.__var.concentration_measured,
            self.__var.user_defined,
            ]
        
        if label is None:
            try:
                label = built_in_data_types[data_type]
            except:
                raise ValueError("You need to provide a label for custom data types")
        
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        
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

            dfallc = dfallc.dropna(how='all')

            if not overwrite:
                if hasattr(self, f'_{data_type}_data'):
                    df_data = getattr(self, f'_{data_type}_data')
                    #print(f'df_data:\n{df_data}')
                    df_data = pd.concat([df_data, dfallc], axis=1)
                    setattr(self, f'_{data_type}_data', df_data)
            else:
                setattr(self, f'_{data_type}_data', dfallc)
            
            if label in state_data:
                self._all_state_data += list(data.columns)
        else:
            raise RuntimeError(f'{data_type.capitalize} data format not supported. Try pandas.DataFrame')
        
        if label in deriv_data:
            C = np.array(dfallc)
            setattr(self, f'_is_{label}_deriv', False)
            for t in range(len(dfallc.index)):
                for l in range(len(dfallc.columns)):
                    if C[t, l] >= 0 or np.isnan(C[t, l]):
                        pass
                    else:
                        setattr(self, f'_is_{label}_deriv', True)
            #print(label, hasattr(self, f'_is_{label}_deriv'))
            if getattr(self, f'_is_{label}_deriv') == True:
                print(
                    f"Warning! Since {label}-matrix contains negative values Kipet is assuming a derivative of {label} has been inputted")

        return None
    
    def add_experimental_data(self, data):
        """Generic function to add all data at once if in the same dataframe.
        At the moment this works for concentration and complementary states
        
        Use this for mixed and missing data
        """
        exp_data = pd.DataFrame(data)
        
        conc_state_headers = self._component_names & set(exp_data.columns)
        if len(conc_state_headers) > 0:
            for i, c in enumerate(conc_state_headers):
                overwrite = True if i == 0 else False
                self.add_concentration_data(pd.DataFrame(exp_data[c].dropna()), overwrite=overwrite)
                
        comp_state_headers = self._complementary_states & set(exp_data.columns)
        if len(comp_state_headers) > 0:
            for i, c in enumerate(comp_state_headers):
                overwrite = True if i == 0 else False
                self.add_complementary_states_data(pd.DataFrame(exp_data[c].dropna()), overwrite=overwrite)
        
        return None
    
    def add_user_defined_data(self, data, overwrite=True):
        """Add concentration data as a wrapper to _add_state_data

        Args:
            data (DataFrame): DataFrame with measurement times as
                              indices and concentrations as columns.

        Returns:
            None

        """
        self._add_state_data(data,
                             data_type='custom',
                             overwrite=overwrite)
        
        return None
        
    def add_concentration_data(self, data, overwrite=True):
        """Add concentration data as a wrapper to _add_state_data

        Args:
            data (DataFrame): DataFrame with measurement times as
                              indices and concentrations as columns.

        Returns:
            None

        """
        self._add_state_data(data,
                             data_type='concentration',
                             overwrite=overwrite)
        
        return None
        
    def add_complementary_states_data(self, data, overwrite=True):
        """Add complementary state data as a wrapper to _add_state_data

        Args:
            data (DataFrame): DataFrame with measurement times as
                              indices and complmentary states as columns.

        Returns:
            None

        """
        self._add_state_data(data,
                             data_type='complementary_states',
                             overwrite=overwrite)
                           
        return None
    
    def add_spectral_data(self, data, overwrite=True):
        """Add spectral data as a wrapper to _add_state_data

        Args:
            data (DataFrame): DataFrame with measurement times as
                              indices and wavelengths as columns.

        Returns:
            None

        """
        self._add_state_data(data,
                             data_type='spectral',
                             overwrite=overwrite)
        
    def add_huplc_data(self, data, overwrite=True): #added for the inclusion of h/uplc data CS
        """Add HPLC or UPLC data as a wrapper to _add_state_data

                Args:
                    data (DataFrame): DataFrame with measurement times as
                                      indices and wavelengths as columns.

                Returns:
                    None
        """
        self._add_state_data(data,
                             data_type='huplc',
                             overwrite=overwrite)

    def add_smoothparam_data(self, data, overwrite=True): #added for mutable smoothing parameter option CS
        """Add smoothing parameters as a wrapper to _add_state_data

                Args:
                    data (DataFrame): DataFrame with measurement times as
                                      indices and wavelengths as columns.

                Returns:
                    None
        """
        self._add_state_data(data,
                             data_type='smoothparam',
                             overwrite=overwrite)


    def add_absorption_data(self, data, overwrite=True):
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
        init = kwds.pop('init', None)
        bounds = kwds.pop('bounds', None)

        if len(args) == 1:
            name = args[0]
            if isinstance(name, six.string_types):
                self._algebraics[name] = None
                if bounds is not None:
                    self._y_bounds[name] = bounds
                if init is not None:
                    self._y_init[name] = init
            elif isinstance(name, list) or isinstance(name, set):
                if bounds is not None:
                    if len(bounds) != len(name):
                        raise RuntimeError('the list of bounds must be equal to the list of parameters')
                for i, n in enumerate(name):
                    self._algebraics[n] = None
                    if bounds is not None:
                        self._y_bounds[n] = bounds[i]
                    if init is not None:
                        self._y_init[n] = init[i]
            else:
                raise RuntimeError('To add an algebraic please pass name')
                
        return None

    # read bounds and initialize for qr and g (unwanted contri variables) from users KH.L
    def add_qr_bounds_init(self, **kwds):
        bounds = kwds.pop('bounds', None)
        init = kwds.pop('init', None)
        
        self._qr_bounds = bounds
        self._qr_init = init
        
        return None
        
    def add_g_bounds_init(self, **kwds):
        bounds = kwds.pop('bounds', None)
        init = kwds.pop('init', None)
        
        self._g_bounds = bounds
        self._g_init = init
    
        return None
    
    def add_dosing_var(self, n_steps):
        
        self._add_dosing_var = True
        self._number_of_steps = n_steps
    
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
        
        return None

    def set_algebraics_rule(self, rule, asdict=False):
        """Set the algebraic expressions.

        Defines the algebraic equations for the system

        Args:
            rule (function): Python function that returns a list of tuples

        Returns:
            None

        """
        if not asdict:
            inspector = inspect.getargspec(rule)
            if len(inspector.args) != 2:
                raise RuntimeError('The rule should have two inputs')
            self._algebraic_constraints = rule
        else:
            self._algebraic_constraints = rule
            self._use_alg_dict = True
        
        
        return None

    def set_objective_rule(self, algebraic_vars):
        """Set the algebraic expressions.

        Defines the algebraic equations for the system

        Args:
            algebraic_vars (str, list): str or list of algebraic variables

        Returns:
            None

        """
        if not isinstance(algebraic_vars, list):
            algebraic_vars = list(algebraic_vars)
        self._custom_objective = algebraic_vars
        
        return None

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

    def _validate_data(self, model):
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
                dummy_balances = self._odes(model, model.start_time)
                if len(self._component_names) + len(self._complementary_states) != len(dummy_balances):
                    print(
                        'WARNING: The number of ODEs is not the same as the number of state variables.\n If this is the desired behavior, some odes must be added after the model is created.')

            else:
                print(
                    'WARNING: differential expressions not specified. Must be specified by user after creating the model')

        if self._algebraics:
            if self._algebraic_constraints and not isinstance(self._algebraic_constraints, dict):
                dummy_balances = self._algebraic_constraints(model, model.start_time)
                if len(self._algebraics) != len(dummy_balances):
                    print(
                        'WARNING: The number of algebraic equations is not the same as the number of algebraic variables.\n If this is the desired behavior, some algebraics must be added after the model is created.')
            else:
                print(
                    'WARNING: algebraic expressions not specified. Must be specified by user after creating the model')

        if self._absorption_data is not None:
            if not self._meas_times:
                raise RuntimeError('Need to add measurement times')


    def _add_algebraic_var(self, model):
        """If algebraics are present, add the algebraic variable to the model
        """
        if len(self._algebraics) > 0:
            
            # if 'init' in self._y_bounds:
            #     bounds = self._y_bounds[]
            
            setattr(model, self.__var.algebraic, Var(model.alltime,
                                                     model.algebraics,
                                                     initialize=1.0))
                                                    #bounds=())
            
            for t in model.alltime:
                for k, v in self._y_bounds.items():
                    lb = v[0]
                    ub = v[1]
                    getattr(model, self.__var.algebraic)[t, k].setlb(lb)
                    getattr(model, self.__var.algebraic)[t, k].setub(ub)
                    
                # for k, v in self._y_init.items():
                #     model.Y[0, k].set_value(v)
        
        return None
    
    def _add_initial_conditions(self, model):
        """Set up the initial conditions for the model"""
        
        model.init_conditions = Var(model.states,
                                    initialize=self._init_conditions,
                                    )
        unknown_init = {}
        
        for var, obj in model.init_conditions.items():
            if self.template_component_data[var].known:
                obj.fix()
            else:
                lb = self.template_component_data[var].bounds[0]
                ub = self.template_component_data[var].bounds[1]
                init = self.template_component_data[var].init
                obj.setlb(lb)
                obj.setub(ub)
                unknown_init[var] = init
        
        if len(unknown_init) > 0:
            model._unknown_init_set = Set(initialize=list(unknown_init.keys()))
            model.Pinit = Var(model._unknown_init_set,
                              initialize=unknown_init)
            
            model.del_component('P_all')    
            model.P_all = Set(initialize=model.parameter_names | model._unknown_init_set,
                                 ordered=True)

        def rule_init_conditions(m, k):
            if k in m.mixture_components:
                return getattr(m, self.__var.concentration_model)[m.start_time, k] - m.init_conditions[k] == 0
            else:
                return getattr(m, self.__var.state_model)[m.start_time, k] - m.init_conditions[k] == 0

        model.init_conditions_c = \
            Constraint(model.states, rule=rule_init_conditions)
            
        if hasattr(model, 'Pinit'):
            
            def rule_Pinit_conditions(m, k):
                if k in m.mixture_components:
                    return m.Pinit[k] - m.init_conditions[k] == 0
                else:
                    return m.Pinit[k] - m.init_conditions[k] == 0
    
            model.Pinit_conditions_c = \
                Constraint(model._unknown_init_set, rule=rule_Pinit_conditions)

        return None

    def _add_model_variables(self, model):
        """Adds the model variables to the pyomo model"""
        
        model_pred_var_name = {
                self.__var.concentration_model : model.mixture_components,
                self.__var.state_model : model.complementary_states,
                    }
        
        for var, model_set in model_pred_var_name.items():
            
            if model_set is None or len(model_set) == 0:
                continue
            
            # Check if any data for the variable type exists (Pyomo 5.7 update)
            if hasattr(model_set, 'ordered_data') and len(model_set.ordered_data()) == 0:
                continue
            
            setattr(model, var, Var(model.alltime,
                                          model_set,
                                          # bounds=(0.0,None),
                                          #units=self._component_units,
                                          initialize=1) 
                    )    
        
            for time, comp in getattr(model, var):
                if time == model.start_time.value:
                    getattr(model, var)[time, comp].value = self._init_conditions[comp]
                   
            setattr(model, f'd{var}dt', DerivativeVar(getattr(model, var),
                                                     # units=self.u.mol/self.u.l,
                                                      wrt=model.alltime)
                    )

        # Variables of provided data - set as fixed variables complementary to above
        fixed_var_name = {
               # 'C' : self._spectral_data,
                self.__var.concentration_measured : self._concentration_data,
                self.__var.state : self._complementary_states_data,
                self.__var.user_defined : self._custom_data,
                    }
        
        for var, data in fixed_var_name.items():
            
            if data is None or len(data) == 0:
                continue
            
            c_dict = dict()
            if hasattr(self, f'_is_{var}_deriv'):
                if getattr(self, f'_is_{var}_deriv') == True:
                    c_bounds = (None, None)
                else:
                    c_bounds = (0.0, None)
                
                if data is not None:    
                    for i, row in data.iterrows():
                         c_dict.update({(i, col): float(row[col]) for col in data.columns if not np.isnan(float(row[col]))})
                
                    setattr(model, f'{var}_indx', Set(initialize=list(c_dict.keys()), ordered=True))
                    
                    # setattr(model, var, Var(model.allmeas_times,
                    #                 model.mixture_components,
                    #                 bounds=c_bounds,
                    #                 initialize=c_dict))
                    
                    setattr(model, var, Var(getattr(model, f'{var}_indx'),
                                                  bounds=c_bounds,
                                                  initialize=c_dict,
                                                  #units=self._component_units,
                                                  )
                            )
                    
                    for k, v in getattr(model, var).items():
                        getattr(model, var)[k].fixed = True
                
                else:
                    setattr(model, var, Var(model.allmeas_times,
                                    model.mixture_components,
                                    bounds=c_bounds,
                                    initialize=1))
        
                    for time, comp in getattr(model, var):
                        if time == model.start_time.value:
                          #  print(f'initial values: {time}, {comp}')
                            getattr(model, var)[time, comp].value = self._init_conditions[comp]

        return None
    
    def _add_model_constants(self, model):
        
        if len(self._constants) > 0:
        
            constant_designator = self.__var.model_constant
            setattr(model, constant_designator, Param(self._constants.keys(),
                                                      initialize=self._constants))
            
        return None
    
    
    def _add_model_parameters(self, model):
        """Add the model parameters to the pyomo model"""
        
        p_info = self.template_parameter_data
        
        print(model.parameter_names.display())
        
        # Initial parameter values
        p_values = p_info.as_dict('value')
        for param, value in p_values.items():
            if value is None:
                if p_info[param].bounds[0] is not None and p_info[param].bounds[1] is not None:
                    p_values[param] = sum(p_info[param].bounds)/2
                    
        if self._scale_parameters:
            setattr(model, self.__var.model_parameter, Var(model.parameter_names,
                                                           bounds = (0.1, 10),
                                                           initialize=1))

        else:
            setattr(model, self.__var.model_parameter, Var(model.parameter_names,
                            initialize=p_values))

        # Set the bounds
        p_bounds = p_info.as_dict('bounds')
        for k, v in p_bounds.items():
            factor = 1
            if self._scale_parameters:
                factor = p_values[k]
                
            print(p_info)

            if p_info[param].lb is not None:
                lb = p_info[param].lb/factor
                getattr(model, self.__var.model_parameter)[k].setlb(lb)

            if p_info[param].ub is not None:
                ub = p_info[param].ub/factor
                getattr(model, self.__var.model_parameter)[k].setub(ub)
                
        #for optional smoothing parameters (CS):
        if isinstance(self._smoothparameters, dict) and self._smoothparam_data is not None:
            ps_dict = dict()
            for k in self._smoothparam_data.columns:
                for c in model.allsmooth_times:
                    ps_dict[c, k] = float(self._smoothparam_data[k][c])

            ps_dict2 = dict()
            for p in self._smoothparameters.keys():
                for t in model.alltime:
                    if t in model.allsmooth_times:
                        ps_dict2[t, p] = float(ps_dict[t, p])
                    else:
                        ps_dict2[t, p] = 0.0

            model.Ps = Param(model.alltime, model.smoothparameter_names, initialize=ps_dict2, mutable=True, default=20.)#here just set to some value that is noc

        # Fix parameters declared as fixed
        p_fixed = p_info.as_dict('fixed')
        for param, v in p_fixed.items():
            getattr(model, self.__var.model_parameter)[param].fixed = v

        return None

    def _add_unwanted_contribution_variables(self, model):
        """Add the bounds for the unwanted contributions, if any"""
        
        if self._qr_bounds is not None:
            qr_bounds = self._qr_bounds
        elif self._qr_bounds is None:
            qr_bounds = None
            
        if self._qr_init is not None:
            qr_init = self._qr_init
        elif self._qr_init is None:
            qr_init = 1.0
            
        model.qr = Var(model.alltime, bounds=qr_bounds, initialize=qr_init)
        
        if self._g_bounds is not None:
            g_bounds = self._g_bounds
        elif self._g_bounds is None:
            g_bounds = None
            
        if self._g_init is not None:
            g_init = self._g_init
        elif self._g_init is None:
            g_init = 0.1
            
        setattr(model, self.__var.unwanted_contribution, Var(model.meas_lambdas, 
                                                             bounds=g_bounds, 
                                                             initialize=g_init))
        
        return None

    # @staticmethod
    def change_time(self, expr_orig, c_mod, new_time, current_model):
        """Method to remove the fixed parameters from the ConcreteModel
        """
        #print(f'TO change:  {expr_orig.to_string()}')

        expr_new_time = expr_orig

        var_dict = c_mod.var_dict

        for model_var, var_obj in var_dict.items():
            
            if getattr(current_model, var_obj.index).dim() == 1:
                old_var = getattr(c_mod, model_var)
                new_var = getattr(current_model, var_obj.index)[var_obj.comp]
            
            else:
                old_var = getattr(c_mod, model_var)
                new_var = getattr(current_model, var_obj.index)[new_time, var_obj.comp]
            
            # print(old_var)
            # print(new_var)
            
            expr_new_time = self._update_expression(expr_new_time, old_var, new_var)
    
        #print(f'Changed TO: {expr_new_time.to_string()}')
        
        return expr_new_time

    @staticmethod
    def _update_expression(expr, replacement_param, change_value):
        """Takes the non-estiambale parameter and replaces it with its intitial
        value
        
        Args:
            expr (pyomo constraint expr): the target ode constraint
            
            replacement_param (str): the non-estimable parameter to replace
            
            change_value (float): initial value for the above parameter
            
        Returns:
            new_expr (pyomo constraint expr): updated constraints with the
                desired parameter replaced with a float
        
        """
        visitor = ReplacementVisitor()
        visitor.change_replacement(change_value)
        visitor.change_suspect(id(replacement_param))
        new_expr = visitor.dfs_postorder_stack(expr)       
        return new_expr

    def _add_model_odes(self, model):
        """Adds the ODE system to the model, if any"""
        
        if hasattr(self, 'reaction_dict'):

            def rule_odes(m, t, k):
                exprs = self.reaction_dict                
               
                if t == m.start_time.value:
                    return Constraint.Skip
                else:
                    if k in m.mixture_components:
                        if k in exprs.keys():
                            deriv_var = f'd{self.__var.concentration_model}dt'
                            final_expr = getattr(m, deriv_var)[t, k] == exprs[k]
                            final_expr  = self.change_time(final_expr, self.c_mod, t, m)
                            return final_expr
                        else:
                            return Constraint.Skip
                    else:
                        if k in exprs.keys():
                            deriv_var = f'd{self.__var.state_model}dt'
                            final_expr = getattr(m, deriv_var)[t, k] == exprs[k]
                            final_expr  = self.change_time(final_expr, self.c_mod, t, m)
                            return final_expr
                        else:
                            return Constraint.Skip

            setattr(model, self.__var.ode_constraints, Constraint(model.alltime,
                                                                  model.states,
                                                                  rule=rule_odes))

        elif self._odes:
            def rule_odes(m, t, k):
                exprs = self._odes(m, t)
            
                if t == m.start_time.value:
                    return Constraint.Skip
                else:
                    if k in m.mixture_components:
                        if k in exprs.keys():
                            deriv_var = f'd{self.__var.concentration_model}dt'
                            return getattr(m, deriv_var)[t, k] == exprs[k]
                        else:
                            return Constraint.Skip
                    else:
                        if k in exprs.keys():
                            deriv_var = f'd{self.__var.state_model}dt'
                            return getattr(m, deriv_var)[t, k] == exprs[k]
                        else:
                            return Constraint.Skip

            setattr(model, self.__var.ode_constraints, Constraint(model.alltime,
                                                                  model.states,
                                                                  rule=rule_odes))
        return None
    
    def _add_algebraic_constraints(self, model):
        """Adds the algebraic constraints the model, if any"""
        
        if self._algebraic_constraints:
            if hasattr(self, 'reaction_dict') or hasattr(self, '_use_alg_dict') and self._use_alg_dict:
                n_alg_eqns = self._algebraic_constraints.keys()
                def rule_algebraics(m, t, k):
                    alg_const = self._algebraic_constraints[k]
                    alg_var = getattr(m, self.__var.algebraic)[t, k]
                    final_expr = alg_var - alg_const == 0.0
                    final_expr  = self.change_time(final_expr, self.c_mod, t, m)
                    return final_expr
            else:
                n_alg_eqns = range(len(self._algebraic_constraints(model, model.start_time)))
                def rule_algebraics(m, t, k):
                      
                    alg_const = self._algebraic_constraints(m, t)[k]
                    final_expr = alg_const == 0.0
                    return final_expr
    
            model.algebraic_consts = Constraint(model.alltime,
                                                n_alg_eqns,
                                                rule=rule_algebraics)
        return None
    
    def _add_objective_custom(self, model):
        """Adds the custom objectives, if any and uses the model variable
        UD for the data. This is where the custom data is stored.
        
        """
        def custom_obj(m, t, y):
            exprs = getattr(m, self.__var.user_defined)[t, y] - getattr(m, self.__var.algebraic)[t, y]
            return exprs**2
        
        obj = 0
        var_name = []
        for index, values in model.UD.items():
            if index[1] not in self._custom_objective:
                continue
            if index[1] not in var_name:
                var_name.append(index[1])
            obj += custom_obj(model, index[0], index[1])
            
        model.custom_obj = obj
       
        return None

    def _add_spectral_variables(self, model):
        """Add D and C variables for the spectral data"""
        
        if self._spectral_data is not None:
            s_data_dict = dict()
            for t in model.meas_times:
                for l in model.meas_lambdas:
                    if t in model.meas_times:
                        s_data_dict[t, l] = float(self._spectral_data[l][t])
                    else:
                        s_data_dict[t, l] = float('nan')

            setattr(model, self.__var.spectra_data, Param(model.meas_times,
                                                          model.meas_lambdas,
                                                          initialize=s_data_dict))
            
            setattr(model, self.__var.concentration_spectra, Var(model.meas_times,
                                                                 model.mixture_components,
                                                                 bounds=(0, None),
                                                                 initialize=1))
            
        return None

    def _check_absorbing_species(self, model):
        """Set up the appropriate S depending on absorbing species"""
        
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
        
        if self.has_spectral_data():    
        
            if self._is_non_abs_set:
                self.set_non_absorbing_species(model, self._non_absorbing, check=False)
                setattr(model, self.__var.spectra_species, Var(model.meas_lambdas,
                                                               model.abs_components,
                                                               bounds=s_bounds,
                                                               initialize=s_dict))
            else:
                setattr(model, self.__var.spectra_species, Var(model.meas_lambdas,
                                                               model.mixture_components,
                                                               bounds=s_bounds,
                                                               initialize=s_dict))

            if self._absorption_data is not None:
                for l in model.meas_lambdas:
                    for k in model.mixture_components:
                        getattr(model, self.__var.spectra_species)[l, k].fixed = True
                        
        return None
    
    def _apply_bounds_to_variables(self, model):
        """User specified bounds to certain model variables are added here"""
        
        for bound_set in self._prof_bounds:
            var = bound_set[0]
            component_name = bound_set[1]
            if bound_set[2] is not None:
                bound_time_start = bound_set[2][0]
                bound_time_end = bound_set[2][1]
            upper_bound = bound_set[3][1]
            lower_bound = bound_set[3][0]
            
            for time, comp in getattr(model, var):
                if component_name == comp or component_name is None:
                    if bound_set[2] is not None:
                        if time >= bound_time_start and time < bound_time_end:
                            getattr(model, var)[time, comp].setlb(lower_bound)
                            getattr(model, var)[time, comp].setub(upper_bound)
                    else:
                        getattr(model, var)[time, comp].setlb(lower_bound)
                        getattr(model, var)[time, comp].setub(upper_bound)
                        
        return None

    def _add_model_smoothing_parameters(self, model):
        """Adds smoothing parameters to the model, if any.
        These are optional smoothing parameter values (mutable) read from a file
        
        """
        if self._smoothparam_data is not None:
            model.smoothparameter_names = Set(initialize=self._smoothparameters.keys()) #added for mutable parameters
            model.smooth_param_datatimes = Set(initialize=sorted(self._smoothparam_data.index))
            
            help_dict=dict()
            for k in self._smoothparam_data.index:
                for j in self._smoothparam_data.columns:
                    help_dict[k, j]=float(self._smoothparam_data[j][k])
            
            model.smooth_param_data = Param(self._smoothparam_data.index, sorted(self._smoothparam_data.columns), initialize=help_dict)

        return None

    def _set_up_times(self, model, start_time, end_time):
        
        if self._times is not None:
            if start_time is None:
                start_time = self._times[0]
            if end_time is None:
                end_time = self._times[1]
        else:
            if start_time is None and end_time is None:
                try:    
                    start_time = 0 #self.datablock.time_span[0]
                    end_time = self.datablock.time_span[1]
                except:
                    raise ValueError('A model requires a start and end time or a dataset')
        
        list_times = self._meas_times
        m_times = sorted(list_times)
        list_feedtimes = self._feed_times  # For inclusion of discrete feeds CS
        feed_times = sorted(list_feedtimes)  # For inclusion of discrete feeds CS
        m_lambdas = list()
        m_alltimes = m_times
        conc_times = list()

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

        if self._complementary_states_data is not None:
            list_times = list_times.union(set(self._complementary_states_data.index))
            list_comps = list(self._complementary_states_data.columns)
            m_times = sorted(list_times)
            m_alltimes = sorted(list_times)
            m_comps = sorted(list_comps)
            
        if self._custom_data is not None:
            list_times = list_times.union(set(self._custom_data.index))
            list_custom = list(self._custom_data.columns)
            m_times = sorted(list_times)
            m_alltimes = sorted(list_times)
            m_custom = sorted(list_custom)

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
                self._check_time_inputs(m_huplctimes, start_time, end_time)
                
        if self._smoothparam_data is not None:
            if m_allsmoothtimes:
                self._check_time_inputs(m_allsmoothtimes, start_time, end_time)
            model.allsmooth_times = Set(initialize=m_allsmoothtimes, ordered=True)

        if m_alltimes:
            self._check_time_inputs(m_alltimes, start_time, end_time)

        self._m_lambdas = m_lambdas

        # For inclusion of discrete feeds CS
        if self._feed_times is not None:
            list_feedtimes = list(self._feed_times)
            feed_times = sorted(list_feedtimes)

        if self._huplc_data is not None:
            model.huplcmeas_times = Set(initialize=m_huplctimes, ordered=True)
            model.huplctime = ContinuousSet(initialize=model.huplcmeas_times,
                                                  bounds=(start_time, end_time))
            
        model.allmeas_times = Set(initialize=m_alltimes, ordered=True) #add for new data structure CS
        model.meas_times = Set(initialize=m_times, ordered=True)
        model.feed_times = Set(initialize=feed_times, ordered=True)  # For inclusion of discrete feeds CS
        model.meas_lambdas = Set(initialize=m_lambdas, ordered=True)
        model.alltime = ContinuousSet(initialize=model.allmeas_times,
                                      bounds=(start_time, end_time)) #add for new data structure CS
        
        model.start_time = Param(initialize=start_time)
        model.end_time = Param(initialize=end_time)
        
        return None
    
    @staticmethod
    def _check_time_inputs(time_set, start_time, end_time):
        """Checks the first and last time of a measurement to see if it's in
        the model time bounds
        
        """
        print(time_set)
        print(start_time, end_time)
        
        if time_set[0] < start_time:
            raise RuntimeError(f'Measurement time {time_set[0]} not within ({start_time}, {end_time})')
        if time_set[-1] > end_time:
            raise RuntimeError(f'Measurement time {time_set[-1]} not within ({start_time}, {end_time})')
    
        return None
    
    def add_step_vars(self, step_data):
        
        if not hasattr(self, '_step_data'):
            self._step_data = None
        self._step_data = step_data
        return None
    
    def _add_time_steps(self, model):
        
        from kipet.library.common.model_funs import step_fun
    
        # Add dosing var
        #if self._add_dosing_var:
        setattr(model, self.__var.dosing_variable, Var(model.alltime,
                                                      [self.__var.dosing_component],
                                                      initialize=0))
            
        if hasattr(self, '_step_data') and len(self._step_data) > 0:
            steps = list(self._step_data.keys())
            
            setattr(model, self.__var.time_step_change, Var(steps, 
                    bounds=(model.start_time, model.end_time), 
                    initialize=(model.end_time - model.start_time)/2))
                   
            
            setattr(model, self.__var.step_variable, Var(model.alltime,
                                                         steps,
                                                         initialize=1.0))
                                          
            #Constraint to keep track of the step values (for diagnostics)
            def rule_step_function(m, t, k):
                step_const = getattr(m, self.__var.step_variable)[t, k] - step_fun(m, t, num=k, **self._step_data[k])
                return step_const == 0.0

            constraint_suffix = 'constraint'
            
            setattr(model, self.__var.step_variable + constraint_suffix,
                    Constraint(model.alltime,
                               steps,
                               rule=rule_step_function))
        return None
            
    def create_pyomo_model(self, start_time=None, end_time=None):
        """Create a pyomo model.

        This method is the core method for further simulation or optimization studies

        Args:
            start_time (float): initial time considered in the model

            end_time (float): final time considered in the model

        Returns:
            Pyomo ConcreteModel

        """
        if self._spectral_data is not None and self._absorption_data is not None:
            raise RuntimeError('Either add absorption data or spectral data but not both')
        
        pyomo_model = ConcreteModel()

        # Declare Sets
        pyomo_model.mixture_components = Set(initialize=list(self._component_names))
        pyomo_model.parameter_names = Set(initialize=self.template_parameter_data.names)
        pyomo_model.complementary_states = Set(initialize=list(self._complementary_states))
        pyomo_model.states = pyomo_model.mixture_components | pyomo_model.complementary_states
       
        pyomo_model.measured_data = Set(initialize=self._all_state_data)
        
        # Set up the model time sets and parameters
        self._set_up_times(pyomo_model, start_time, end_time)
        
        # Set up the model by calling the following methods:
        self._add_model_variables(pyomo_model)
        self._add_model_parameters(pyomo_model)
            
        pyomo_model.algebraics = Set(initialize=list(self._algebraics.keys()))
            
        self._add_algebraic_var(pyomo_model)
        self._add_model_constants(pyomo_model)
        self._add_initial_conditions(pyomo_model)
        self._add_spectral_variables(pyomo_model)
        self._add_model_smoothing_parameters(pyomo_model)
        
        # If unwanted contributions are being handled:
        if self._G_contribution is not None:
            self._add_unwanted_contribution_variables(pyomo_model)

         # Add time step variables
        self._add_time_steps(pyomo_model)

        if not hasattr(self, 'c_mod'):
            self.c_mod = Comp(pyomo_model)
        
        if hasattr(self, 'early_return') and self.early_return:
            return pyomo_model

        # Validate the model before writing constraints
        self._validate_data(pyomo_model)

        # Add constraints
        self._add_model_odes(pyomo_model)
        self._add_algebraic_constraints(pyomo_model)
        
        if self._custom_objective:
            self._add_objective_custom(pyomo_model)
        
        # Check the absorbing species sets
        self._check_absorbing_species(pyomo_model)
        
        # Add bounds, is specified
        self._apply_bounds_to_variables(pyomo_model)
       
        # Add given state standard deviations to the pyomo model
        if self._state_sigmas is not None:
            state_sigmas = {k: v for k, v in self._state_sigmas.items() if k in pyomo_model.measured_data}
            pyomo_model.sigma = Param(pyomo_model.measured_data, initialize=state_sigmas) #, mutable=True)
        else:
            pyomo_model.sigma = Param(pyomo_model.measured_data, initialize=1)#, mutable=True)
 
        # In case of a second call after known_absorbing has been declared
        if self._huplc_data is not None and self._is_huplc_abs_set:
            self.set_huplc_absorbing_species(pyomo_model, self._huplc_absorbing, self._vol, self._solid_spec, check=False)

        if self._is_known_abs_set:  
            self.set_known_absorbing_species(pyomo_model,
                                             self._known_absorbance,
                                             self._known_absorbance_data,
                                             check=False
                                             )
            
        return pyomo_model

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

    # def set_estinit_extra_species(self, model, initextra_est_list, check=True):#added for the estimation of initial conditions which have to be complementary state vars CS
    #     # type: (ConcreteModel, list, bool) -> None
    #     """Sets the non absorbing component of the model.

    #     Args:
    #         initextra_est_list: List of to be estimated initial conditions of complementary state variables.
    #         model: The corresponding model.
    #         check: Safeguard against setting this up twice.
    #     """
    #     if hasattr(model, 'estim_init'):
    #         print("To be estimated initial conditions of complementary state variables were already set up before.")
    #         return

    #     if (self._initextra_est_list and check):
    #         raise RuntimeError('To be estimated initial conditions of complementary state variables have been already set up.')

    #     self._initextraest = initextra_est_list
    #     self._estim_init = True
    #     model.estim_init=Param(initialize=self._estim_init)
        
    #     return None

    # def add_init_extra(self, *args, **kwds):#added for the estimation of initial conditions which have to be complementary state vars CS
    #     """Add a kinetic parameter(s) to the model.

    #     Note:
    #         Plan to change this method add parameters as PYOMO variables

    #         This method tries to mimic a template implementation. Depending
    #         on the argument type it will behave differently

    #     Args:
    #         param1 (str): Species name. Creates a variable species

    #         param1 (list): Species names. Creates a list of variable species

    #         param1 (dict): Map species name(s) to value(s). Creates a fixed parameter(s)

    #     Returns:
    #         None

    #     """
    #     bounds = kwds.pop('bounds', None)
    #     init = kwds.pop('init', None)
    #     self._estim_init=True

    #     if len(args) == 1:
    #         name = args[0]
    #         if isinstance(name, six.string_types):
    #             self._initextraparams[name] = None
    #             if bounds is not None:
    #                 self._initextraparams_bounds[name] = bounds
    #             if init is not None:
    #                 self._initextraparams_init[name] = init
    #         elif isinstance(name, list) or isinstance(name, set):
    #             if bounds is not None:
    #                 if len(bounds) != len(name):
    #                     raise RuntimeError('the list of bounds must be equal to the list of species')
    #             for i, n in enumerate(name):
    #                 self._initextraparams[n] = None
    #                 if bounds is not None:
    #                     self._initextraparams_bounds[n] = bounds[i]
    #                 if init is not None:
    #                     self._initextraparams_init[n] = init[i]
    #         elif isinstance(name, dict):
    #             if bounds is not None:
    #                 if len(bounds) != len(name):
    #                     raise RuntimeError('the list of bounds must be equal to the list of species')
    #             for k, v in name.items():
    #                 self._initextraparams[k] = v
    #                 if bounds is not None:
    #                     self._initextraparams_bounds[k] = bounds[k]
    #                     print(bounds[k])
    #                 if init is not None:
    #                     self._initextraparams_init[k] = init[k]
    #         else:
    #             raise RuntimeError('Species data not supported. Try str')
    #     elif len(args) == 2:
    #         first = args[0]
    #         second = args[1]
    #         if isinstance(first, six.string_types):
    #             self._initextraparams[first] = second
    #             if bounds is not None:
    #                 self._initextraparams_bounds[first] = bounds
    #             if init is not None:
    #                 self._initextraparams_init[first] = init
    #         else:
    #             raise RuntimeError('Species argument not supported. Try str,val')
    #     else:
    #         raise RuntimeError('Species argument not supported. Try str,val')

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

        C = getattr(model, self.__var.concentration_spectra)
        Z = getattr(model, self.__var.concentration_model)

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
        S = getattr(model, self.__var.spectra_species)
        C = getattr(model, self.__var.concentration_spectra)
        Z = getattr(model, self.__var.concentration_model)
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

        C = getattr(model, self.__var.concentration_spectra)
        Z = getattr(model, self.__var.concentration_model)

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
