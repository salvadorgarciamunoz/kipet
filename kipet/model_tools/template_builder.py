"""TemplateBuilder - handles inputs for generating the Pyomo model"""

# Standard library imports
import inspect
import itertools
import logging
import warnings

# Third party imports
import numpy as np
import pandas as pd
import six
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.environ import (ConcreteModel, Constraint, ConstraintList, 
                           Param, Reals, Var, Set, Suffix)

# KIPET library imports
from kipet.model_components.component_expression import Comp
from kipet.model_tools.visitor_classes import ReplacementVisitor
from kipet.general_settings.variable_names import VariableNames

logger = logging.getLogger('ModelBuilderLogger')


class TemplateBuilder(object):
    """Helper class for creation of models.
    
    This class takes the arguments, settings, and data from the ReactionModel
    and uses it to generate the Pyomo model used in simulation or parameter
    estimation. The TemplateBuilder object is not meant to be accessed directly
    by the user.

    """
    __var = VariableNames()

    def __init__(self, **kwargs):
        """Initialization for the TemplateBuilder object.

        """
        self._smoothparameters = dict()  # added for mutable parameters CS
        self._smoothparameters_mutable = dict() #added for mutable parameters CS

        #added for initial condition parameter estimates: CS
        self._initextraparams = dict()
        self._initextraparams_init = dict()  # added for parameter initial guess CS
        self._initextraparams_bounds = dict()
        self._y_bounds = dict()  # added for additional optional bounds CS
        self._y_init = dict()
        self._prof_bounds = list()  # added for additional optional bounds MS
        self._prof_points = list()
        self._point_times = list()
        self._point_wavelengths = list()
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
        self._init_absorption_data = None
        self._G_contribution = None
        self._qr_bounds = None
        self._qr_init = None
        self._g_bounds = None
        self._g_init = None
        self._add_dosing_var = False
        self._solid_spec = None

    def add_state_variance(self, sigma_dict):
        """Provide a variance for the measured states
        
        :param dict sigma_dict: A dictionary of the measured states with known variance. Provide the value of
            sigma (standard deviation).
            
        :return: None

        """
        self._state_sigmas = sigma_dict 
        
        return None

    def set_parameter_scaling(self, use_scaling):
        """Makes an option to use the scaling method implemented for estimability

        :param bool use_scaling: Defaults to False, for using scaled parameters
            
        :return: None
        
        """
        self._scale_parameters = use_scaling
    
        return None
        
    def set_model_times(self, times):
        """Add the times to the builder template.
        
        :param tuple times: start and end times for the pyomo model
        
        :return: None
            
        """
        self._times = times

    def add_model_element(self, BlockObject):
        """Adds the ReactionModel component blocks

        :param ComponentBlock BlockObject: The component block from ReactionModel

        :return None:

        """
        data_type = BlockObject.attr_class_set_name.rstrip('s')
        block_attr_name = f'template_{data_type}_data'
        setattr(self, block_attr_name, BlockObject)
        return None
 
    def add_smoothparameter(self, *args, **kwds):
        """Add kinetic parameter(s) to the model.

        :param tuple args: name (str) and dataframe (pandas.DataFrame)
        :param dict kwds: Arguments, mutable as bool

        :return: None

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
    
    def input_data(self, data_block_dict=None, spectral_data=None):
        """Inputs the data from the ReactionModel. This will also figure out the times for the models if not provided.

        :param dict data_block_dict: The input data
        :param bool spectral_data: Indicates if the data is spectral

        :return: None
        
        """
        time_span = 0
        time_conversion_factor = 1
        
        data_type_labels = {'component' : 'concentration',
                            'state': 'complementary_states',
                            'algebraic': 'custom',
                            'uplc': ''
                            }

        if data_block_dict is not None:
            for data_type, data_label in data_type_labels.items():
                if hasattr(self, f'template_{data_type}_data'):
                    c_info = getattr(self, f'template_{data_type}_data')
                    for comp in c_info:
                        if hasattr(comp, 'data_link'):
                            data_block = data_block_dict[comp.data_link]
                            time_span = max(time_span, data_block.time_span[1]*time_conversion_factor)
                            data_frame = data_block.data[comp.name]
                            if not comp.use_orig_units:
                                data_frame *= comp.conversion_factor
                            # data_frame.index = data_frame.index*time_conversion_factor
                            self._add_state_data(data_frame, data_label, overwrite=False)

        # Spectral data is handled differently
        if spectral_data is not None:
            self._add_state_data(spectral_data.data, 'spectral')
            # spectral_data.data.index = spectral_data.data.index*time_conversion_factor
            time_span = max(time_span, spectral_data.data.index.max())
                
        self.time_span_max = time_span
       
        return None
    
    def clear_data(self):
        """Method to clear the model data

        :return: None

        """
        self._spectral_data = None
        self._concentration_data = None
        self._complementary_states_data = None # added for complementary state data (Est.) KM
        self._huplc_data = None #added for additional data CS
        self._smoothparam_data = None  # added for additional smoothing parameter data CS
        self._absorption_data = None
        
        return None

    def _add_state_data(self, data, data_type, label=None, overwrite=True):
        """Generic method for adding data (concentration or complementary 
        state data) - uses the measured data attribute to process
        
        :param pandas.DataFrame data: DataFrame with measurement times as indices and concentrations as columns.
        :param str data_type: The name of the attribute for where the data is to be stored.
        :param str label: The label used to descibe the data in the pyomo model index.

        :return: None

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
            self.__var.huplc_data,
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
            if getattr(self, f'_is_{label}_deriv') == True:
                print(
                    f"Warning! Since {label}-matrix contains negative values Kipet is assuming a derivative of {label} has been inputted")

        return None
           
    def add_huplc_data(self, data, overwrite=True):
        """Add HPLC or UPLC data as a wrapper to _add_state_data

        :param pandas.DataFrame data: DataFrame with measurement times as indices and wavelengths as columns.
        :param bool overwrite: Option to overwrite exisiting data

        :return: None

        """
        self._add_state_data(data,
                             data_type='huplc',
                             overwrite=overwrite)
        
        self._is_huplc_abs_set = True
        self._huplc_absorbing = data.columns

    def add_smoothparam_data(self, data, overwrite=True):
        """Add smoothing parameter data

        :param pandas.DataFrame data: DataFrame with measurement times as indices and wavelengths as columns.
        :param bool overwrite: Option to overwrite exisiting data

        :return: None

        """
        self._add_state_data(data,
                             data_type='smoothparam',
                             overwrite=overwrite)

    def add_init_absorption_data(self, data):
        """Initialize the absorption using simulated data

        :param pandas.DataFrame data: DataFrame with measurement times as indices and wavelengths as columns.

        :return: None

        """
        if isinstance(data, pd.DataFrame):
            self._init_absorption_data = data
        else:
            raise RuntimeError('Spectral data format not supported. Try pandas.DataFrame')

    def add_absorption_data(self, data, overwrite=True):
        """Add absorption data

        :param pandas.DataFrame data: DataFrame with measurement times as indices and wavelengths as columns.
        :param bool overwrite: Option to overwrite exisiting data

        :return: None

        """
        if isinstance(data, pd.DataFrame):
            self._absorption_data = data
        else:
            raise RuntimeError('Spectral data format not supported. Try pandas.DataFrame')

    @staticmethod
    def round_time(time):
        """Round the time to a fixed number of significant digits

        :param float time: The time value

        :return: time rounded to fixed number of decimals (6)
        :rtype: float

        """
        return round(time, 6)

    def add_feed_times(self, times):
        """Add measurement times to the model

        :param array-like times: feeding points

        :return: None

        """
        for t in times:
            t = self.round_time(t)
            self._feed_times.add(t)

    def add_qr_bounds_init(self, bounds=None, init=None):
        """Read bounds and initialize for qr variables

        :param tuple bounds: The qr bounds
        :param float init: The initial qr value

        :return: None

        """
        self._qr_bounds = bounds
        self._qr_init = init
        
        return None
        
    def add_g_bounds_init(self, bounds=None, init=None):
        """Read bounds and initialize for g variables

        :param tuple bounds: The qr bounds
        :param float init: The initial qr value

        :return: None

        """
        self._g_bounds = bounds
        self._g_init = init
    
        return None
    
    def add_dosing_var(self, n_steps):
        """Adds a dosing variable to the template

        :param int n_steps: Number of steps

        :return: None
        """
        self._add_dosing_var = True
        self._number_of_steps = n_steps
    
    def set_odes_rule(self, rule):
        """Defines the ordinary differential equations that define the dynamics of the model

        :param dict rule: Model expressions for the rate equations

        :return: None

        """
        self._odes = rule
        
        return None

    def set_algebraics_rule(self, rule, asdict=False):
        """Defines the algebraic equations for the system

        :param dict rule: Dictionary of algebraic expressions

        :return: None

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

        :param list algebraic_vars: List of algebraic variables

        :return: None

        """
        if not isinstance(algebraic_vars, list):
            algebraic_vars = list(algebraic_vars)
        self._custom_objective = algebraic_vars
        
        return None

    def bound_profile(self, var, bounds, comp=None, profile_range=None):
        """function that allows the user to bound a certain profile to some value

        :param GeneralVar var: The pyomo variable that we will bound
        :param str comp:The component that bound applies to
        :param tuple profile_range: The range within the set to be bounded
        :param tuple bounds: The values to bound the profile to

        :return: None

        """
        self._prof_bounds.append([var, comp, profile_range, bounds])

    def bound_point(self, var, bounds, comp=None, point=None):
        """function that allows the user to bound a certain profile to some value

        :param GeneralVar var: The pyomo variable that we will bound
        :param str comp:The component that bound applies to
        :param tuple profile_range: The range within the set to be bounded
        :param tuple bounds: The values to bound the profile to

        :return: None

        """
        if var != 'S':
            self._point_times.append(point)
        else:
            self._point_wavelengths.append(point)
        
        self._prof_points.append([var, comp, point, bounds])

    def _validate_data(self, model):
        """Verify all inputs to the model make sense.

        This method is not suppose to be used by users. Only for developers use

        :param ConcreteModel model: The created Pyomo model

        :return: None

        """
        if not self.template_component_data.component_set('concentration'):
            warnings.warn('The Model does not have any mixture components')
        else:
            if self._odes:
                if isinstance(self._odes, dict):
                    dummy_balances = self._odes
                else:
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

        :param ConcreteModel model: The created Pyomo model

        :return: None

        """
        if hasattr(self, 'template_algebraic_data'):    
            a_info = self.template_algebraic_data
        
            setattr(model, self.__var.algebraic, Var(model.alltime,
                                                     a_info.names,
                                                     initialize=a_info.as_dict('value')))
                                                     
            for alg in a_info:
                if alg.bounds is not None:
                    for t in model.alltime:
                        getattr(model, self.__var.algebraic)[t, alg.name].setlb(alg.lb)
                        getattr(model, self.__var.algebraic)[t, alg.name].setub(alg.ub)
                        
            # for alg in a_info:
            #     if alg.step is not None:
                     
            #     setattr(model, self.__var.step_variable, Var(model.alltime,
            #                                                  steps,
            #                                                  initialize=1.0))
                
            #     #Constraint to keep track of the step values (for diagnostics)
            #     def rule_step_function(m, t, k):
            #         step_const = getattr(m, self.__var.step_variable)[t, k] - step_fun(m, t, num=k, **self._step_data[k])
            #         return step_const == 0.0
    
            #     constraint_suffix = 'constraint'
                
            #     setattr(model, self.__var.step_variable + constraint_suffix,
            #             Constraint(model.alltime,
            #                        steps,
            #                        rule=rule_step_function))
    
        return None
    
    def _add_initial_conditions(self, model):
        """Set up the initial conditions for the model

        :param ConcreteModel model: The created Pyomo model

        :return: None
        """
        var_info = dict(self.template_component_data._dict)
        
        if hasattr(self, 'template_state_data'):
            var_info.update(self.template_state_data._dict)
        
        model.init_conditions = Var(model.states,
                                    initialize={k: v.value for k, v in var_info.items()},
                                    )
        unknown_init = {}
        
        for var, obj in model.init_conditions.items():
            if var_info[var].known:
                obj.fix()
            else:
                lb = var_info[var].lb
                ub = var_info[var].ub
                init = var_info[var].value
                obj.setlb(lb)
                obj.setub(ub)
                unknown_init[var] = init
        
        if len(unknown_init) > 0:
            model._unknown_init_set = Set(initialize=list(unknown_init.keys()))
            setattr(model, self.__var.concentration_init, Var(model._unknown_init_set,
                                                             initialize=unknown_init))
            
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
            
        if hasattr(model, self.__var.concentration_init):
            
            def rule_Pinit_conditions(m, k):
                if k in m.mixture_components:
                    return getattr(m, self.__var.concentration_init)[k] - m.init_conditions[k] == 0
                else:
                    return getattr(m, self.__var.concentration_init)[k] - m.init_conditions[k] == 0
    
            model.Pinit_conditions_c = \
                Constraint(model._unknown_init_set, rule=rule_Pinit_conditions)

        return None

    def _add_model_variables(self, model):
        """Adds the model variables to the pyomo model

        :param ConcreteModel model: The created Pyomo model

        :return: None
        """
        c_info = self.template_component_data
        self._component_names = c_info.names
        
        var_set = [c_info]
        
        if hasattr(self, 'template_state_data'):
            s_info = self.template_state_data
            self._complementary_states = s_info.names
            var_set.append(s_info)
        
        for var_class in var_set:
        
            v_info = var_class    
        
            model_pred_var_name = {
                    'components' : [self.__var.concentration_model, model.mixture_components],
                    'states' : [self.__var.state_model, model.complementary_states],
                        }
            
            var, model_set = model_pred_var_name[var_class.attr_class_set_name]
            
            if hasattr(model_set, 'ordered_data') and len(model_set.ordered_data()) == 0:
                continue
            
            setattr(model, var, Var(model.alltime,
                                          model_set,
                                          initialize=1) 
                    )    
        
            for time, comp in getattr(model, var):
                if time == model.start_time.value:
                    
                    getattr(model, var)[time, comp].value = v_info[comp].value
                   
            setattr(model, f'd{var}dt', DerivativeVar(getattr(model, var),
                                                      wrt=model.alltime)
                    )

        fixed_var_name = {
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
                    
                    setattr(model, var, Var(getattr(model, f'{var}_indx'),
                                                  bounds=c_bounds,
                                                  initialize=c_dict,
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
                            getattr(model, var)[time, comp].value = v_info[comp].value

        return None
    
    def _add_model_constants(self, model):
        """Add the model constants

        :param ConcreteModel model: The created Pyomo model

        :return: None

        """
        if hasattr(self, 'template_constant_data'):
        
            con_info = self.template_constant_data
            
            setattr(model, self.__var.model_constant, Var(con_info.names,
                                                            initialize=con_info.as_dict('value'),
                                                            ))    
        
            for param, obj in getattr(model, self.__var.model_constant).items():
                obj.fix()
          
        return None

    def _add_model_parameters(self, model):
        """Add the model parameters to the pyomo model

        :param ConcreteModel model: The created Pyomo model

        :return: None

        """
        if hasattr(self, 'template_parameter_data'):
            
            p_info = self.template_parameter_data
         
            # Initial parameter values
            p_values = p_info.as_dict('value')
            for param, value in p_values.items():
                if value is None:
                    if p_info[param].bounds[0] is not None and p_info[param].bounds[1] is not None:
                        p_values[param] = sum(p_info[param].bounds)/2
                else:
                    p_values[param] = value
                        
            if self._scale_parameters:
                setattr(model, self.__var.model_parameter, Var(model.parameter_names,
                                                               bounds = (0.1, 10),
                                                               initialize=1))
    
            else:
                # print(p_values)
                setattr(model, self.__var.model_parameter,
                        Var(model.parameter_names,
                            initialize=p_values))
    
            # Set the bounds
            p_bounds = p_info.as_dict('bounds')
            for k, v in p_bounds.items():
                factor = 1
                if self._scale_parameters:
                    factor = p_values[k]
                    
                #print(p_info)
    
                if p_info[k].lb is not None:
                    lb = p_info[k].lb/factor
                    getattr(model, self.__var.model_parameter)[k].setlb(lb)
    
                if p_info[k].ub is not None:
                    ub = p_info[k].ub/factor
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
        """Add the bounds for the unwanted contributions, if any

        :param ConcreteModel model: The created Pyomo model

        :return: None

        """
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

    def change_time(self, expr_orig, c_mod, new_time, current_model):
        """Method to remove the fixed parameters from the ConcreteModel and replace them with real parameters. This
        converts the dummy variables used to generate the expressions.

        .. note::

            At the moment, this only supports one and two dimensional variables

        :param expression expr_orig: The original user generated expression
        :param dict c_mod: The dict of dummy variables
        :param float new_time: The time to replace in the variable index
        :param ConcreteModel current_model: The model containing the variables

        :return expr_new_time: The expression with the new time

        """
        expr_new_time = expr_orig
        var_dict = c_mod
        
        for model_var, obj_list in var_dict.items():
            if not isinstance(obj_list[1].index(), int):
                old_var = obj_list[1]
                new_var = getattr(current_model, obj_list[0])[new_time, model_var]
        
            else:
                old_var = obj_list[1]
                new_var = getattr(current_model, obj_list[0])[model_var]
        
            expr_new_time = self._update_expression(expr_new_time, old_var, new_var)
    
        return expr_new_time

    @staticmethod
    def _update_expression(expr, replacement_param, change_value):
        """Takes the non-estiambale parameter and replaces it with its intitial
        value
        
        :param expression expr: the target ode constraint
        :param str replacement_param: the non-estimable parameter to replace
        :param float change_value: initial value for the above parameter
            
        :return: expression new_expr: Updated constraints with the desired parameter replaced with a float
        
        """
        visitor = ReplacementVisitor()
        visitor.change_replacement(change_value)
        visitor.change_suspect(id(replacement_param))
        new_expr = visitor.dfs_postorder_stack(expr)       
        return new_expr

    def _add_model_odes(self, model):
        """Adds the ODE system to the model, if any

        :param ConcreteModel model: The created Pyomo model

        :return: None

        """
        if isinstance(self._odes, dict):
            
            def rule_odes(m, t, k):
                exprs = self._odes                
               
                if t == m.start_time.value:
                    return Constraint.Skip
                else:
                    if k in m.mixture_components:
                        if k in exprs.keys():
                            deriv_var = f'd{self.__var.concentration_model}dt'
                            final_expr = getattr(m, deriv_var)[t, k] == exprs[k].expression
                            final_expr  = self.change_time(final_expr, self.c_mod, t, m)
                            return final_expr
                        else:
                            return Constraint.Skip
                    else:
                        if k in exprs.keys():
                            deriv_var = f'd{self.__var.state_model}dt'
                            final_expr = getattr(m, deriv_var)[t, k] == exprs[k].expression
                            final_expr  = self.change_time(final_expr, self.c_mod, t, m)
                            return final_expr
                        else:
                            return Constraint.Skip

            setattr(model, self.__var.ode_constraints, Constraint(model.alltime,
                                                                  model.states,
                                                                  rule=rule_odes))
        return None
    
    def _add_algebraic_constraints(self, model):
        """Adds the algebraic constraints the model, if any

        :param ConcreteModel model: The created Pyomo model

        :return: None

        """
        if self._algebraic_constraints:
            if hasattr(self, 'reaction_dict') or hasattr(self, '_use_alg_dict') and self._use_alg_dict:
                n_alg_eqns = list(self._algebraic_constraints.keys())
                
                def rule_algebraics(m, t, k):
                    alg_const = self._algebraic_constraints[k].expression
                    alg_var = getattr(m, self.__var.algebraic)[t, k]
                    final_expr = alg_var - alg_const == 0.0
                    final_expr  = self.change_time(final_expr, self.c_mod, t, m)
                    return final_expr
    
            model.algebraic_consts = Constraint(model.alltime,
                                                n_alg_eqns,
                                                rule=rule_algebraics)
        return None
    
    def _add_objective_custom(self, model):
        """Adds the custom objectives, if any and uses the model variable
        UD for the data. This is where the custom data is stored.

        .. note::

            This only handles one addition such objective term type

        :param ConcreteModel model: The created Pyomo model

        :return: None

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

    def _add_spectral_variables(self, model, estimator):
        """Add D and C variables for the spectral data

        :param ConcreteModel model: The created Pyomo model
        :param bool is_simulation: Indicates if the model is for simulation or not

        :return: None

        """
        if self._spectral_data is not None:
            s_data_dict = dict()
            index_list = list(itertools.product(self._spectral_data.index, self._spectral_data.columns))
            
            for index in index_list:
                try:
                    s_data_dict[index] = float(self._spectral_data.loc[index])
                except:
                    s_data_dict[index] = float('nan')

            setattr(model, self.__var.spectra_data, Param(model.times_spectral,
                                                          model.meas_lambdas,
                                                          domain=Reals,
                                                          initialize=s_data_dict))
            
            if estimator != 'simulator':
                setattr(model, self.__var.concentration_spectra, Var(model.times_spectral,
                                                                 model.mixture_components,
                                                                 bounds=(0, None),
                                                                 initialize=1))
            
        return None

    def _check_absorbing_species(self, model):
        """Set up the appropriate S depending on absorbing species

        :param ConcreteModel model: The created Pyomo model

        :return: None

        """
        if self._absorption_data is not None:
            s_dict = dict()
            for k in self._absorption_data.columns:
                for l in self._absorption_data.index:
                    s_dict[l, k] = float(self._absorption_data[k][l])
        
        elif self._init_absorption_data is not None:
            s_dict = dict()
            for k in self._init_absorption_data.columns:
                for l in self._init_absorption_data.index:
                    s_dict[l, k] = float(self._init_absorption_data[k][l])
                    
        else:
            s_dict = 0.0
        
        if self._is_D_deriv == True:
            s_bounds = (None, None)
        else:
            s_bounds = (0.0, None)
        
        if self.has_spectral_data():    
            
            self.set_non_absorbing_species(model, self._non_absorbing, check=False)
            
            if self._is_non_abs_set:
                
                setattr(model, self.__var.spectra_species, Var(model.meas_lambdas,
                                                               model.abs_components,
                                                               bounds=s_bounds,
                                                               initialize=0.0))
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
    
    def _check_bounds_entries(self):
        """Checks the bounded variables after the model has been built.
        
        :return: None
        
        """
        for bound_set in self._prof_bounds:

            var = bound_set[0]
            comp = bound_set[1]
            profile_range = bound_set[2]
            bounds = bound_set[3]
        
            if not isinstance(var, str):
                raise RuntimeError('var argument needs to be type string')
    
            if var not in ['C', 'U', 'S']:
                raise RuntimeError('var argument needs to be either C, U, or S')
    
            if comp is not None:
                if not isinstance(comp, str):
                    raise RuntimeError('comp argument needs to be type string')
                if comp not in self.template_component_data.names:
                    raise RuntimeError('comp needs to be one of the components')
    
            if profile_range is not None:
                if not isinstance(profile_range, tuple):
                    raise RuntimeError('profile_range needs to be a tuple')
                    if profile_range[0] > profile_range[1]:
                        raise RuntimeError('profile_range[0] must be greater than profile_range[1]')
    
            if not isinstance(bounds, tuple):
                raise RuntimeError('bounds needs to be a tuple')
                
        return None
        
    def _apply_bounds_to_variables(self, model):
        """User specified bounds to certain model variables are added here

        :param ConcreteModel model: The created Pyomo model

        :return: None

        """
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
                        
        for bound_set in self._prof_points:
            var = bound_set[0]
            component_name = bound_set[1]
            time = bound_set[2]
            upper_bound = bound_set[3][1]
            lower_bound = bound_set[3][0]
            getattr(model, var)[time, comp].setlb(lower_bound)
            getattr(model, var)[time, comp].setub(upper_bound)
                        
        return None

    def _add_model_smoothing_parameters(self, model):
        """Adds smoothing parameters to the model, if any.
        These are optional smoothing parameter values (mutable) read from a file

        :param ConcreteModel model: The created Pyomo model

        :return: None
        
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

    def _set_up_times(self, model, start_time, end_time, estimator):
        """This method sets up the times for the model based on all of the inputs.

        :param ConcreteModel model: The created Pyomo model
        :param float start_time: The start time (0)
        :param float end_time: The end time
        :param bool is_simulation: Indicates if the model is for simulation

        :return: None

        """
        if self._times is not None:
            if start_time is None:
                start_time = self._times[0]
            if end_time is None:
                end_time = self._times[1]
        else:
            if start_time is None and end_time is None:
                try:    
                    start_time = 0
                    end_time = self.time_span_max
                except:
                    raise ValueError('A model requires a start and end time or a dataset')
        
        start_time = self.round_time(start_time)
        end_time = self.round_time(end_time)
        model_times = {}
        model_data = {}
        
        model_times['feed'] = sorted(list(set(self._feed_times)))
        model_times['spectral'] = sorted(list(set(self._spectral_data.index))) if self._spectral_data is not None else list()
        model_times['concentration'] = sorted(list(set(self._concentration_data.index))) if self._concentration_data is not None else list()
        model_times['states'] = sorted(list(set(self._complementary_states_data.index))) if self._complementary_states_data is not None else list()
        model_times['smooth'] = sorted(list(set(self._smoothparam_data.index))) if self._smoothparam_data is not None else list()
        # model_times['point'] = set(self._point_times)
    
        model_data['lambda_D'] = sorted(list(self._spectral_data.columns)) if self._spectral_data is not None else list()
        model_data['lambda_S'] = sorted(list(self._absorption_data.index)) if self._absorption_data is not None else list()
        # model_data['lambda_point'] = set(self._point_wavelengths)

        if estimator == 'p_estimator':
            model_times['uplc'] = sorted(list(set(self._huplc_data.index))) if self._huplc_data is not None else list()
            model_times['custom'] = sorted(list(set(self._custom_data.index))) if self._custom_data is not None else list()
    
        model_times['all_times'] = sorted(list(set().union(*model_times.values())))
        model_data['all_lambdas'] = sorted(list(set().union(*model_data.values())))
        
        for key, value in model_times.items():
            if key not in ['times', 'all_times']:
                setattr(model, f'times_{key}', Set(initialize=value, ordered=True))
        
        model.feed_times = Set(initialize=model_times['feed'], ordered=True)
        model.meas_lambdas = Set(initialize=model_data['all_lambdas'], ordered=True)
        
        model.allmeas_times = Set(initialize=model_times['all_times'], ordered=True)
        model.alltime = ContinuousSet(initialize=model.allmeas_times, bounds=(start_time, end_time))
        
        model.start_time = Param(initialize=start_time, domain=Reals)
        model.end_time = Param(initialize=end_time, domain=Reals)
        
        if 'uplc' in model_times:
            model.huplcmeas_times = Set(initialize=model_times['uplc'], ordered=True)
            model.huplctime = ContinuousSet(initialize=model.huplcmeas_times, bounds=(start_time, end_time))
            
        self.start_time = start_time
        self.end_time = end_time
        self.model_times = model_times
        
        return None
    
    @staticmethod
    def _check_time_inputs(time_set, start_time, end_time):
        """Checks the first and last time of a measurement to see if it's in
        the model time bounds

        :param array-like time_set: The list of all times
        :param float start_time: The start time
        :param float end_time: The end time

        :return: None

        """
        if time_set[0] < start_time:
            raise RuntimeError(f'Measurement time {time_set[0]} not within ({start_time}, {end_time})')
        if time_set[-1] > end_time:
            raise RuntimeError(f'Measurement time {time_set[-1]} not within ({start_time}, {end_time})')
    
        return None
    
    def add_step_vars(self, step_data):
        """Adds a step variable if the data is provided

        :param dict step_data: The step data for the ReactionModel

        :return None

        """
        if not hasattr(self, '_step_data'):
            self._step_data = None
        self._step_data = step_data
        return None
    
    def _add_time_steps(self, model):
        """Builds the step functions into the model using the step_fun method

        :param ConcreteModel model: The created Pyomo model

        :return: None

        """
        from kipet.model_tools.model_functions import step_fun

        # Add dosing var
        setattr(model, self.__var.dosing_variable, Var(model.alltime,
                                                      [self.__var.dosing_component],
                                                      initialize=0))
            
        def build_step_funs(m, t, k, data):
            
            step = 0
            off_modifier = 0
            for i in range(len(data)):
                if i > 0 and data[i]['switch'] == 'off':
                    off_modifier += 1
                step += step_fun(m, t, num=f'{k}_{i}', **data[i])
                
            return step - off_modifier
        
        def rule_step_function(m, t, k):
            step_const = getattr(m, self.__var.step_variable)[t, k] - build_step_funs(m, t, k, self._step_data[k])    # step_fun(m, t, num=k, **self._step_data[k])
            return step_const == 0.0
        
        if hasattr(self, '_step_data') and len(self._step_data) > 0:
            steps = list(self._step_data.keys())
            
            #for step in steps:
                #step_info = self._step_data[step]
                #num_of_steps = len(step_info)
                #step_index = list(range(num_of_steps))
                  
            tsc_index = []
            tsc_init = {}
            tsc_bounds = {}
            tsc_fixed = {}
        
            for k, v in self._step_data.items():
                for i, step_dict in enumerate(v):
                    tsc_init[f'{k}_{i}'] = step_dict['time']
                    tsc_bounds[f'{k}_{i}'] = step_dict.get('bounds', (model.start_time, model.end_time))
                    tsc_fixed[f'{k}_{i}'] = step_dict['fixed']
                    tsc_index.append(f'{k}_{i}')
                    
            
            setattr(model, self.__var.time_step_change, Var(tsc_index,
                                                           # step_index,
                    initialize=tsc_init,
                    ))
                   
            # Set the bounds
            for k, v in getattr(model, self.__var.time_step_change).items():
                v.setlb(tsc_bounds[k][0])
                v.setub(tsc_bounds[k][1])
                
                if tsc_fixed[k]:
                    v.fix()
            
            setattr(model, self.__var.step_variable, Var(model.alltime,
                                                         steps,
                                         #                step_index,
                                                         initialize=1.0))
                
            #Constraint to keep track of the step values (for diagnostics)

            constraint_suffix = 'constraint'
            
            setattr(model, self.__var.step_variable + constraint_suffix,
                    Constraint(model.alltime,
                               steps,
                         #      step_index,
                               rule=rule_step_function))   
            
            # If needed, this section may be implemented
            # def rule_step_alg(m, t, k, k2, factor):
            #       print(t, k, k2, factor)
            #       step_const = 1/factor*getattr(m, self.__var.algebraic)[t, k] - step_fun(m, t, num=k2, **self._step_data[k2])
            #       return step_const == 0.0

            # if hasattr(self, 'template_algebraic_data'):
            #     alg_steps = self.template_algebraic_data.steps
                
            #     alg_values = {alg.name: alg.value for alg in alg_steps.values()}
            #     alg_set = [alg.name for alg in alg_steps.values()]
            #     alg_step = {alg.name: alg.step for alg in alg_steps.values()}
                           
            #     steps = list(set(steps).difference(alg_set))                  
        
            #     setattr(model, self.__var.algebraic + constraint_suffix,
            #             Constraint(model.alltime,
            #                        alg_set,
            #                        alg_step.values(),
            #                        alg_values.values(),
            #                        rule=rule_step_alg,
            #                        ))
                
        return None
            
    def create_pyomo_model(self, start_time=None, end_time=None, estimator='simulator'):
        """Create a pyomo model from the provided data.

        This method is the core method for further simulation or optimization studies

        :param flaot start_time: Initial time considered in the model
        :param float end_time: Final time considered in the model
        :param bool is_simulation: Indicates if the model is for simulation

        :return: The finished Pyomo ConcreteModel
        :rtype: ConcreteModel

        """
        print(f'# TemplateBuilder: Preparing model for {estimator}')
        is_simulation = estimator == 'simulator' 
        
        if self._spectral_data is not None and self._absorption_data is not None:
            raise RuntimeError('Either add absorption data or spectral data but not both')
        
        pyomo_model = ConcreteModel()

        # Declare Sets
        pyomo_model.mixture_components = Set(initialize=self.template_component_data.names)
        
        if not hasattr(self, 'template_parameter_data'):
            parameter_names = []
        else:
            parameter_names = self.template_parameter_data.names
        pyomo_model.parameter_names = Set(initialize=parameter_names)
        
        if not hasattr(self, 'template_state_data'):
            state_names = []
        else:
            state_names = self.template_state_data.names
        pyomo_model.complementary_states = Set(initialize=state_names)
        
        pyomo_model.states = pyomo_model.mixture_components | pyomo_model.complementary_states
        pyomo_model.measured_data = Set(initialize=self._all_state_data)
        
        # Set up the model time sets and parameters
        self._set_up_times(pyomo_model, start_time, end_time, estimator)
        
        # Set up the model elements by calling the following methods:
        self._add_model_variables(pyomo_model)
        self._add_model_parameters(pyomo_model)
            
        if not hasattr(self, 'template_algebraic_data'):
            alg_names = []
        else:
            alg_names = self.template_algebraic_data.names
        pyomo_model.algebraics = Set(initialize=alg_names)
            
        self._add_algebraic_var(pyomo_model)
        self._add_model_constants(pyomo_model)
        self._add_initial_conditions(pyomo_model)
        self._add_spectral_variables(pyomo_model, estimator)
        
        # This may be added below (no examples of this)
        self._add_model_smoothing_parameters(pyomo_model)
        
        # If unwanted contributions are being handled:
        if self._G_contribution is not None:
            self._add_unwanted_contribution_variables(pyomo_model)

         # Add time step variables
        self._add_time_steps(pyomo_model)

        if not hasattr(self, 'c_mod'):
            self.c_mod = Comp(pyomo_model)
            
        # Exit point for the dummy model
        if hasattr(self, 'early_return') and self.early_return:
            return pyomo_model

        # Validate the model before writing constraints
        self._validate_data(pyomo_model)

        # Add constraints
        self._add_model_odes(pyomo_model)
        self._add_algebraic_constraints(pyomo_model)
        
        # Add objective terms for custom data
        if self._custom_objective and estimator == 'p_estimator':
            self._add_objective_custom(pyomo_model)
        
        # Check the absorbing species sets
        if not is_simulation:
            self._check_absorbing_species(pyomo_model)
        
        # Add bounds, is specified
        if not is_simulation:
            self._apply_bounds_to_variables(pyomo_model)
       
        # Add given state standard deviations to the pyomo model
        self._state_sigmas = self.template_component_data.var_variances()
        if hasattr(self, 'template_state_data'):
            self._state_sigmas.update(**self.template_state_data.var_variances())
        
        # Add state/component variances
        for k, v in self._state_sigmas.items():
            if v is None:
                self._state_sigmas[k] = 1
       
        state_sigmas = {k: v for k, v in self._state_sigmas.items() if k in pyomo_model.measured_data}
        pyomo_model.sigma = Param(
            pyomo_model.measured_data, 
            domain=Reals, 
            initialize=state_sigmas
        )
       
        # Add UPLC data too the ParameterEstimator model
        if estimator ==  'p_estimator':
            if self._huplc_data is not None and self._is_huplc_abs_set:
                self.set_huplc_absorbing_species(pyomo_model, self._huplc_absorbing, self._vol, self._solid_spec, check=False)

        # In case of a second call after known_absorbing has been declared
        
        # This needs to be handled like a normal update
        # If a species has .S that is not None, then this should be called here
        
        if not is_simulation:
        
            known_abs = []
            known_abs_data = {}
            for comp in self.template_component_data:
                if comp.S is not None:
                    known_abs.append(comp.name)
                    known_abs_data[comp.name] = comp.S
            
            self._known_absorbance = known_abs
            self._known_absorbance_data = known_abs_data
            
            if len(self._known_absorbance) > 0:  
                self.set_known_absorbing_species(
                    pyomo_model,
                    self._known_absorbance,
                    self._known_absorbance_data,
                    check=False
                )
            
        return pyomo_model

    @property
    def num_parameters(self):
        """Returns the number of parameters

        :return: The number of paramters
        :rtype: float

        """
        return len(self._parameters)

    @property
    def num_mixture_components(self):
        """Returns the number of components

        :return: The number of components
        :rtype: float

        """
        return len(self._component_names)

    @property
    def num_complementary_states(self):
        """Returns the number of states

        :return: The number of states
        :rtype: float

        """
        return len(self._complementary_states)

    @property
    def num_algebraics(self):
        """Returns the number of algebraics

        :return: The number of algebraics
        :rtype: float

        """
        return len(self._algebraics)

    @property
    def measurement_times(self):
        """Returns the measurement times

        :return: The measurement times
        :rtype: float

        """
        return self._meas_times

    @property
    def huplcmeasurement_times(self):
        """Returns the HUPLC measurement times

        :return: The number of HUPLC measurement times
        :rtype: float

        """
        return self._huplcmeas_times

    @property
    def feed_times(self):
        """Returns the feed times

        :return: The feed times
        :rtype: float

        """
        return self._feed_times  # added for feeding points CS

    def has_spectral_data(self):
        """Return bool if spectral data is in the model

        :return: True if spectral data is in the model

        """
        return self._spectral_data is not None

    def has_adsorption_data(self):
        """Return bool if adsorption data is in the model

        :return: True if adsorption data is in the model

        """
        return self._absorption_data is not None

    def has_concentration_data(self):
        """Return bool if concentration data is in the model

        :return: True if concentration data is in the model

        """
        return self._concentration_data is not None

    def optional_warmstart(self, model):
        """Updates suffixes with previous results

        :param ConcreteModel model: The current Pyomo model

        :return: None

        """
        if hasattr(model, 'dual') and hasattr(model, 'ipopt_zL_out') and hasattr(model, 'ipopt_zU_out') and hasattr(model, 'ipopt_zL_in') and hasattr(model, 'ipopt_zU_in'):
            print('warmstart model components already set up before.')
        else:
            model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

    def set_non_absorbing_species(self, model, non_abs_list, check=True):
        """Sets the non absorbing component of the model.
        
        .. note::
            
            This could be simplified by making Cs always exist. This would
            remove quite a few if statements throughout the code.

        :param list non_abs_list: List of non absorbing components.
        :param ConcreteModel model: The corresponding model.
        :param bool check: Safeguard against setting this up twice.

        :return: None

        """
        # if hasattr(model, 'non_absorbing'):
        #     print("non-absorbing species were already set up before.")
        #     return

        # if (self._is_non_abs_set and check):
        #     raise RuntimeError('Non absorbing species have been already set up.')
        if hasattr(model, 'abs_components'):
            model.del_component('abs_components')
        
        self._is_non_abs_set = True
        self._non_absorbing = [] if non_abs_list is None else non_abs_list
        #model.add_component('non_absorbing', Set(initialize=self._non_absorbing))

        # Exclude non absorbing species from S matrix and create subset Cs of C:
        #model.add_component('abs_components_names', Set())
        
        set_all_components = set(model.mixture_components)
        set_non_abs_components = set(self._non_absorbing)
        set_abs_components = set_all_components.difference(set_non_abs_components)
        
        list_abs_components = sorted(list(set_abs_components))
        
        model.add_component('abs_components', Set(initialize=list_abs_components))
        #model.add_component('Cs', Var(model.times_spectral, list_abs_components))
        #model.add_component('abs_subset_contraint', ConstraintList())
        
        # for time in model.times_spectral:
        #     for comp in model.abs_components:
        #         model.abs_subset_contraint.add(model.Cs[time, comp] == model.C[time, comp])
                    
        return None

    def set_known_absorbing_species(self, model, known_abs_list, absorbance_data, check=True):
        """Sets the known absorbance profiles for specific components of the model.

        :param list known_abs_list: List of known species absorbance components.
        :param ConcreteModel model: The corresponding model.
        :param pandas.DataFrame absorbance_data: the dataframe containing the known component spectra
        :param bool check: Safeguard against setting this up twice.

        :return: None

        """
        if hasattr(model, 'known_absorbance'):
            print("species with known absorbance were already set up before.")
            return

        if (self._is_non_abs_set and check):
            raise RuntimeError('Species with known absorbance were already set up before.')

        self._is_known_abs_set = True
        model.add_component('known_absorbance', Set(initialize=self._known_absorbance))
        S = getattr(model, self.__var.spectra_species)
        lambdas = getattr(model, 'meas_lambdas')
        model.known_absorbance_data = self._known_absorbance_data
        
        for component in self._known_absorbance:
            for l in lambdas:
                S[l, component].set_value(self._known_absorbance_data[component].loc[l, component])
                S[l, component].fix()
                
        return None

    def set_huplc_absorbing_species(self, model, huplc_abs_list, vol=None, solid_spec=None, check=True):
        """Sets the non absorbing component of the model.

        :param list huplc_abs_list: List of huplc absorbing components.
        :param float vol: reactor volume
        :param str solid_spec: solid species observed with huplc data
        :param ConcreteModel model: The corresponding model.
        :param bool check: Safeguard against setting this up twice.

        :return: None

        """
        if hasattr(model, 'huplc_absorbing'):
            print("huplc-absorbing species were already set up before.")
            return

        if (self._is_huplc_abs_set and check):
            raise RuntimeError('huplc-absorbing species have been already set up.')

        self._is_huplc_abs_set = True
        #self._vol=vol
        #model.add_component('vol', Param(initialize=self._vol, within=Reals))

        self._huplc_absorbing = huplc_abs_list
        model.add_component('huplc_absorbing', Set(initialize=self._huplc_absorbing))
        times = getattr(model, 'huplcmeas_times')
        timesh = getattr(model, 'huplctime')

        model.add_component('huplcabs_components_names', Set())
        huplcabscompsnames = [name for name in set(sorted(self._huplc_absorbing))]
        model.add_component('huplcabs_components', Set(initialize=huplcabscompsnames))

        self._solid_spec=solid_spec
        if solid_spec is not None:
            solid_spec_arg1keys = [key for key in set(sorted(self._solid_spec.keys()))]
            model.add_component('solid_spec_arg1', Set(initialize=solid_spec_arg1keys))
            solid_spec_arg2keys = [key for key in set(sorted(self._solid_spec.values()))]
            model.add_component('solid_spec_arg2', Set(initialize=solid_spec_arg2keys))
            model.add_component('solid_spec', Var(model.solid_spec_arg1, model.solid_spec_arg2, initialize=self._solid_spec))

        Dhat_dict=dict()
        for k in self._huplc_data.columns:
            for c in self._huplc_data.index:
                Dhat_dict[c, k] = float(self._huplc_data[k][c])

        model.add_component('fixed_Dhat', ConstraintList())
        model.add_component('Dhat', Var(times, huplcabscompsnames,initialize=Dhat_dict))
        model.add_component('Chat', Var(timesh, huplcabscompsnames,initialize=1))
        model.add_component('matchChatZ', ConstraintList())
        
        return None
