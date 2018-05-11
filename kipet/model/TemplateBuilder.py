import six
import pandas as pd
import itertools
import inspect
import numbers
import copy
import logging
import warnings

from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import sys

try:
    if sys.version_info.major > 3:
        import importlib
        importlib.util.find_spec("casadi")
    else:
        import imp
        imp.find_module('casadi')
    from kipet.model.CasadiModel import CasadiModel
    from kipet.model.CasadiModel import KipetCasadiStruct
    found_casadi = True
except ImportError:
    found_casadi = False

logger = logging.getLogger('ModelBuilderLogger')


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
        self._parameters_bounds = dict()
        self._init_conditions = dict()
        self._spectral_data = None
        self._absorption_data = None
        self._odes = None
        self._meas_times = set()
        self._complementary_states = set()
        self._algebraics = set()
        self._algebraic_constraints = None
        self._non_absorbing = None
        self._is_non_abs_set = False

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

        algebraics = kwargs.pop('algebraics', set())
        for y in algebraics:
            self._algebraics.add(y)

    def add_parameter(self, *args, **kwds):
        """Add a kinetic parameter(s) to the model.

        Note:
            Plan to change this method add parameters as PYOMO variables
            
            This method tries to mimic a template implmenetation. Depending 
            on the argument type it will behave differently

        Args:
            param1 (str): Parameter name. Creates a variable parameter  
            
            param1 (list): Parameter names. Creates a list of variable parameters  
            
            param1 (dict): Map parameter name(s) to value(s). Creates a fixed parameter(s)

        Returns:
            None

        """
        bounds = kwds.pop('bounds', None)

        if len(args) == 1:
            name = args[0]
            if isinstance(name, six.string_types):
                self._parameters[name] = None
                if bounds is not None:
                    self._parameters_bounds[name] = bounds
            elif isinstance(name, list) or isinstance(name, set):
                if bounds is not None:
                    if len(bounds) != len(name):
                        raise RuntimeError('the list of bounds must be equal to the list of parameters')
                for i, n in enumerate(name):
                    self._parameters[n] = None
                    if bounds is not None:
                        self._parameters_bounds[n] = bounds[i]
            elif isinstance(name, dict):
                if bounds is not None:
                    if len(bounds) != len(name):
                        raise RuntimeError('the list of bounds must be equal to the list of parameters')
                for k, v in name.items():
                    self._parameters[k] = v
                    if bounds is not None:
                        self._parameters_bounds[k] = bounds[k]
            else:
                raise RuntimeError('Kinetic parameter data not supported. Try str')
        elif len(args) == 2:
            first = args[0]
            second = args[1]
            if isinstance(first, six.string_types):
                self._parameters[first] = second
                if bounds is not None:
                    self._parameters_bounds[first] = bounds
            else:
                raise RuntimeError('Parameter argument not supported. Try str,val')
        else:
            raise RuntimeError('Parameter argument not supported. Try str,val')

    def add_mixture_component(self, *args):
        """Add a component (reactive or product) to the model.

        This method will keep track of the number of components in the model
        It will hel creating the Z,C and S variables. 

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
        if len(args) == 1:
            input = args[0]
            if isinstance(input, dict):
                for key, val in input.items():
                    if not isinstance(val, numbers.Number):
                        raise RuntimeError('The init condition must be a number. Try str, float')
                    self._component_names.add(key)
                    self._init_conditions[key] = val
            else:
                raise RuntimeError('Mixture component data not supported. Try dict[str]=float')
        elif len(args) == 2:
            name = args[0]
            init_condition = args[1]

            if not isinstance(init_condition, numbers.Number):
                raise RuntimeError('The second argument must be a number. Try str, float')

            if isinstance(name, six.string_types):
                self._component_names.add(name)
                self._init_conditions[name] = init_condition
            else:
                raise RuntimeError('Mixture component data not supported. Try str, float')
        else:
            raise RuntimeError('Mixture component data not supported. Try str, float')

    def add_spectral_data(self, data):
        """Add spectral data 

        Args:
            data (DataFrame): DataFrame with measurement times as 
                              indices and wavelengths as columns. 

        Returns:
            None

        """
        if isinstance(data, pd.DataFrame):
            self._spectral_data = data
        else:
            raise RuntimeError('Spectral data format not supported. Try pandas.DataFrame')

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

    def add_measurement_times(self, times):
        """Add measurement times to the model 

        Args:
            times (array_like): measurement points 

        Returns:
            None

        """
        for t in times:
            self._meas_times.add(t)

    def add_complementary_state_variable(self, *args):
        """Add an extra state variable to the model.

        This method add new state variables to the model. Extra or complementary states because 
        concentrations are also state variables

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
        if len(args) == 1:
            input = args[0]
            if isinstance(input, dict):
                for key, val in input.items():
                    if not isinstance(val, numbers.Number):
                        raise RuntimeError('The init condition must be a number. Try str, float')
                    self._complementary_states.add(key)
                    self._init_conditions[key] = val
            else:
                raise RuntimeError('Complementary state data not supported. Try dict[str]=float')
        elif len(args) == 2:
            name = args[0]
            init_condition = args[1]

            if not isinstance(init_condition, numbers.Number):
                raise RuntimeError('The second argument must be a number. Try str, float')

            if isinstance(name, six.string_types):
                self._complementary_states.add(name)
                self._init_conditions[name] = init_condition
            else:
                raise RuntimeError('Complementary state data not supported. Try str, float')
        else:
            print(len(args))
            raise RuntimeError('Complementary state data not supported. Try str, float')

    def add_algebraic_variable(self, *args):
        """Add an algebraic variable to the model
            
            This method tries to mimic a template implmenetation. Depending 
            on the argument type it will behave differently

        Args:
            param1 (str): Variable name. Creates a variable parameter  
            
            param1 (list): Variable names. Creates a list of variable parameters  

        Returns:
            None

        """
        name = args[0]
        if isinstance(name, six.string_types):
            self._algebraics.add(name)
        elif isinstance(name, list) or isinstance(name, set):
            for y in name:
                self._algebraics.add(y)
        else:
            raise RuntimeError('To add an algebraic please pass name')

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

    def _validate_data(self, model, start_time, end_time):
        """Verify all inputs to the model make sense.

        This method is not suppose to be use by users. Only for developers use

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
                raise RuntimeError('Need to add measumerement times')

    def create_pyomo_model(self, start_time, end_time):
        """Create a pyomo model.

        This method is the core method for further simulation or optimization studies

        Args:
            start_time (float): initial time considered in the model

            end_time (float): final time considered in the model

        Returns:
            Pyomo ConcreteModel

        """
        # Model
        pyomo_model = ConcreteModel()

        # Sets
        pyomo_model.mixture_components = Set(initialize=self._component_names)
        pyomo_model.parameter_names = Set(initialize=self._parameters.keys())
        pyomo_model.complementary_states = Set(initialize=self._complementary_states)
        pyomo_model.states = pyomo_model.mixture_components | pyomo_model.complementary_states

        pyomo_model.algebraics = Set(initialize=self._algebraics)

        list_times = self._meas_times
        m_times = sorted(list_times)
        m_lambdas = list()
        if self._spectral_data is not None and self._absorption_data is not None:
            raise RuntimeError('Either add absorption data or spectral data but not both')

        if self._spectral_data is not None:
            list_times = list_times.union(set(self._spectral_data.index))
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
                raise RuntimeError('Measurement time {0} not within ({1},{2})'.format(m_times[0], start_time, end_time))
            if m_times[-1] > end_time:
                raise RuntimeError(
                    'Measurement time {0} not within ({1},{2})'.format(m_times[-1], start_time, end_time))

        pyomo_model.meas_times = Set(initialize=m_times, ordered=True)
        pyomo_model.meas_lambdas = Set(initialize=m_lambdas, ordered=True)

        pyomo_model.time = ContinuousSet(initialize=pyomo_model.meas_times,
                                         bounds=(start_time, end_time))

        # Variables
        pyomo_model.Z = Var(pyomo_model.time,
                            pyomo_model.mixture_components,
                            # bounds=(0.0,None),
                            initialize=1)

        pyomo_model.dZdt = DerivativeVar(pyomo_model.Z,
                                         wrt=pyomo_model.time)

        pyomo_model.P = Var(pyomo_model.parameter_names,
                            # bounds = (0.0,None),
                            initialize=1)
        # set bounds P
        for k, v in self._parameters_bounds.items():
            lb = v[0]
            ub = v[1]
            pyomo_model.P[k].setlb(lb)
            pyomo_model.P[k].setub(ub)

        pyomo_model.C = Var(pyomo_model.meas_times,
                            pyomo_model.mixture_components,
                            bounds=(0.0, None),
                            initialize=1)

        pyomo_model.X = Var(pyomo_model.time,
                            pyomo_model.complementary_states,
                            initialize=1.0)

        pyomo_model.dXdt = DerivativeVar(pyomo_model.X,
                                         wrt=pyomo_model.time)

        pyomo_model.Y = Var(pyomo_model.time,
                            pyomo_model.algebraics,
                            initialize=1.0)

        if self._absorption_data is not None:
            s_dict = dict()
            for k in self._absorption_data.columns:
                for l in self._absorption_data.index:
                    s_dict[l, k] = float(self._absorption_data[k][l])
        else:
            s_dict = 1.0

        pyomo_model.S = Var(pyomo_model.meas_lambdas,
                            pyomo_model.mixture_components,
                            bounds=(0.0, None),
                            initialize=s_dict)

        if self._absorption_data is not None:
            for l in pyomo_model.meas_lambdas:
                for k in pyomo_model.mixture_components:
                    pyomo_model.S[l, k].fixed = True

        # Parameters
        pyomo_model.init_conditions = Param(pyomo_model.states,
                                            initialize=self._init_conditions, mutable=True)
        pyomo_model.start_time = Param(initialize=start_time)
        pyomo_model.end_time = Param(initialize=end_time)

        # Fixes parameters that were given numeric values
        for p, v in self._parameters.items():
            if v is not None:
                pyomo_model.P[p].value = v
                pyomo_model.P[p].fixed = True

        # spectral data
        if self._spectral_data is not None:
            s_data_dict = dict()
            for t in pyomo_model.meas_times:
                for l in pyomo_model.meas_lambdas:
                    s_data_dict[t, l] = float(self._spectral_data[l][t])

            pyomo_model.D = Param(pyomo_model.meas_times,
                                  pyomo_model.meas_lambdas,
                                  initialize=s_data_dict)

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


            pyomo_model.odes = Constraint(pyomo_model.time,
                                          pyomo_model.states,
                                          rule=rule_odes)

        # the generation of the constraints is not efficient but not critical
        if self._algebraic_constraints:
            n_alg_eqns = len(self._algebraic_constraints(pyomo_model, start_time))

            def rule_algebraics(m, t, k):
                alg_const = self._algebraic_constraints(m, t)[k]
                return alg_const == 0.0

            pyomo_model.algebraic_consts = Constraint(pyomo_model.time,
                                                      range(n_alg_eqns),
                                                      rule=rule_algebraics)
        if self._is_non_abs_set:  #: in case of a second call after non_absorbing has been declared
            self.set_non_absorbing_species(pyomo_model, self._non_absorbing, check=False)
        return pyomo_model

    def create_casadi_model(self, start_time, end_time):
        """Create a casadi model.

        Casadi models are for simulation purpuses mainly

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
            casadi_model.meas_lambdas = m_lambdas

            # Variables                
            casadi_model.Z = KipetCasadiStruct('Z', list(casadi_model.mixture_components), dummy_index=True)
            casadi_model.X = KipetCasadiStruct('X', list(casadi_model.complementary_states), dummy_index=True)
            casadi_model.Y = KipetCasadiStruct('Y', list(casadi_model.algebraics), dummy_index=True)
            casadi_model.P = KipetCasadiStruct('P', list(casadi_model.parameter_names))
            casadi_model.C = KipetCasadiStruct('C', list(casadi_model.meas_times))
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

    def has_spectral_data(self):
        return self._spectral_data is not None

    def has_adsorption_data(self):
        return self._absorption_data is not None

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

        S = getattr(model, 'S')
        C = getattr(model, 'C')
        Z = getattr(model, 'Z')
        times = getattr(model, 'meas_times')
        lambdas = getattr(model, 'meas_lambdas')
        for component in self._non_absorbing:
            for l in lambdas:
                S[l, component].set_value(0)
                S[l, component].fix()

        model.add_component('fixed_C', ConstraintList())
        new_con = getattr(model, 'fixed_C')
        for time in times:
            for component in self._non_absorbing:
                new_con.add(C[time, component] == Z[time, component])

