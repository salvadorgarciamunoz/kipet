import six
import pandas as pd
import itertools
import copy
import logging
import warnings

from pyomo.environ import *
from pyomo.dae import *
import numpy as np
from CasadiModel import *


logger = logging.getLogger('ModelBuilderLogger')

class TemplateBuilder(object):
    
    def __init__(self):
        self._component_names = set()
        self._parameters = dict()
        self._init_conditions = dict()
        self._spectral_data = None
        self._absorption_data = None
        self._mass_balances = None
        self._meas_times = set()
        self._complementary_states = dict()
        self._odes = None
        
    def add_parameter(self,*args):
        if len(args) == 1:
            name = args[0]
            if isinstance(name,six.string_types):
                self._parameters[name] = None
            elif isinstance(name,list) or isinstance(name,set):
                for n in name:
                    self._parameters[n] = None
            elif isinstance(name,dict):
                for k,v in name.iteritems():
                    self._parameters[k] = v
            else:
                raise RuntimeError('Kinetic parameter data not supported. Try str')
        elif len(args) == 2:
            first = args[0]
            second = args[1]
            if isinstance(first,six.string_types):
                self._parameters[first] = second
            else:
                raise RuntimeError('Parameter argument not supported. Try str,val')
        else:
            raise RuntimeError('Parameter argument not supported. Try str,val')
        
    def add_mixture_component(self,*args):
        if len(args) == 1:
            input = args[0]
            if isinstance(input,dict):
                for key,val in input.iteritems():
                    self._component_names.add(key)
                    self._init_conditions[key] = val
            else:
                raise RuntimeError('Mixture component data not supported. Try dict[str]=float')
        elif len(args)==2:
            name = args[0]
            init_condition = args[1]
            if isinstance(name,six.string_types):
                self._component_names.add(name)
                self._init_conditions[name] = init_condition
            else:
                raise RuntimeError('Mixture component data not supported. Try str, float')
        else:
            print(len(args))
            raise RuntimeError('Mixture component data not supported. Try str, float')
            
    def add_spectral_data(self,data):
        if isinstance(data,pd.DataFrame):
            self._spectral_data = data
        else:
            raise RuntimeError('Spectral data format not supported. Try pandas.DataFrame')
        
    def add_absorption_data(self,data):
        if isinstance(data,pd.DataFrame):
            self._absorption_data = data
        else:
            raise RuntimeError('Spectral data format not supported. Try pandas.DataFrame')

    def add_measurement_times(self,times):
        for t in times:
            self._meas_times.add(t)

    def add_complementary_state_variable(self,*args):
        if len(args) == 1:
            input = args[0]
            if isinstance(input,dict):
                for key,val in input.iteritems():
                    self._complementary_states[key] = val
            else:
                raise RuntimeError('Complementary state data not supported. Try dict[str]=float')
        elif len(args)==2:
            name = args[0]
            init_condition = args[1]
            if isinstance(name,six.string_types):
                self._complementary_states[name] = init_condition
            else:
                raise RuntimeError('Complementary state data not supported. Try str, float')
        else:
            print(len(args))
            raise RuntimeError('Complementary state data not supported. Try str, float')
        
    def set_mass_balances_rule(self,rule):
        self._mass_balances = rule

    def set_complementary_ode_rule(self,rule):
        self._odes = rule

    def _validate_data(self,model,start_time,end_time):
        if not self._component_names:
            raise warnings.warn('The Model does not have any mixture components')
        else:
            dummy_balances = self._mass_balances(model,start_time)
            if len(self._component_names)!=len(dummy_balances):
                raise RuntimeError('The number of mixture components needs to be the same \
                as the number of mass balances.\n Use set_mass_balances_rule')
        if len(self._complementary_states):
            dummy_balances = self._odes(start_time)
            if len(self._complementary_states)!=len(model,dummy_balances):
                raise RuntimeError('The number of state variables (excluding Z) needs to be the same \
                as the number complementary odes.\n Use set_complementary_ode_rule')

        if self._absorption_data is not None:
            if not self._meas_times:
                raise RuntimeError('Need to add measumerement times')
        
    def create_pyomo_model(self,start_time,end_time):
        # Model
        pyomo_model = ConcreteModel()
        
        # Sets
        pyomo_model.mixture_components = Set(initialize = self._component_names)
        pyomo_model.parameter_names = Set(initialize = self._parameters.keys())
        pyomo_model.complementary_states = Set(initialize = self._complementary_states.keys())
        
        list_times = list(self._meas_times)
        self._meas_times = sorted(list_times)
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
            if m_times[0]<start_time:
                raise RuntimeError('Measurement time {0} not within ({1},{2})'.format(m_times[0],start_time,end_time))
            if m_times[-1]>end_time:
                raise RuntimeError('Measurement time {0} not within ({1},{2})'.format(m_times[-1],start_time,end_time))

        pyomo_model.meas_times = Set(initialize = m_times,ordered=True)
        pyomo_model.meas_lambdas = Set(initialize = m_lambdas,ordered=True)
        
        pyomo_model.time = ContinuousSet(initialize = pyomo_model.meas_times,bounds = (start_time,end_time))

        # Variables
        pyomo_model.Z = Var(pyomo_model.time,
                            pyomo_model.mixture_components,
                            bounds=(0.0,None),
                            initialize=1)
        
        pyomo_model.dZdt = DerivativeVar(pyomo_model.Z,
                                         wrt=pyomo_model.time)

        pyomo_model.P = Var(pyomo_model.parameter_names,
                            initialize=1)
        
        pyomo_model.C = Var(pyomo_model.meas_times,
                                  pyomo_model.mixture_components,
                                  bounds=(0.0,None),
                                  initialize=1)

        pyomo_model.X = Var(pyomo_model.time,
                            pyomo_model.complementary_states,
                            initialize = 1)

        pyomo_model.dXdt = DerivativeVar(pyomo_model.X,
                                         wrt=pyomo_model.time)

        if self._absorption_data is not None:
            s_dict = dict()
            for k in self._absorption_data.columns:
                for l in self._absorption_data.index:
                    s_dict[l,k] = float(self._absorption_data[k][l])
        else:
            s_dict = 1.0
            
        pyomo_model.S = Var(pyomo_model.meas_lambdas,
                            pyomo_model.mixture_components,
                            bounds=(0.0,None),
                            initialize=s_dict)

        if self._absorption_data is not None:
            for l in pyomo_model.meas_lambdas:
                for k in pyomo_model.mixture_components:
                    pyomo_model.S[l,k].fixed = True
        
        # Parameters
        pyomo_model.init_conditions = Param(pyomo_model.mixture_components,
                                            initialize=self._init_conditions)
        pyomo_model.start_time = Param(initialize = start_time)
        pyomo_model.end_time = Param(initialize = end_time)
        
        
        # Fixes parameters that were given numeric values
        for p,v in self._parameters.iteritems():
            if v is not None:
                pyomo_model.P[p].value = v
                pyomo_model.P[p].fixed = True
        
        # spectral data
        if self._spectral_data is not None:
            s_data_dict = dict()
            for t in pyomo_model.meas_times:
                for l in pyomo_model.meas_lambdas:
                    s_data_dict[t,l] = float(self._spectral_data[l][t])

            pyomo_model.D = Param(pyomo_model.meas_times,
                                  pyomo_model.meas_lambdas,
                                  initialize = s_data_dict)

        # validate the model before writing constraints
        self._validate_data(pyomo_model,start_time,end_time)
            
        # add ode contraints to pyomo model
        def rule_init_conditions(model,k):
            st = start_time
            return model.Z[st,k] == self._init_conditions[k]
        pyomo_model.init_conditions_c = \
            Constraint(self._component_names,rule=rule_init_conditions)

        def rule_mass_balances(m,t,k):
            exprs = self._mass_balances(m,t)
            if t == m.start_time.value:
                return Constraint.Skip
            else:
                return m.dZdt[t,k] == exprs[k] 
        pyomo_model.mass_balances = Constraint(pyomo_model.time,
                                     pyomo_model.mixture_components,
                                     rule=rule_mass_balances)

        def rule_init_conditions_x(model,k):
            st = start_time
            return model.X[st,k] == self._complementary_states[k]
        pyomo_model.init_conditions_x = \
            Constraint(pyomo_model.complementary_states,
                       rule=rule_init_conditions_x)
        
        def rule_complementary_odes(m,t,n):
            exprs = self._odes(m,t)
            if t == m.start_time.value:
                return Constraint.Skip
            else:
                return m.dXdt[t,n] == exprs[n]
        pyomo_model.complementary_odes = Constraint(pyomo_model.time,
                                                    pyomo_model.complementary_states,
                                                    rule=rule_complementary_odes)
            
        return pyomo_model

    def create_casadi_model(self,start_time,end_time):

        # Model
        casadi_model = CasadiModel()
        
        # Sets
        casadi_model.mixture_components = copy.deepcopy(self._component_names)
        casadi_model.parameter_names = self._parameters.keys()
        casadi_model.complementary_states = self._complementary_states.keys()
        
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
            if m_times[0]<start_time:
                raise RuntimeError('Measurement time {0} not within ({1},{2})'.format(m_times[0],start_time,end_time))
            if m_times[-1]>end_time:
                raise RuntimeError('Measurement time {0} not within ({1},{2})'.format(m_times[-1],start_time,end_time))

        casadi_model.meas_times = m_times
        casadi_model.meas_lambdas = m_lambdas

        # Variables                
        casadi_model.Z = KinetCasadiStruct('Z',list(casadi_model.mixture_components),dummy_index=True)
        casadi_model.X = KinetCasadiStruct('X',list(casadi_model.complementary_states),dummy_index=True)
        casadi_model.P = KinetCasadiStruct('P',list(casadi_model.parameter_names))
        casadi_model.C = KinetCasadiStruct('C',list(casadi_model.meas_times))
        casadi_model.S = KinetCasadiStruct('S',list(casadi_model.meas_lambdas))

        
        # Parameters
        casadi_model.init_conditions = self._init_conditions
        casadi_model.init_conditions_x = copy.deepcopy(self._complementary_states)
        casadi_model.start_time = start_time
        casadi_model.end_time = end_time
        
        if self._absorption_data is not None:
            for l in casadi_model.meas_lambdas:
                for k in casadi_model.mixture_components:
                    casadi_model.S[l,k] = float(self._absorption_data[k][l])

        if self._spectral_data is not None:
            casadi_model.D = dict()
            for t in casadi_model.meas_times:
                for l in casadi_model.meas_lambdas:
                    casadi_model.D[t,l] = float(self._spectral_data[l][t])
        
        # Fixes parameters that were given numeric values
        for p,v in self._parameters.iteritems():
            if v is not None:
                casadi_model.P[p] = v

        # validate the model before writing constraints
        self._validate_data(casadi_model,start_time,end_time)
                
        # ignores the time indes t=0
        casadi_model.mass_balance_exprs = self._mass_balances(casadi_model,0)
        if self._complementary_states:
            casadi_model.complementary_odes = self._odes(casadi_model,0) 
        return casadi_model

    def write_dat_file(self,filename,start_time,end_time,fixed_dict=None):
        f = open(filename,'w')
        f.write('# abstract PyomoConcentrationModel.dat AMPL format\n')
        # Sets
        f.write('set parameter_names := ')
        for name in self._parameter_names:
            f.write('{} '.format(name))
        f.write(';\n')
        
        f.write('set component_names := ')
        for name in self._component_names:
            f.write('{} '.format(name))
        f.write(';\n')

        if fixed_dict:
            f.write('set fixed_parameter_names := ')
            for key,val in fixed_dict.iteritems():
                f.write('{0} '.format(key))
            f.write(';\n')
        else:
            f.write('set fixed_parameter_names := ;\n')


        f.write('set time := {0} {1};\n\n'.format(start_time,end_time))

        # Params
        f.write('param start_time := {};\n'.format(start_time))
        f.write('param end_time := {};\n'.format(end_time))
        
        f.write('param init_conditions := \n')
        for key,val in self._init_conditions.iteritems():
            f.write('{0} {1}\n'.format(key,val))
        f.write(';\n')

        if fixed_dict:
            f.write('param fixed_parameters := \n')
            for key,val in fixed_dict.iteritems():
                f.write('{0} {1}\n'.format(key,val))
            f.write(';\n')
        else:
            f.write('param fixed_parameters := ;\n')
        f.close()
