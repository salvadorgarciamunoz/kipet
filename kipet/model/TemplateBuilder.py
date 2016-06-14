import six
import pandas as pd
import itertools
import copy
import logging

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
        self._odes = None
        self._meas_times = []
        
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
            print len(args)
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
            self._meas_times.append(t)

    def set_rule_ode_expressions_dict(self,rule):
        self._odes = rule
        
    def create_pyomo_model(self,start_time,end_time):
        # Model
        pyomo_model = ConcreteModel()
        
        # Sets
        pyomo_model.mixture_components = Set(initialize = self._component_names)
        pyomo_model.parameter_names = Set(initialize = self._parameters.keys())
        
        m_times = list()
        m_lambdas = list()
        if self._spectral_data is not None and self._absorption_data is not None:
            raise RuntimeError('Either add absorption data or spectral data but not both')

        if self._spectral_data is not None:
            m_times = list(self._spectral_data.index)
            m_lambdas = list(self._spectral_data.columns)
        if self._absorption_data is not None:
            if not self._meas_times:
                raise RuntimeError('Need to add measumerement times')
            m_times = list(self._meas_times)
            m_lambdas = list(self._absorption_data.index)

        pyomo_model.measurement_times = Set(initialize = m_times)
        pyomo_model.measurement_lambdas = Set(initialize = m_lambdas)
        
            
        
        pyomo_model.time = ContinuousSet(initialize = pyomo_model.measurement_times,bounds = (start_time,end_time))

        # Variables
        pyomo_model.C = Var(pyomo_model.time,
                            pyomo_model.mixture_components,
                            bounds=(0.0,None),
                            initialize=1)
        
        pyomo_model.dCdt = DerivativeVar(pyomo_model.C,
                                         wrt=pyomo_model.time)

        pyomo_model.P = Var(pyomo_model.parameter_names,
                            initialize=1)
        
        pyomo_model.C_noise = Var(pyomo_model.measurement_times,
                                  pyomo_model.mixture_components,
                                  bounds=(0.0,None),
                                  initialize=1)

        if self._absorption_data is not None:
            s_dict = dict()
            for k in self._absorption_data.columns:
                for l in self._absorption_data.index:
                    s_dict[l,k] = float(self._absorption_data[k][l])
        else:
            s_dict = 1.0
            
        pyomo_model.S = Var(pyomo_model.measurement_lambdas,
                            pyomo_model.mixture_components,
                            bounds=(0.0,None),
                            initialize=s_dict)

        if self._absorption_data is not None:
            for l in pyomo_model.measurement_lambdas:
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
            for t in pyomo_model.measurement_times:
                for l in pyomo_model.measurement_lambdas:
                    s_data_dict[t,l] = float(self._spectral_data[l][t])

            pyomo_model.spectral_data = Param(pyomo_model.measurement_times,
                                              pyomo_model.measurement_lambdas,
                                              initialize = s_data_dict)

        def rule_init_conditions(model,k):
            st = start_time
            return model.C[st,k] == self._init_conditions[k]
        pyomo_model.init_conditions_c = \
            Constraint(self._component_names,rule=rule_init_conditions)

        # add ode contraints to pyomo model
        def rule_mass_balances(m,t,k):
            exprs = self._odes(m,t)
            if t == m.start_time.value:
                return Constraint.Skip
            else:
                return m.dCdt[t,k] == exprs[k] 
        pyomo_model.mass_balances = Constraint(pyomo_model.time,
                                     pyomo_model.mixture_components,
                                     rule=rule_mass_balances)
        
        return pyomo_model

    def create_casadi_model(self,start_time,end_time):

        # Model
        casadi_model = CasadiModel()
        
        # Sets
        casadi_model.mixture_components = copy.deepcopy(self._component_names)
        casadi_model.parameter_names = self._parameters.keys()
        
        m_times = list()
        m_lambdas = list()
        if self._spectral_data is not None and self._absorption_data is not None:
            raise RuntimeError('Either add absorption data or spectral data but not both')

        if self._spectral_data is not None:
            m_times = list(self._spectral_data.index)
            m_lambdas = list(self._spectral_data.columns)
        if self._absorption_data is not None:
            if not self._meas_times:
                raise RuntimeError('Need to add measumerement times')
            m_times = list(self._meas_times)
            m_lambdas = list(self._absorption_data.index)

        casadi_model.measurement_times = m_times
        casadi_model.measurement_lambdas = m_lambdas

        # Variables                
        casadi_model.C = KinetCasadiStruct('C',list(casadi_model.mixture_components),dummy_index=True)
        casadi_model.P = KinetCasadiStruct('P',list(casadi_model.parameter_names))
        casadi_model.C_noise = KinetCasadiStruct('C_noise',list(casadi_model.measurement_times))
        casadi_model.S = KinetCasadiStruct('S',list(casadi_model.measurement_lambdas))
        
        # Parameters
        casadi_model.init_conditions = self._init_conditions
        casadi_model.start_time = start_time
        casadi_model.end_time = end_time
        
        if self._absorption_data is not None:
            for l in casadi_model.measurement_lambdas:
                for k in casadi_model.mixture_components:
                    casadi_model.S[l,k] = float(self._absorption_data[k][l])

        if self._spectral_data is not None:
            casadi_model.spectral_data = dict()
            for t in casadi_model.measurement_times:
                for l in casadi_model.measurement_lambdas:
                    casadi_model.spectral_data[t,l] = float(self._spectral_data[l][t])
        
        # Fixes parameters that were given numeric values
        for p,v in self._parameters.iteritems():
            if v is not None:
                casadi_model.P[p] = v
                
        # ignores the time indes t=0
        casadi_model.diff_exprs = self._odes(casadi_model,0)
        
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
