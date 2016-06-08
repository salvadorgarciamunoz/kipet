from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import logging
import pandas as pd
from ModelBuilderHelper import *
logger = logging.getLogger('ConcentrationModel')

class PyomoModelBuilderHelper(ModelBuilderHelper):
    
    def __init__(self):
        super(PyomoModelBuilderHelper, self).__init__()

    def create_pyomo_concrete_model(self,start_time,end_time,fixed_dict=None):

        # Model
        pyomo_model = ConcreteModel()
        
        # Sets
        pyomo_model.time = ContinuousSet(bounds=(start_time,end_time))
        pyomo_model.mixture_component_names = Set(initialize=self._component_names)
        pyomo_model.parameter_names = Set(initialize=self._paramemter_names)

        # Variables
        pyomo_model.C = Var(pyomo_model.time,
                            self._component_names,
                            bounds=(0.0,None),
                            initialize=1)
        
        pyomo_model.dCdt = DerivativeVar(pyomo_model.C,
                                         wrt=pyomo_model.time)

        pyomo_model.kinetic_parameter = Var(self._paramemter_names,
                                            initialize=1)

        # Parameters
        pyomo_model.init_conditions = Param(pyomo_model.mixture_component_names,
                                            initialize=self._init_conditions)
        pyomo_model.start_time = Param(initialize = start_time)
        pyomo_model.end_time = Param(initialize = end_time)
        
        def rule_init_conditions(model,k):
            st = start_time
            return model.C[st,k] == self._init_conditions[k]
        pyomo_model.init_conditions_c = \
            Constraint(self._component_names,rule=rule_init_conditions)

        if fixed_dict:
            # TODO: check if keys are parameters
            for key,val in fixed_dict.iteritems():
                pyomo_model.kinetic_parameter[key].value = val
                pyomo_model.kinetic_parameter[key].fixed = True

        return pyomo_model
        
    def write_dat_file(self,filename,start_time,end_time,fixed_dict=None):
        f = open(filename,'w')
        f.write('# abstract PyomoConcentrationModel.dat AMPL format\n')
        # Sets
        f.write('set parameter_names := ')
        for name in self._paramemter_names:
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

    

if __name__ == "__main__":
    
    helper = PyomoModelBuilderHelper()
    
    helper.add_mixture_component('A',1)
    helper.add_mixture_component('B',0)
    helper.add_kinetic_parameter('k')
    
    fix_dict = {'k':0.01}
    model = helper.create_pyomo_concrete_model(0.0,200.0,fixed_dict=fix_dict)

    def rule_mass_A(m,t):
        if t == m.start_time:
            return Constraint.Skip
        else:
            return m.dCdt[t,'A']== -m.kinetic_parameter['k']*m.C[t,'A']
    model.mass_balance_A = Constraint(model.time,rule=rule_mass_A)
    
    def rule_mass_B(m,t):
        if t == m.start_time:
            return Constraint.Skip
        else:
            return m.dCdt[t,'B']== m.kinetic_parameter['k']*m.C[t,'A']
    model.mass_balance_B = Constraint(model.time,rule=rule_mass_B)
    
    model.pprint()
    
    
