# -*- coding: utf-8 -*-

from __future__ import print_function
from pyomo.environ import *
from pyomo.dae import *
from kipet.library.ResultsObject import *
from kipet.library.Simulator import *
from kipet.library.PyomoSimulatorChange import *
#from kipet.library.fe_factory_0fix import *
#from kipet.library.fe_factory_feed_v2_timepluspatchwithfixgeneralcleanmult import *
from kipet.library.fe_factory_feed_v2_timepluspatchwithfixgeneralcleanmult0fixclean import *
import warnings
import six
import sys

__author__ = 'Michael Short'  #: July 2018

class FESimulator(PyomoSimulator):
    def __init__(self, model):
        """
            FESimulator class:
    
                This class is just an interface that allows the user of Kipet to easily implement the more general 
                fe_factory class designed by David M. Thierry without having to re-write the model to fit those
                arguments. It takes in a standard Kipet/Pyomo model, rewrites it and calls fe_factory.
                More information on fe_factory is included in that class description.
    
                Args:
                    model (ConcreteModel): The original Pyomo model created in the Kipet script
        """
        super(FESimulator, self).__init__(model)
        self.p_sim =  PyomoSimulator(model)
        self.c_sim = self.p_sim.model.clone()
        self.param_dict = {}
        self.param_name = "P"

        # check all parameters are fixed before simulating
        for p_sim_data in six.itervalues(self.p_sim.model.P):
            if not p_sim_data.fixed:
                raise RuntimeError('For simulation fix all parameters. Parameter {} is unfixed'.format(p_sim_data.cname()))

        #Build the parameter dictionary in the format that fe_factory uses    
        for k,v in six.iteritems(self.p_sim.model.P):
            self.param_dict["P",k] = v.value

        #Build the initial condition dictionary in the format that fe_factory uses
        self.ics_ = {} 

        for t, k in six.iteritems(self.p_sim.model.Z):
            st = self.p_sim.model.start_time
            if t[0] == st:
                self.ics_['Z', t[1]] = k.value
                
        #Now to set the additional state values
        for t, v in six.iteritems(self.p_sim.model.X):
            if t[0] == st:
                self.ics_['X',t[1]] = v.value

    def call_fe_factory(self, inputs_sub=None, jump_states=None, jump_times=None, feed_times=None, fixedtraj=False, fixedy=False, yfix=None, yfixtraj=None):#added for inclusion of discrete jumps CS
        """
        call_fe_factory:
    
                This function applies all the inputs necessary for fe_factory to work, using Kipet syntax.
    
                Args:
                    inputs_sub (dict): dictionary of inputs
                    jump_states (dict): dictionary of which variables and states are inputted and by how much
                    jump_times (dict): dictionary in same form as jump_states with times of input
                    feed_times (list): list of additional times needed, should be the same times as jump_times
        """
        #added for inclusion of inputs of different kind CS
        self.inputs_sub = None
        self.inputs_sub=inputs_sub
        self.fixedtraj=fixedtraj
        self.fixedy=fixedy
        self.yfix = yfix
        self.yfixtraj = yfixtraj

        self.jump_times=jump_times #added for inclusion of discrete jumps CS
        self.jump_states=jump_states
        self.feed_times=feed_times
        init = fe_initialize(self.p_sim.model, self.c_sim,
                         init_con="init_conditions_c",
                         param_name=self.param_name,
                         param_values=self.param_dict,
                         inputs_sub=self.inputs_sub)
    
        init.load_initial_conditions(init_cond=self.ics_)

        if jump_times!=None and jump_states!=None:
            init.load_discrete_jump(jump_states, jump_times, feed_times) #added for inclusion of discrete jumps
        init.run()
