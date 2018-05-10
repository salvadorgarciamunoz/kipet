#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.utils.data_tools import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from kipet.utils.fe_factory import *
from pyomo.opt import *
import pickle

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
            
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================
     
    # create template model 
    builder = TemplateBuilder()

    # components
    components = dict()
    components['AH'] = 0.395555
    components['B'] = 0.0351202
    components['C'] = 0.0
    components['BH+'] = 0.0
    components['A-'] = 0.0
    components['AC-'] = 0.0
    components['P'] = 0.0

    builder.add_mixture_component(components)

    # add algebraics
    algebraics = [0, 1, 2, 3, 4, 5]  # the indices of the rate rxns
    # note the fifth component. Which basically works as an input

    builder.add_algebraic_variable(algebraics)

    params = dict()
    params['k0'] = 49.7796
    params['k1'] = 8.93156
    params['k2'] = 1.31765
    params['k3'] = 0.310870
    params['k4'] = 3.87809

    builder.add_parameter(params)

    # add additional state variables
    extra_states = dict()
    extra_states['V'] = 0.0629418

    builder.add_complementary_state_variable(extra_states)

    # stoichiometric coefficients
    gammas = dict()
    gammas['AH'] = [-1, 0, 0, -1, 0]
    gammas['B'] = [-1, 0, 0, 0, 1]
    gammas['C'] = [0, -1, 1, 0, 0]
    gammas['BH+'] = [1, 0, 0, 0, -1]
    gammas['A-'] = [1, -1, 1, 1, 0]
    gammas['AC-'] = [0, 1, -1, -1, -1]
    gammas['P'] = [0, 0, 0, 1, 1]

    def rule_algebraics(m, t):
        r = list()
        r.append(m.Y[t, 0] - m.P['k0'] * m.Z[t, 'AH'] * m.Z[t, 'B'])
        r.append(m.Y[t, 1] - m.P['k1'] * m.Z[t, 'A-'] * m.Z[t, 'C'])
        r.append(m.Y[t, 2] - m.P['k2'] * m.Z[t, 'AC-'])
        r.append(m.Y[t, 3] - m.P['k3'] * m.Z[t, 'AC-'] * m.Z[t, 'AH'])
        r.append(m.Y[t, 4] - m.P['k4'] * m.Z[t, 'AC-'] * m.Z[t, 'BH+'])
        return r
    #: there is no ae for Y[t,5] because step equn under rule_odes functions as the switch for the "C" equation

    builder.set_algebraics_rule(rule_algebraics)


    def rule_odes(m, t):
        exprs = dict()
        eta = 1e-2
        #: @dthierry: This thingy is now framed in terms of `m.Y[t, 5]`
        step = 0.5 * ((m.Y[t, 5] + 1) / ((m.Y[t, 5] + 1) ** 2 + eta ** 2) ** 0.5 + (210.0 - m.Y[t,5]) / ((210.0 - m.Y[t, 5]) ** 2 + eta ** 2) ** 0.5)
        exprs['V'] = 7.27609e-05 * step
        V = m.X[t, 'V']
        # mass balances
        for c in m.mixture_components:
            exprs[c] = sum(gammas[c][j] * m.Y[t, j] for j in m.algebraics if j != 5) - exprs['V'] / V * m.Z[t, c]
            if c == 'C':
                exprs[c] += 0.02247311828 / (m.X[t, 'V'] * 210) * step
        return exprs


    builder.set_odes_rule(rule_odes)

    filename = 'trimmed.csv'
    D_frame = read_spectral_data_from_csv(filename)
    meas_times = sorted(D_frame.index)

    model = builder.create_pyomo_model(0, 600)

    #=========================================================================
    #USER INPUT SECTION - FE Factory
    #=========================================================================
     

    sim = PyomoSimulator(model)
    mod = sim.model.clone()
    
    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')

    #: we now need to explicitly tell the initial conditions and parameter values
    param_name = "P"
    param_dict = {}
    param_dict["P", "k0"] = 49.7796
    param_dict["P", "k1"] = 8.93156
    param_dict["P", "k2"] = 1.31765
    param_dict["P", "k3"] = 0.310870
    param_dict["P", "k4"] = 3.87809

    ics_ = dict()
    ics_['Z', 'AH'] = 0.395555
    ics_['Z', 'B'] = 0.0351202
    ics_['Z', 'C'] = 0.0
    ics_['Z', 'BH+'] = 0.0
    ics_['Z', 'A-'] = 0.0
    ics_['Z', 'AC-'] = 0.0
    ics_['Z', 'P'] = 0.0
    ics_['X', 'V'] = 0.0629418

    inputs_sub = {}
    inputs_sub['Y'] = [5]

    #: define the values for our simulation
    for key in sim.model.time.value:
        sim.model.Y[key, 5].set_value(key)
        sim.model.Y[key, 5].fix()  #if you don't fix this, fe_factory is will not work complain.

    init = fe_initialize(sim.model, mod,
                         init_con="init_conditions_c",
                         param_name=param_name,
                         param_values=param_dict,
                         inputs_sub=inputs_sub)
    
    init.load_initial_conditions(init_cond=ics_)
   
    init.run()

    #=========================================================================
    #USER INPUT SECTION - SIMULATION
    #=========================================================================
     
    #: now the final run, just as before
    # simulate
    options = {}
    results = sim.run_sim('ipopt',
                          tee=True,
                          solver_opts=options)
    if with_plots:
        # display concentration results    
        results.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        plt.show()
    
        #results.Y[0].plot.line()
        results.Y[1].plot.line()
        results.Y[2].plot.line()
        results.Y[3].plot.line()
        results.Y[4].plot.line()
        plt.xlabel("time (s)")
        plt.ylabel("rxn rates (mol/L*s)")
        plt.title("Rates of rxn")
        plt.show()

    #D_frame.plot.line(legend=False)
    #plt.show()
    
    #plot_spectral_data(D_frame,dimension='3D')
    
    #results.Z.to_csv("initialization.csv")
