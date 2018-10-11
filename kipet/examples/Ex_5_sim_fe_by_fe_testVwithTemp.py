#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from kipet.library.TemplateBuilderChange import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.data_tools import *
from kipet.library.FESimulatormult import *
from pyomo.opt import *
import pickle
import os

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
    algebraics = ['0', '1', '2', '3', '4', '5', 'k4T', 'Temp']  # the indices of the rate rxns
    # note the fifth, sixth and seventh components. Which basically work as inputs

    builder.add_algebraic_variable(algebraics)

    # Load Temp data:
    dataDirectory = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe()))), 'data_sets'))
    Ttraj = os.path.join(dataDirectory, 'Tempvalues.csv')
    fixed_Ttraj = read_absorption_data_from_csv(Ttraj)

    params = dict()
    params['k0'] = 49.7796
    params['k1'] = 8.93156
    params['k2'] = 1.31765
    params['k3'] = 0.310870
    params['k4Tr'] = 3.87809
    params['E'] = 20.

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
        r.append(m.Y[t, '0'] - m.P['k0'] * m.Z[t, 'AH'] * m.Z[t, 'B'])
        r.append(m.Y[t, '1'] - m.P['k1'] * m.Z[t, 'A-'] * m.Z[t, 'C'])
        r.append(m.Y[t, '2'] - m.P['k2'] * m.Z[t, 'AC-'])
        r.append(m.Y[t, '3'] - m.P['k3'] * m.Z[t, 'AC-'] * m.Z[t, 'AH'])
        r.append(m.Y[t, '4'] - m.Y[t, 'k4T'] * m.Z[t, 'AC-'] * m.Z[t, 'BH+'])  # 6 is k4T
        Tr = 303.15
        # add temperature dependence via Arrhenius law
        r.append(m.Y[t, 'k4T'] - m.P['k4Tr'] * exp(m.P['E'](1 / m.Y[t, 'Temp'] - 1 / Tr)))
        return r


    #: there is no ae for Y[t,5] because step equn under rule_odes functions as the switch for the "C" equation

    builder.set_algebraics_rule(rule_algebraics)


    def rule_odes(m, t):
        exprs = dict()
        eta = 1e-2
        #: @dthierry: This thingy is now framed in terms of `m.Y[t, 5]`
        step = 0.5 * ((m.Y[t, '5'] + 1) / ((m.Y[t, '5'] + 1) ** 2 + eta ** 2) ** 0.5 + (210.0 - m.Y[t, '5']) / (
                    (210.0 - m.Y[t, '5']) ** 2 + eta ** 2) ** 0.5)
        exprs['V'] = 7.27609e-05 * step
        V = m.X[t, 'V']
        # mass balances
        for c in m.mixture_components:
            exprs[c] = gammas[c][0] * m.Y[t, '0'] + gammas[c][1] * m.Y[t, '1'] + gammas[c][2] * m.Y[t, '2'] + gammas[c][
                3] * m.Y[t, '3'] + gammas[c][4] * m.Y[t, '4'] - exprs['V'] / V * m.Z[t, c]
            if c == 'C':
                exprs[c] += 0.02247311828 / (m.X[t, 'V'] * 210) * step
        return exprs


    builder.set_odes_rule(rule_odes)

    # Add time points where feed as discrete jump should take place:
    #builder.add_measurement_times([100., 300.])
    feed_times = [100., 200.]
    builder.add_feed_times(feed_times)

    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ),'data_sets'))
    filename =  os.path.join(dataDirectory,'trimmed.csv')

    D_frame = read_spectral_data_from_csv(filename)
    meas_times = sorted(D_frame.index)#add feed times and meas times before adding data to model
    builder.add_spectral_data(D_frame)

    model = builder.create_pyomo_model(0, 700) #changed from 600 due to data
    #=========================================================================
    #USER INPUT SECTION - FE Factory
    #=========================================================================

    # call FESimulator
    # FESimulator re-constructs the current TemplateBuilder into fe_factory syntax
    # there is no need to call PyomoSimulator any more as FESimulator is a child class
    sim = FESimulator(model)

    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')

    #Since the model cannot discriminate inputs from other algebraic elements, we still
    #need to define the inputs as inputs_sub
    inputs_sub = {}
    inputs_sub['Y'] = ['5','Temp']
    sim.fix_from_trajectory('Y', 'Temp', fixed_Ttraj)


    fixedy = True  # instead of things above
    fixedtraj = True

    # #since these are inputs we need to fix this
    # for key in sim.model.time.value:
    #     sim.model.The.set_value(key)
    #     sim.model.Y[key, '5'].fix()

    #this will allow for the fe_factory to run the element by element march forward along
    #the elements and also automatically initialize the PyomoSimulator model, allowing
    #for the use of the run_sim() function as before. We only need to provide the inputs
    #to this function as an argument dictionary

    Z_step = {'AH': .3} #Which component and which amount is added
    X_step = {'V': 20.}
    jump_states = {'Z': Z_step, 'X': X_step}
    jump_points1 = {'AH': 100.0}#Which component is added at which point in time
    jump_points2 = {'V': 200.}
    jump_times = {'Z': jump_points1, 'X': jump_points2}

    init = sim.call_fe_factory(inputs_sub, jump_states, jump_times, feed_times, fixedy, fixedtraj)

    # init = sim.call_fe_factory(inputs_sub)
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
    
        # #results.Y[0].plot.line()
        # results.Y[1].plot.line(legend=True)
        # results.Y[2].plot.line(legend=True)
        # results.Y[3].plot.line(legend=True)
        # results.Y[4].plot.line(legend=True)
        # plt.xlabel("time (s)")
        # plt.ylabel("rxn rates (mol/L*s)")
        # plt.title("Rates of rxn")
        # plt.show()

        results.X.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Volume (L)")
        plt.title("total volume")
        plt.show()
    #D_frame.plot.line(legend=False)
    #plt.show()
    
    #plot_spectral_data(D_frame,dimension='3D')
    
    #results.Z.to_csv("initialization.csv")
