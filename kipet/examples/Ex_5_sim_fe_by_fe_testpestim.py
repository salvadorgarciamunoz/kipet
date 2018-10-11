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
from kipet.library.PyomoSimulatorChange import *
from kipet.library.ParameterEstimatorChange import *
#from kipet.library.VarianceEstimatorChange import *
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
    algebraics = ['0', '1', '2', '3', '4', '5']  # the indices of the rate rxns
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
        r.append(m.Y[t, '0'] - m.P['k0'] * m.Z[t, 'AH'] * m.Z[t, 'B'])
        r.append(m.Y[t, '1'] - m.P['k1'] * m.Z[t, 'A-'] * m.Z[t, 'C'])
        r.append(m.Y[t, '2'] - m.P['k2'] * m.Z[t, 'AC-'])
        r.append(m.Y[t, '3'] - m.P['k3'] * m.Z[t, 'AC-'] * m.Z[t, 'AH'])
        r.append(m.Y[t, '4'] - m.P['k4'] * m.Z[t, 'AC-'] * m.Z[t, 'BH+'])  # 6 is k4T
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

    #builder.add_measurement_times([100., 300.])
    # feed_times = [100., 200.]
    feed_times = [101.035, 303.126]
    builder.add_feed_times(feed_times)

    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ),'data_sets'))
    filename =  os.path.join(dataDirectory,'trimmedcoarser2.csv')

    D_frame = read_spectral_data_from_csv(filename)
    #meas_times = sorted(D_frame.index)#add feed times and meas times before adding data to model
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
    sim.apply_discretization('dae.collocation', nfe=30, ncp=3, scheme='LAGRANGE-RADAU')

    #Since the model cannot discriminate inputs from other algebraic elements, we still
    #need to define the inputs as inputs_sub
    inputs_sub = {}
    inputs_sub['Y'] = ['5']

    #since these are inputs we need to fix this
    for key in sim.model.time.value:
        sim.model.Y[key, '5'].set_value(key)
        sim.model.Y[key, '5'].fix()


    fixedy=True #instead of things above
    yfix={}
    yfix['Y']=['5']#needed in case of different input fixes

    #this will allow for the fe_factory to run the element by element march forward along 
    #the elements and also automatically initialize the PyomoSimulator model, allowing
    #for the use of the run_sim() function as before. We only need to provide the inputs 
    #to this function as an argument dictionary

    #
    # Z_step = {'AH': .3} #Which component and which amount is added
    # X_step = {'V': 20.}
    # jump_states = {'Z': Z_step, 'X': X_step}
    # jump_points1 = {'AH': 100.0}#Which component is added at which point in time
    # jump_points2 = {'V': 200.}
    # jump_times = {'Z': jump_points1, 'X': jump_points2}
    Z_step = {'AH': .3} #Which component and which amount is added
    X_step = {'V': 20.}
    jump_states = {'Z': Z_step, 'X': X_step}
    jump_points1 = {'AH': 101.035}#Which component is added at which point in time
    jump_points2 = {'V': 303.126}
    jump_times = {'Z': jump_points1, 'X': jump_points2}

    init = sim.call_fe_factory(inputs_sub, jump_states, jump_times, feed_times)#, fixedy, yfix)#yfix

    # init = sim.call_fe_factory(inputs_sub)
    # sys.exit()
    #=========================================================================
    #USER INPUT SECTION - SIMULATION
    #========================================================================
    #: now the final run, just as before
    # simulate
    options = {}
    # options = { 'acceptable_constr_viol_tol': 0.1, 'constr_viol_tol':0.1,# 'required_infeasibility_reduction':0.999999,
    #                'print_user_options': 'yes'}#,
    results = sim.run_sim('ipopt',
                          tee=True,
                          solver_opts=options)
    # #sys.exit()
    # model = builder.create_pyomo_model(0, 700)  # changed from 600 due to data
    # v_estimator = VarianceEstimator(model)
    # v_estimator.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')
    # v_estimator.initialize_from_trajectory('Z', results.Z)
    # v_estimator.initialize_from_trajectory('S', results.S)
    # v_estimator.initialize_from_trajectory('C', results.C)
    #
    # options = {}
    # options['mu_init'] = 1e-7
    # options['bound_push'] = 1e-8
    # options['linear_solver'] = 'ma57'
    # options['ma57_pivot_order'] = 4
    # # options['tol'] = 1e-9
    # # print(opt_model.pprint())
    # # opt_model.pprint(filename="fullfemodel.txt")
    # # f=open("fullfemodel.txt","w")
    # # f.write(opt_model.pprint())
    # # f.close()
    # A_set = [l for i, l in enumerate(model.meas_lambdas) if (i % 4 == 0)]
    # # inputs_sub['Y'] = ['Temp']
    # #
    # # dataDirectory = os.path.abspath(
    # #     os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    # #         inspect.currentframe()))), '..', 'Christina', 'data_sets'))
    # # Ttraj = os.path.join(dataDirectory, 'relevantTr-datafordataset2.csv')
    # # fixed_Ttraj = read_absorption_data_from_csv(Ttraj)
    # # trajs = dict()
    # # trajs[('Y', 'Temp')] = fixed_Ttraj
    # # trajs = dict()
    #
    # # jump_times =  # added for inclusion of discrete jumps CS
    # # jump_states =
    #
    # #: this will take a sweet-long time to solve.
    # results_variances = v_estimator.run_opt('ipopt',
    #                                         tee=True,
    #                                         solver_options=options,
    #                                         tolerance=1e-5,
    #                                         max_iter=15,
    #                                         subset_lambdas=A_set,
    #                                         inputs_sub=inputs_sub,
    #                                         # trajectories=trajs,
    #                                         jump=True,
    #                                         jump_times=jump_times,
    #                                         jump_states=jump_states,
    #                                         fixedy=True)#added
    #
    # print("\nThe estimated variances are:\n")
    #
    # for k, v in six.iteritems(results_variances.sigma_sq):
    #     print(k, v)
    # #
    # # The
    # # estimated
    # # variances
    # # are:
    # #
    # # ('C', 7.514833351202009e-07)
    # # ('B', 3.8330972320493624e-07)
    # # ('AH', 0.49560695729672)
    # # ('A-', 4.3306068150649534e-07)
    # # ('BH+', 0.5659793794156868)
    # # ('AC-', 4.432377193670943e-07)
    # # ('P', 4.596013737184771e-07)
    # # ('device', 4.93456158881624e-06)
    # sys.exit()
    # if with_plots:
    #     results_variances.C.plot.line(legend=True)
    #     plt.xlabel("time (h)")
    #     plt.ylabel("Concentration (mol/L)")
    #     plt.title("Concentration Profile")
    #
    #     results_variances.S.plot.line(legend=True)
    #     plt.xlabel("Wavelength (cm)")
    #     plt.ylabel("Absorbance (L/(mol cm))")
    #     plt.title("Absorbance  Profile")
    #
    #     plt.show()
    # sigmas = results_variances.sigma_sq

    # if with_plots:
    #     # display concentration results
    #     results.Z.plot.line(legend=True)
    #     plt.xlabel("time (s)")
    #     plt.ylabel("Concentration (mol/L)")
    #     plt.title("Concentration Profile")
    #     plt.show()
    #
    #     #results.Y[0].plot.line()
    #     results.Y[1].plot.line(legend=True)
    #     results.Y[2].plot.line(legend=True)
    #     results.Y[3].plot.line(legend=True)
    #     results.Y[4].plot.line(legend=True)
    #     plt.xlabel("time (s)")
    #     plt.ylabel("rxn rates (mol/L*s)")
    #     plt.title("Rates of rxn")
    #     plt.show()
    #
    #     results.X.plot.line(legend=True)
    #     plt.xlabel("time (s)")
    #     plt.ylabel("Volume (L)")
    #     plt.title("total volume")
    #     plt.show()
    # #D_frame.plot.line(legend=False)
    # #plt.show()
    #
    # #plot_spectral_data(D_frame,dimension='3D')
    #
    # #results.Z.to_csv("initialization.csv")
    sigmas={'C': 7.514833351202009e-07,
            'B': 3.8330972320493624e-07,
            'AH': 0.49560695729672,
            'A-': 4.3306068150649534e-07,
            'BH+': 0.5659793794156868,
            'AC-': 4.432377193670943e-07,
            'P': 4.596013737184771e-07,
            'device': 4.93456158881624e-06}
    # # ('C', 7.514833351202009e-07)
    # # ('B', 3.8330972320493624e-07)
    # # ('AH', 0.49560695729672)
    # # ('A-', 4.3306068150649534e-07)
    # # ('BH+', 0.5659793794156868)
    # # ('AC-', 4.432377193670943e-07)
    # # ('P', 4.596013737184771e-07)
    # # ('device', 4.93456158881624e-06)

    # and define our parameter estimation problem and discretization strategy
    model = builder.create_pyomo_model(0, 700)  # changed from 600 due to data
    model.del_component(params)
    # builder.add_parameter('k0', bounds=(0,100.))
    # builder.add_parameter('k1', bounds=(0,20.))
    # builder.add_parameter('k2', bounds=(0,4.))
    # builder.add_parameter('k3', bounds=(0,1.))
    # builder.add_parameter('k4', bounds=(0,8.))
    model.del_component(params)
    builder.add_parameter('k0', bounds=(0,100.))
    builder.add_parameter('k1', 8.93156)#bounds=(0,20.))
    builder.add_parameter('k2', 1.31765)#bounds=(0,4.))
    builder.add_parameter('k3', 0.310870)#bounds=(0,1.))
    builder.add_parameter('k4', 3.87809)#bounds=(0,8.))
    model =builder.create_pyomo_model(0,700)
    p_estimator = ParameterEstimator(model)
    p_estimator.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')

    # # Certain problems may require initializations and scaling and these can be provided from the
    # # varininace estimation step. This is optional.
    # p_estimator.initialize_from_trajectory('Z', results_variances.Z)
    # p_estimator.initialize_from_trajectory('S', results_variances.S)
    # p_estimator.initialize_from_trajectory('C', results_variances.C)
    p_guess = {'k0': 49.7796}#, 'k1': 8.93156, 'k2': 1.31765, 'k3': 0.310870, 'k4': 3.87809}  # , 'k5m':0.3, 'k5p':0.05, 'k6':0.3, 'k7':0.05}
    # builder.add_parameter('k1',1.3)
    # builder.add_parameter('k2',5.8)
    # builder.add_parameter('k3',1.2)
    # builder.add_parameter('k4',10) #change initial parameter values

    #raw_results = p_estimator.run_lsq_given_P('ipopt', p_guess, tee=False)

    p_estimator.initialize_from_trajectory('Z', results.Z)
    p_estimator.initialize_from_trajectory('S', results.S)
    p_estimator.initialize_from_trajectory('dZdt', results.dZdt)
    p_estimator.initialize_from_trajectory('C', results.C)
    # Again we provide options for the solver
    options = dict()

    options['mu_init'] = 1e-6
    options['bound_push'] = 1e-6
    # finally we run the optimization
    results_pyomo = p_estimator.run_opt('ipopt',
                                        tee=True,
                                        solver_opts=options,
                                        variances=sigmas,
                                        with_d_vars=True,
                                        #covariance=True,
                                        inputs_sub=inputs_sub,
                                        #trajectories=trajs,
                                        jump=True,
                                        jump_times=jump_times,
                                        jump_states=jump_states,
                                        fixedy=True,
                                        yfix=yfix
                                        )

    # And display the results
    print("The estimated parameters are:")
    for k, v in six.iteritems(results_pyomo.P):
        print(k, v)
    # display results
    if with_plots:
        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile with noise")

        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile without noise")

        results_pyomo.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")

        results_pyomo.X.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Volume (L)")
        plt.title("total volume")
        plt.show()

    # results_variances = v_estimator.run_opt('ipopt',
    #                                         tee=True,
    #                                         solver_options=options,
    #                                         tolerance=1e-5,
    #                                         max_iter=15,
    #                                         subset_lambdas=A_set,
    #                                         inputs_sub=inputs_sub,
    #                                         # trajectories=trajs,
    #                                         jump=True,
    #                                         jump_times=jump_times,
    #                                         jump_states=jump_states,
    #                                         fixedy=True)#added