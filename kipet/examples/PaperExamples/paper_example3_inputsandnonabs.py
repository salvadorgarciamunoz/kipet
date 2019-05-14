#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem
# Paper Example 3
# Inputs, extra_states and non-absorbing species
# Slight modification of the Michael's reaction
# Estimation with known variances of spectral data using pyomo discretization
#
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j


from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.data_tools import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import inspect
from kipet.library.FESimulator import *
from pyomo.core import *
from pyomo.opt import *
import pickle
import os
from itertools import count, takewhile


if __name__ == "__main__":

    with_plots = True
    if len(sys.argv) == 2:
        if int(sys.argv[1]):
            with_plots = False
    # =========================================================================
    # USER INPUT SECTION - MODEL BUILDING - See user guide for how this section functions
    # =========================================================================

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
    algebraics = ['0', '1', '2', '3', '4', '5']#, 'Temp']  # the indices of the rate rxns
    # note the fifth, sixth and seventh components. Which basically work as inputs

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
        r.append(m.Y[t, '4'] - m.P['k4'] * m.Z[t, 'AC-'] * m.Z[t, 'BH+'])

        return r
    #: there is no ae for Y[t,5] because step equn under rule_odes functions as the switch for the "C" equation

    builder.set_algebraics_rule(rule_algebraics)


    def rule_odes(m, t):
        exprs = dict()
        eta = 1e-2
        step = 0.5 * ((m.Y[t, '5'] + 1) / ((m.Y[t, '5'] + 1) ** 2 + eta ** 2) ** 0.5 + (210.0 - m.Y[t,'5']) / ((210.0 - m.Y[t, '5']) ** 2 + eta ** 2) ** 0.5)
        exprs['V'] = 7.27609e-05 * step
        V = m.X[t, 'V']
        # mass balances
        for c in m.mixture_components:
            exprs[c] = gammas[c][0] * m.Y[t, '0'] + gammas[c][1] * m.Y[t, '1']+gammas[c][2] * m.Y[t, '2']+gammas[c][3] * m.Y[t, '3']+ gammas[c][4] * m.Y[t, '4'] - exprs['V'] / V * m.Z[t, c]
            if c == 'C':
                exprs[c] += 0.02247311828 / (m.X[t, 'V'] * 210) * step
        return exprs


    builder.set_odes_rule(rule_odes)
    
    #Add time points where feed as discrete jump should take place:
    feed_times=[101.035, 303.126]#, 400.
    builder.add_feed_times(feed_times)

    model = builder.create_pyomo_model(0, 600)

    non_abs = ['AC-']
    builder.set_non_absorbing_species(model, non_abs)
    
    #Load data:
    dataDirectory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data_sets'))
    filenameD =  os.path.join(dataDirectory,'FeCaseexamplewithoutTemp_D_data_input_noiselesspoints.csv')
    D_frame = read_spectral_data_from_csv(filenameD)
    builder.add_spectral_data(D_frame)
    model = builder.create_pyomo_model(0., 600.) 
    
    #Absorbing Simulation for Initialization:
    #=========================================================================
    #USER INPUT SECTION - FE Factory
    #=========================================================================
     
    #call FESimulator
    #FESimulator re-constructs the current TemplateBuilder into fe_factory syntax
    #there is no need to call PyomoSimulator any more as FESimulator is a child class 
    sim = FESimulator(model)
    
    ## defines the discrete points wanted in the concentration profile
    sim.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')

    #Since the model cannot discriminate inputs from other algebraic elements, we still
    #need to define the inputs as inputs_sub
    inputs_sub = {}
    inputs_sub['Y'] = ['5']
   
    fixedy = True
    yfix={}
    yfix['Y']=['5']#needed in case of different input fixes
    #since these are inputs we need to fix this
    for key in sim.model.time.value:
        sim.model.Y[key, '5'].set_value(key)
        sim.model.Y[key, '5'].fix()

    #this will allow for the fe_factory to run the element by element march forward along 
    #the elements and also automatically initialize the PyomoSimulator model, allowing
    #for the use of the run_sim() function as before. We only need to provide the inputs 
    #to this function as an argument dictionary
    
    #New Inputs for discrete feeds
    Z_step = {'AH': .03}#Which component and which amount is added
    X_step = {'V': .01}
    jump_states = {'Z': Z_step, 'X': X_step}
    jump_points1 = {'AH': 101.035}#Which component is added at which point in time
    jump_points2 = {'V': 303.126}
    jump_times = {'Z': jump_points1, 'X': jump_points2}

    init = sim.call_fe_factory(inputs_sub, jump_states, jump_times, feed_times)
    options = {}

    results_sim = sim.run_sim('ipopt',
                          tee=True)
                          
    if with_plots:
        
        results_sim.C.plot.line(legend=True)
        plt.xlabel("time (h)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        plt.show()

    # =========================================================================
    # USER INPUT SECTION - Parameter Estimation
    # # ========================================================================
    sigmas={'AH': 1e-10,
            'B':1e-10,
            'C': 1e-10,
            'BH+': 1e-10,
            'A-':1e-10,
            'P':1e-10,
            'device':1e-10}

    model = builder.create_pyomo_model(0., 600.)#0.51667

    model.del_component(params)
    builder.add_parameter('k0', init=0.9*49.7796,bounds=(0.0, 100.0))
    builder.add_parameter('k1', 8.93156)
    builder.add_parameter('k2', init=0.9*1.31765,bounds=(0.0,100.))
    builder.add_parameter('k3', init=0.9*0.310870, bounds=(0.,100.))
    builder.add_parameter('k4', init=0.9*3.87809,bounds=(0.0, 100.))
    model = builder.create_pyomo_model(0., 600.) 

    # and define our parameter estimation problem and discretization strategy
    p_estimator = ParameterEstimator(model)
    p_estimator.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')

    # Certain problems may require initializations and scaling and these can be provided from the
    # variance estimation step. This is optional.
    ##############
    p_estimator.initialize_from_trajectory('Z', results_sim.Z)
    p_estimator.initialize_from_trajectory('S', results_sim.S)
    p_estimator.initialize_from_trajectory('C', results_sim.C)
    p_estimator.initialize_from_trajectory('dZdt', results_sim.dZdt)
    ################
    
    # Again we provide options for the solver
    options = dict()
    options['mu_init'] = 1e-7
    options['bound_push'] = 1e-8
    
    # finally we run the optimization
    results_pyomo = p_estimator.run_opt('ipopt_sens',
                                        tee=True,
                                        solver_opts=options,
                                        variances=sigmas,
                                        with_d_vars=True,
                                        covariance=True,
                                        inputs_sub=inputs_sub,
                                        jump=True,
                                        jump_times=jump_times,
                                        jump_states=jump_states,
                                        feed_times=feed_times,# has to be added here for error checks in ParameterEstimator
                                        fixedy=True,
                                        report_time = True,
                                        yfix=yfix)
    lof = p_estimator.lack_of_fit()
    # And display the results
    print("The estimated parameters are:")
    for k, v in six.iteritems(results_pyomo.P):
        print(k, v)
        
    # display results
    if with_plots:
        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (h)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")


        results_pyomo.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")

        plt.figure(3)
        transpD_frame = np.transpose(D_frame)
        plt.plot(transpD_frame)  # line(legend=True)
        # plt.plot.line(legend=True)
        plt.legend
        plt.xlabel("Wavelength (cm)")#Wavenumber (1/cm)")
        plt.ylabel("Absorbance (L/(mol cm))")# ((L cm)/mol)")  #
        plt.title("D")

        plt.show()

   