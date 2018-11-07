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
from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.data_tools import *
from kipet.library.fe_factory import *
from kipet.library.FESimulator import *
from kipet.library.EstimabilityAnalyzer import *
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
    algebraics = [0, 1, 2, 3, 4, 5]  # the indices of the rate rxns
    # note the fifth component. Which basically works as an input

    builder.add_algebraic_variable(algebraics)

    params = dict()
    params['k0'] = 49.7796
    params['k1'] = 8.93156
    params['k2'] = 1.31765
    params['k3'] = 0.310870
    params['k4'] = 3.87809

    #builder.add_parameter(params)
    builder.add_parameter('k0',bounds=(20.0,80.0))
    builder.add_parameter('k1',bounds=(5.0,15.0))
    builder.add_parameter('k2',bounds=(0.00,5.0))
    builder.add_parameter('k3',bounds=(0.00,1.0))
    builder.add_parameter('k4',bounds=(0.00,7.0))
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
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename =  os.path.join(dataDirectory,'Michaels_C_data_input_noise.csv')

    D_frame = read_concentration_data_from_csv(filename)
    meas_times = sorted(D_frame.index)
    builder.add_measurement_times(meas_times)
    builder.add_concentration_data(D_frame)    
    model = builder.create_pyomo_model(0, 600)

    #=========================================================================
    #USER INPUT SECTION - FE Factory
    #=========================================================================
    '''
    # call FESimulator
    # FESimulator re-constructs the current TemplateBuilder into fe_factory syntax
    # there is no need to call PyomoSimulator any more as FESimulator is a child class 
    sim = FESimulator(model)
    
    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')

    #Since the model cannot discriminate inputs from other algebraic elements, we still
    #need to define the inputs as inputs_sub
    inputs_sub = {}
    inputs_sub['Y'] = [5]

    #since these are inputs we need to fix this
    for key in sim.model.time.value:
        sim.model.Y[key, 5].set_value(key)
        sim.model.Y[key, 5].fix()

    #this will allow for the fe_factory to run the element by element march forward along 
    #the elements and also automatically initialize the PyomoSimulator model, allowing
    #for the use of the run_sim() function as before. We only need to provide the inputs 
    #to this function as an argument dictionary
    init = sim.call_fe_factory(inputs_sub)
    
    #=========================================================================
    #USER INPUT SECTION - SIMULATION
    #=========================================================================
     
    #: now the final run, just as before
    # simulate
    options = {}
    results = sim.run_sim('ipopt',
                          tee=True,
                          solver_opts=options)
    
    #=========================================================================
    #USER INPUT SECTION - VARIANCE ESTIMATION 
    #=========================================================================
    # For this problem we have an input D matrix that has some noise in it
    # We can therefore use the variance estimator described in the Overview section
    # of the documentation and Section 4.3.3
    v_estimator = VarianceEstimator(model)
    v_estimator.apply_discretization('dae.collocation',nfe=50,ncp=3,scheme='LAGRANGE-RADAU')
    
    # It is often requried for larger problems to give the solver some direct instructions
    # These must be given in the form of a dictionary
    options = {}
    # While this problem should solve without changing the deault options, example code is 
    # given commented out below. See Section 5.6 for more options and advice.
    # options['bound_push'] = 1e-8
    # options['tol'] = 1e-9
    
    # The set A_set is then decided. This set, explained in Section 4.3.3 is used to make the
    # variance estimation run faster and has been shown to not decrease the accuracy of the variance 
    # prediction for large noisey data sets.
    A_set = [l for i,l in enumerate(model.meas_lambdas) if (i % 7 == 0)]
    
    # Finally we run the variance estimatator using the arguments shown in Seciton 4.3.3
    results_variances = v_estimator.run_opt('ipopt',
                                            tee=True,
                                            solver_options=options,
                                            tolerance=1e-5,
                                            max_iter=15,
                                            subset_lambdas=A_set)

    # Variances can then be displayed 
    print("\nThe estimated variances are:\n")
    for k,v in six.iteritems(results_variances.sigma_sq):
        print(k, v)

    # and the sigmas for the parameter estimation step are now known and fixed
    sigmas = results_variances.sigma_sq
    
    '''
    #=========================================================================
    # USER INPUT SECTION - PARAMETER ESTIMATION 
    #=========================================================================
    # In order to run the paramter estimation we create a pyomo model as described in section 4.3.4

    # and define our parameter estimation problem and discretization strategy
    p_estimator = ParameterEstimator(model)
    p_estimator.apply_discretization('dae.collocation',nfe=50,ncp=3,scheme='LAGRANGE-RADAU')
    
    #p_estimator.initialize_from_trajectory('Z',results_variances.Z)
    #p_estimator.initialize_from_trajectory('S',results_variances.S)
    #p_estimator.initialize_from_trajectory('C',results_variances.C)     
    
    # Again we provide options for the solver, this time providing the scaling that we set above
    options = dict()
    #options['nlp_scaling_method'] = 'gradient-based'
    #options['bound_relax_factor'] = 0
    #options['nlp_scaling_method'] = 'user-scaling'
    #options['mu_strategy'] = 'adaptive'
    options['mu_init'] = 1e-6
    #options['bound_push'] =1e-6
    options['linear_solver']='ma27'
    # finally we run the optimization

    sigmas = {'AH':1e-10,'B':1e-11,'C':1e-10, 'BH+':1e-10,'A-':1e-10,'AC-':1e-10,'P':1e-10,'device':1e-10}
    results_pyomo = p_estimator.run_opt('k_aug',
                                        variances=sigmas,
                                      tee=True,
                                      solver_opts = options,
                                      covariance=True)
    if with_plots:
        
        results.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        plt.show()
        # display concentration results    
        results.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        plt.show()
        
        
        #results.Y[0].plot.line()
        results.Y[1].plot.line(legend=True)
        results.Y[2].plot.line(legend=True)
        results.Y[3].plot.line(legend=True)
        results.Y[4].plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("rxn rates (mol/L*s)")
        plt.title("Rates of rxn")
        plt.show()

        results.X.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Volume (L)")
        plt.title("total volume")
        plt.show()
    #D_frame.plot.line(legend=False)
    #plt.show()
    
    #plot_spectral_data(D_frame,dimension='3D')
    
    #results.Z.to_csv("initialization.csv")
