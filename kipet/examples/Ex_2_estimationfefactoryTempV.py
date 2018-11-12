
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with unknown variances of spectral data using fe-factory model with inputs
#
#		\frac{dZ_a}{dt} = -k_1*Z_a - (Vdot/V)*Z_a	                                                        Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2(T)*Z_b	- (Vdot/V)*Z_b	+ {1. m_Badd/V/1.1 for t<1.1s or 2. 0 for t>1.1s} Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2(T)*Z_b - (Vdot/V)*Z_c	                                                        Z_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 
#       Initial concentration 

from __future__ import print_function
from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.data_tools import *
from kipet.library.FESimulator import *
import matplotlib.pyplot as plt
import os
import sys
import inspect
import six

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
        
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================

    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_spectral_data
    #################################################################################    
    builder = TemplateBuilder()    
    components = {'A':1e-3,'B':0,'C':0}
    builder.add_mixture_component(components)
    
    algebraics=['1','2','3','k2T','Temp']
    builder.add_algebraic_variable(algebraics)
    
    # Load Temp data:
    dataDirectory = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe()))), 'data_sets'))
    Ttraj = os.path.join(dataDirectory, 'Tempvalues.csv')
    fixed_Ttraj = read_absorption_data_from_csv(Ttraj)
    
    params = dict()
    params['k1'] = 1.0
    params['k2Tr'] = 0.2265
    params['E'] = 2.

    builder.add_parameter(params)
    
    # add additional state variables
    extra_states = dict()
    extra_states['V'] = 0.0629418
    
    
    builder.add_complementary_state_variable(extra_states)

    # stoichiometric coefficients
    gammas = dict()
    gammas['A'] = [-1, 0]
    gammas['B'] = [1, -1]
    gammas['C'] = [0, 1]
    
    
    # define system of DAEs

    def rule_algebraics(m, t):
        r = list()
        r.append(m.Y[t, '1'] - m.P['k1'] * m.Z[t, 'A'])
        r.append(m.Y[t, '2'] - m.Y[t,'k2T'] * m.Z[t, 'B'])
        Tr = 303.15
        # add temperature dependence via Arrhenius law
        r.append(m.Y[t, 'k2T'] - m.P['k2Tr'] * exp(m.P['E'](1 / m.Y[t, 'Temp'] - 1 / Tr)))
        return r
    
    builder.set_algebraics_rule(rule_algebraics)

    def rule_odes(m, t):
        exprs = dict()
        eta = 1e-2

        step = 0.5 * ((m.Y[t, '3'] + 1) / ((m.Y[t, '3'] + 1) ** 2 + eta ** 2) ** 0.5 + (1.1 - m.Y[t, '3']) / (
                    (1.1 - m.Y[t, '3']) ** 2 + eta ** 2) ** 0.5)
        exprs['V'] = 7.27609e-05 * step
        V = m.X[t, 'V']
        # mass balances
        for c in m.mixture_components:
            exprs[c] = gammas[c][0] * m.Y[t, '1'] + gammas[c][1] * m.Y[t, '2']  - exprs['V'] / V * m.Z[t, c]
            if c == 'B':
                exprs[c] += 0.002247311828 / (m.X[t, 'V'] * 1.1) * step
        return exprs
    
    builder.set_odes_rule(rule_odes)
    
    #Add time points where feed as discrete jump should take place: before adding data to the model!!!
    feed_times=[3.6341]
    builder.add_feed_times(feed_times)
    
    
    
    # Load spectral data from the relevant file location. As described in section 4.3.1
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename =  os.path.join(dataDirectory,'Dij.txt')
    D_frame = read_spectral_data_from_txt(filename)
    
    builder.add_spectral_data(D_frame)
    
    
    opt_model = builder.create_pyomo_model(0.0,10.0)
    
    #=========================================================================
    #USER INPUT SECTION - FE Factory
    #=========================================================================

    # call FESimulator
    # FESimulator re-constructs the current TemplateBuilder into fe_factory syntax
    # there is no need to call PyomoSimulator any more as FESimulator is a child class
    sim = FESimulator(opt_model)

    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')

    #Since the model cannot discriminate inputs from other algebraic elements, we still
    #need to define the inputs as inputs_sub
    inputs_sub = {}
    inputs_sub['Y'] = ['3','Temp']
    sim.fix_from_trajectory('Y', 'Temp', fixed_Ttraj)


    trajs = dict()
    trajs[('Y', 'Temp')] = fixed_Ttraj

    fixedy = True  # instead of things above
    fixedtraj = True
    yfix={}
    yfix['Y']=['3']#needed in case of different input fixes
    #print('yfix:',yfix.keys())
    yfixtraj={}
    yfixtraj['Y']=['Temp']
    
    ## #since these are inputs we need to fix this
    for key in sim.model.time.value:
        sim.model.Y[key, '3'].set_value(key)
        sim.model.Y[key, '3'].fix()
    Z_step = {'A': .01} #Which component and which amount is added
    X_step = {'V': 20.}
    jump_states = {'Z': Z_step, 'X': X_step}
    jump_points1 = {'A': 3.6341}#Which component is added at which point in time
    jump_points2 = {'V': 3.6341}
    jump_times = {'Z': jump_points1, 'X': jump_points2}
    
    init = sim.call_fe_factory(inputs_sub=inputs_sub, jump_states=jump_states, jump_times=jump_times, feed_times=feed_times)
    # They should be added in this way in case some arguments of all the ones available are not necessary here.
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
    if with_plots:
        
        results.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile Without Noise")
        plt.show()
        #results.Y[0].plot.line()
        results.Y['1'].plot.line(legend=True)
        results.Y['2'].plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("rxn rates (mol/L*s)")
        plt.title("Rates of rxn")
        plt.show()
        
        results.X.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Volume (L)")
        plt.title("total volume")
        plt.show()
#=========================================================================
    #USER INPUT SECTION - Variance Estimation
#========================================================================
    model=builder.create_pyomo_model(0.0,10.0)
    #Now introduce parameters as non fixed
    model.del_component(params)
    builder.add_parameter('k1',bounds=(0.0,5.0))
    builder.add_parameter('k2Tr',0.2265)
    builder.add_parameter('E',2.)
    model = builder.create_pyomo_model(0, 10)
    v_estimator = VarianceEstimator(model)
    v_estimator.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')
    v_estimator.initialize_from_trajectory('Z', results.Z)
    v_estimator.initialize_from_trajectory('S', results.S)
    
    options = {}

    A_set = [l for i, l in enumerate(model.meas_lambdas) if (i % 4 == 0)]

    #
    # #: this will take a sweet-long time to solve.
    results_variances = v_estimator.run_opt('ipopt',
                                            tee=True,
                                            solver_options=options,
                                            tolerance=1e-5,
                                            max_iter=15,
                                            subset_lambdas=A_set,
                                            inputs_sub=inputs_sub,
                                            trajectories=trajs,
                                            jump=True,
                                            jump_times=jump_times,
                                            jump_states=jump_states,
                                            fixedy=True,
                                            fixedtraj=True,
                                            yfix=yfix,
                                            yfixtraj=yfixtraj,
                                            feed_times=feed_times
                                            )
    
    print("\nThe estimated variances are:\n")
    
    for k, v in six.iteritems(results_variances.sigma_sq):
        print(k, v)

    sigmas = results_variances.sigma_sq

    if with_plots:
        # display concentration results
        results.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile with Noise")

        results.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile Without Noise")
        plt.show()
        #results.Y[0].plot.line()
        results.Y['1'].plot.line(legend=True)
        results.Y['2'].plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("rxn rates (mol/L*s)")
        plt.title("Rates of rxn")
        plt.show()
        
        results.X.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Volume (L)")
        plt.title("total volume")
        plt.show()

#=========================================================================
    #USER INPUT SECTION - Parameter Estimation
#========================================================================
    #In case the variances are known, they can also be fixed instead of running the Variance Estimation.
    #sigmas={'C': 1e-10,
            #'B': 1e-10,
            #'A': 1e-10,
            #'device': 1e-10}
    model =builder.create_pyomo_model(0.0,10)
    #Now introduce parameters as non fixed
    model.del_component(params)
    builder.add_parameter('k1',bounds=(0.0,5.0))
    builder.add_parameter('k2Tr',0.2265)
    builder.add_parameter('E',2.)
    
    model =builder.create_pyomo_model(0.0,10)
    p_estimator = ParameterEstimator(model)
    p_estimator.apply_discretization('dae.collocation', nfe=50, ncp=3, scheme='LAGRANGE-RADAU')

    # # Certain problems may require initializations and scaling and these can be provided from the
    # # variance estimation step. This is optional.
    # p_estimator.initialize_from_trajectory('Z', results_variances.Z)
    # p_estimator.initialize_from_trajectory('S', results_variances.S)
    # p_estimator.initialize_from_trajectory('C', results_variances.C)

    p_estimator.initialize_from_trajectory('Z', results.Z)
    p_estimator.initialize_from_trajectory('S', results.S)
    p_estimator.initialize_from_trajectory('dZdt', results.dZdt)
    p_estimator.initialize_from_trajectory('C', results.C)


    # # Again we provide options for the solver
    options = dict()

    # finally we run the optimization
    results_pyomo = p_estimator.run_opt('ipopt_sens',
                                        tee=True,
                                        solver_opts=options,
                                        variances=sigmas,
                                        with_d_vars=True,
                                        covariance=True,
                                        inputs_sub=inputs_sub,
                                        trajectories=trajs,
                                        jump=True,
                                        jump_times=jump_times,
                                        jump_states=jump_states,
                                        fixedy=True,
                                        fixedtraj=True,
                                        yfix=yfix,
                                        yfixtraj=yfixtraj,
                                        feed_times=feed_times
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
        plt.title("Concentration Profile with Noise")

        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile Without Noise")
        plt.show()
        
        results_pyomo.Y['1'].plot.line(legend=True)
        results.Y['2'].plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("rxn rates (mol/L*s)")
        plt.title("Rates of rxn")
        plt.show()

        results_pyomo.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")
    
        results_pyomo.X.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Volume (L)")
        plt.title("total volume")
        plt.show()
