#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with unknow variancesof spectral data using pyomo discretization 
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2*Z_b	                Z_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 

from __future__ import print_function
from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.opt.ParameterEstimator import *
from kipet.opt.VarianceEstimator import *
import matplotlib.pyplot as plt

from kipet.utils.data_tools import *
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
   
    # Load spectral data
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
            inspect.currentframe()))), 'data_sets'))
    filename = os.path.join(dataDirectory, 'Dij.txt')
    D_frame = read_spectral_data_from_txt(filename)

    # build dae block for optimization problems
    #################################################################################    
    builder = TemplateBuilder()
    components = {'A': 1e-3, 'B': 0, 'C': 0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1', bounds=(0.1, 4.0))
    builder.add_parameter('k2', bounds=(0.01, 1.0))
    builder.add_spectral_data(D_frame)


    # define explicit system of ODEs
    def rule_odes(m, t):
        exprs = dict()
        exprs['A'] = -m.P['k1'] * m.Z[t, 'A']
        exprs['B'] = m.P['k1'] * m.Z[t, 'A'] - m.P['k2'] * m.Z[t, 'B']
        exprs['C'] = m.P['k2'] * m.Z[t, 'B']
        return exprs

    builder.set_odes_rule(rule_odes)
    opt_model = builder.create_pyomo_model(0.0, 15.0)

    #################################################################################
    #: non absorbing species.
    non_abs = ['C']
    builder.set_non_absorbing_species(opt_model, non_abs)
    #################################################################################
    
    #=========================================================================
    #USER INPUT SECTION - VARIANCE ESTIMATOR
    #=========================================================================
   
    v_estimator = VarianceEstimator(opt_model)

    v_estimator.apply_discretization('dae.collocation', nfe=60, ncp=1, scheme='LAGRANGE-RADAU')

    options = {}
    #options['bound_push'] = 1e-8
    # options['tol'] = 1e-9
    A_set = [l for i, l in enumerate(opt_model.meas_lambdas) if (i % 7 == 0)]
    #: this will take a sweet-long time to solve.
    results_variances = v_estimator.run_opt('ipopt',
                                            tee=True,
                                            solver_options=options,
                                            tolerance=1e-4,
                                            max_iter=5,
                                            subset_lambdas=A_set)

    print("\nThe estimated variances are:\n")

    for k, v in six.iteritems(results_variances.sigma_sq):
        print(k, v)

    sigmas = results_variances.sigma_sq
    #################################################################################
    opt_model = builder.create_pyomo_model(0.0, 15.0)

    #=========================================================================
    #USER INPUT SECTION - PARAMETER ESTIMATOR
    #=========================================================================

    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation', nfe=60, ncp=1, scheme='LAGRANGE-RADAU')

    # Provide good initial guess obtained by variance estimation
    p_estimator.initialize_from_trajectory('Z', results_variances.Z)
    p_estimator.initialize_from_trajectory('S', results_variances.S)
    p_estimator.initialize_from_trajectory('C', results_variances.C)

    p_estimator.scale_variables_from_trajectory('Z', results_variances.Z)
    p_estimator.scale_variables_from_trajectory('S', results_variances.S)
    p_estimator.scale_variables_from_trajectory('C', results_variances.C)

    # dont push bounds i am giving you a good guess
    options = dict()
    #options['nlp_scaling_method'] = 'user-scaling'
    options['mu_init'] = 1e-6
    options['bound_push'] =1e-6
    results_pyomo = p_estimator.run_opt('ipopt_sens',
                                        tee=True,
                                        solver_opts=options,
                                        variances=sigmas,
                                        covariance=True)
    print("The estimated parameters are:")
    for k, v in six.iteritems(results_pyomo.P):
        print(k, v)

    if with_plots:

    # display results
        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_pyomo.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")

        plt.show()

