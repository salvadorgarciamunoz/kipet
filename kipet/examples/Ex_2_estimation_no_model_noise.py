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
#       Initial concentration 

from __future__ import print_function
from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.data_tools import *
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
       
    
    # Load spectral data from the relevant file location. As described in section 4.3.1
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename =  os.path.join(dataDirectory,'Dij.txt')
    D_frame = read_file(filename)

    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_spectral_data
    #################################################################################
    builder = TemplateBuilder()
    components = {'A': 1e-3, 'B': 0, 'C': 0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1', init=4.0, bounds=(0.0, 5.0))
    # There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder.add_parameter('k2', bounds=(0.0, 1.0))
    builder.add_spectral_data(D_frame)


    # define explicit system of ODEs
    def rule_odes(m, t):
        exprs = dict()
        exprs['A'] = -m.P['k1'] * m.Z[t, 'A']
        exprs['B'] = m.P['k1'] * m.Z[t, 'A'] - m.P['k2'] * m.Z[t, 'B']
        exprs['C'] = m.P['k2'] * m.Z[t, 'B']
        return exprs


    builder.set_odes_rule(rule_odes)
    builder.bound_profile(var='S', bounds=(0, 200))
    opt_model = builder.create_pyomo_model(0.0, 10.0)

    # =========================================================================
    # USER INPUT SECTION - VARIANCE ESTIMATION
    # =========================================================================

    # For this problem we have an input D matrix that has some noise in it
    # We can therefore use the variance estimator described in the Overview section
    # of the documentation and Section 4.3.3
    v_estimator = VarianceEstimator(opt_model)
    v_estimator.apply_discretization('dae.collocation', nfe=100, ncp=1, scheme='LAGRANGE-RADAU')

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
    A_set = [l for i, l in enumerate(opt_model.meas_lambdas) if (i % 4 == 0)]

    # Finally we run the variance estimatator using the arguments shown in Seciton 4.3.3
    worst_case_device_var = v_estimator.solve_max_device_variance('ipopt',
                                                                  tee=False,
                                                                  # subset_lambdas = A_set,
                                                                  solver_opts=options)

    # Variances can then be displayed
    print("\nThe estimated variance is:\n")
    print(worst_case_device_var)

    # =========================================================================
    # USER INPUT SECTION - PARAMETER ESTIMATION
    # =========================================================================
    # In order to run the paramter estimation we create a pyomo model as described in section 4.3.4
    # opt_model = builder.create_pyomo_model(0.0,10.0)

    # and define our parameter estimation problem and discretization strategy
    p_estimator = ParameterEstimator(opt_model)

    options = dict()
    # options['nlp_scaling_method'] = 'user-scaling'
    options['linear_solver'] = 'ma57'

    # Since, for this case we only need delta and not the model variance, we add the additional option
    # to exclude model variance and then run the optimization

    results_pyomo = p_estimator.run_opt('ipopt',
                                        tee=True,
                                        model_variance=False,
                                        solver_opts=options,
                                        variances=worst_case_device_var)

    # And display the results
    print("The estimated parameters are:")
    for k, v in six.iteritems(results_pyomo.P):
        print(k, v)

    # display results
    if with_plots:
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_pyomo.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")
        plt.show()
        
#         #%%
# def rule_objective(model):
#     """This function defines the objective function for the estimability
    
#     This is equation 5 from Chen and Biegler 2020. It has the following
#     form:
        
#     .. math::
#         \min J = \frac{1}{2}(\mathbf{w}_m - \mathbf{w})^T V_{\mathbf{w}}^{-1}(\mathbf{w}_m - \mathbf{w})
        
#     Originally KIPET was designed to only consider concentration data in
#     the estimability, but this version now includes complementary states
#     such as reactor and cooling temperatures. If complementary state data
#     is included in the model, it is detected and included in the objective
#     function.
    
#     Args:
#         model (pyomo.core.base.PyomoModel.ConcreteModel): This is the pyomo
#         model instance for the estimability problem.
            
#     Returns:
#         obj (pyomo.environ.Objective): This returns the objective function
#         for the estimability optimization.
    
#     """
#     obj = 0

#     for k in set(model.mixture_components.value_list) & set(model.measured_data.value_list):
#         for t, v in model.C.items():
#             obj += 0.5*(model.C[t] - model.Z[t]) ** 2 / model.sigma[k]**2
    
#     for k in set(model.complementary_states.value_list) & set(model.measured_data.value_list):
#         for t, v in model.U.items():
#             obj += 0.5*(model.X[t] - model.U[t]) ** 2 / model.sigma[k]**2      

#     return Objective(expr=obj)

    
#     #%%
    
# # Still does not converge - why?
# # Add the sigmas to the problem to get it working!
    
# import numpy as np
# import pandas as pd

# from kipet.library.NestedSchurDecomposition import NestedSchurDecomposition as NSD
# #from kipet.examples.Ex_17_CSTR_setup import make_model_dict

# # For simplicity, all of the models and simulated data are generated in
# # the following function

# # models, parameters = make_model_dict() 
# builder = TemplateBuilder()
# components = {'A': 1e-3, 'B': 0, 'C': 0}
# builder.add_mixture_component(components)
# builder.add_parameter('k1', init=p_estimator.model.P['k1'].value, bounds=(0, 5))
# # There is also the option of providing initial values: Just add init=... as additional argument as above.
# builder.add_parameter('k2', init=p_estimator.model.P['k2'].value, bounds=(0, 1))
# #builder.add_spectral_data(D_frame)

# builder.set_odes_rule(rule_odes)
# builder.set_model_times((0, 10))


# times = list()
# species = list()
# for k, v in p_estimator.model.Z.items():
#     if k[0] not in times:
#         times.append(k[0])
#     if k[1] not in species:
#         species.append(k[1])

# Z_data = pd.DataFrame(np.zeros((len(times), len(species))), index=times, columns=species)

# for r in times:
#     for c in species:
#         Z_data.loc[r, c] = pm.Z[r,c].value


# builder.add_concentration_data(pd.DataFrame(Z_data))
        
# #builder.bound_profile(var='S', bounds=(0, 200))
# nsd_model = builder.create_pyomo_model()


# models = {1: nsd_model}


# for k, model in models.items():
    
#      model.measured_data = Set(initialize=model.mixture_components)
#      model.sigma = Param(model.measured_data, initialize=1)
#      model.objective = rule_objective(model)

#      if not model.alltime.get_discretization_info():
        
#         model_pe = ParameterEstimator(model)
#         model_pe.apply_discretization('dae.collocation',
#                                         ncp = 3,
#                                         nfe = 50,
#                                         scheme = 'LAGRANGE-RADAU')

# # This is still needed and should include the union of all parameters in all
# # models

# #factor = np.random.uniform(low=0.8, high=1.2, size=len(parameters))

# p_init = {k: v.value for k, v in models[1].P.items()}

# d_init_guess = {k: (1*p_init[k], p.bounds) for k, p in nsd_model.P.items()}
# #d_init_guess = {p.name: (p.init, p.bounds) for i, p in enumerate(parameters)}


# # NSD routine

# # If there is only one parameter it does not really update - it helps if you
# # let the other infomation "seep in" from the other experiments.

# # This is very slow and takes a long time for spectra problems
# # I need to try this with multiple data sets.

# parameter_var_name = 'P'
# options = {
#     'method': 'trust-constr',
#     'use_est_param': False,   # Use this to reduce model based on EP
#     'use_scaling' : True,
#     'cross_updating' : True,
#     }

# print(d_init_guess)

# nsd = NSD(models, d_init_guess, parameter_var_name, options)
# results, od = nsd.nested_schur_decomposition(debug=True)


# print(f'\nThe final parameter values are:\n{nsd.parameters_opt}')
        
# nsd.plot_paths('')

# #%%

# builder = TemplateBuilder()
# components = {'A': 1e-3, 'B': 0, 'C': 0}
# builder.add_mixture_component(components)
# builder.add_parameter('k1', init=nsd.parameters_opt['k1'])
# # There is also the option of providing initial values: Just add init=... as additional argument as above.
# builder.add_parameter('k2', init=nsd.parameters_opt['k2'])
# #builder.add_spectral_data(D_frame)


# # define explicit system of ODEs
# def rule_odes(m, t):
#     exprs = dict()
#     exprs['A'] = -m.P['k1'] * m.Z[t, 'A']
#     exprs['B'] = m.P['k1'] * m.Z[t, 'A'] - m.P['k2'] * m.Z[t, 'B']
#     exprs['C'] = m.P['k2'] * m.Z[t, 'B']
#     return exprs


# builder.set_odes_rule(rule_odes)
# builder.set_model_times((0, 10))
# #builder.bound_profile(var='S', bounds=(0, 200))
# sim_model = builder.create_pyomo_model()

# simulator = PyomoSimulator(sim_model)

# for k, v in simulator.model.P.items():

#     simulator.model.P[k].fix()

# simulator.apply_discretization('dae.collocation',
#                                 ncp = 3,
#                                 nfe = 50,
#                                 scheme = 'LAGRANGE-RADAU')

# options = {'solver_opts' : dict(linear_solver='ma57')}

# results_pyomo_sim1 = simulator.run_sim('ipopt',
#                                   tee=False,
#                                   solver_options=options,
#                                   )

# results_pyomo_sim1.Z.plot.line(legend=True)
# plt.xlabel("time (s)")
# plt.ylabel("Concentration (mol/L)")
# plt.title("NSD Concentration Profile")

# builder = TemplateBuilder()
# components = {'A': 1e-3, 'B': 0, 'C': 0}
# builder.add_mixture_component(components)
# builder.add_parameter('k1', init=p_estimator.model.P['k1'].value)
# # There is also the option of providing initial values: Just add init=... as additional argument as above.
# builder.add_parameter('k2', init=p_estimator.model.P['k2'].value)
# #builder.add_spectral_data(D_frame)


# # define explicit system of ODEs
# def rule_odes(m, t):
#     exprs = dict()
#     exprs['A'] = -m.P['k1'] * m.Z[t, 'A']
#     exprs['B'] = m.P['k1'] * m.Z[t, 'A'] - m.P['k2'] * m.Z[t, 'B']
#     exprs['C'] = m.P['k2'] * m.Z[t, 'B']
#     return exprs


# builder.set_odes_rule(rule_odes)
# builder.set_model_times((0, 10))
# #builder.bound_profile(var='S', bounds=(0, 200))
# sim_model = builder.create_pyomo_model()

# simulator = PyomoSimulator(sim_model)

# for k, v in simulator.model.P.items():

#     simulator.model.P[k].fix()

# simulator.apply_discretization('dae.collocation',
#                                 ncp = 3,
#                                 nfe = 50,
#                                 scheme = 'LAGRANGE-RADAU')

# options = {'solver_opts' : dict(linear_solver='ma57')}

# results_pyomo_sim2 = simulator.run_sim('ipopt',
#                                   tee=False,
#                                   solver_options=options,
#                                   )

# results_pyomo_sim2.Z.plot.line(legend=True)
# plt.xlabel("time (s)")
# plt.title('p_estimator')
# plt.ylabel("Concentration (mol/L)")
# plt.title("Kipet Concentration Profile")
# plt.show()



