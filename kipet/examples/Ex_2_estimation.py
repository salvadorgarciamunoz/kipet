
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
#%%
# if __name__ == "__main__":

#     with_plots = True
#     if len(sys.argv)==2:
#         if int(sys.argv[1]):
#             with_plots = False
 
        
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=====================================================`====================
       
    
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
    components = {'A':1e-3,'B':0,'C':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1', init=4.0, bounds=(0.0,5.0)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder.add_parameter('k2', init=2, bounds=(0.0,1.0))
    builder.add_spectral_data(D_frame)


    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    builder.set_odes_rule(rule_odes)
    builder.bound_profile(var='S', bounds=(0, 200))
    opt_model = builder.create_pyomo_model(0.0,10.0)
    
    #%
    
    #=========================================================================
    #USER INPUT SECTION - VARIANCE ESTIMATION 
    #=========================================================================
    # For this problem we have an input D matrix that has some noise in it
    # We can therefore use the variance estimator described in the Overview section
    # of the documentation and Section 4.3.3
    v_estimator = VarianceEstimator(opt_model)
    v_estimator.apply_discretization('dae.collocation',nfe=60,ncp=1,scheme='LAGRANGE-RADAU')
    
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
    A_set = [l for i,l in enumerate(opt_model.meas_lambdas) if (i % 4 == 0)]
    
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
    
    
    #=========================================================================
    # USER INPUT SECTION - PARAMETER ESTIMATION 
    #=========================================================================
    # In order to run the paramter estimation we create a pyomo model as described in section 4.3.4
    opt_model = builder.create_pyomo_model(0.0,10.0)

    # and define our parameter estimation problem and discretization strategy
    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=60,ncp=1,scheme='LAGRANGE-RADAU')
    
    # Certain problems may require initializations and scaling and these can be provided from the 
    # varininace estimation step. This is optional.
    p_estimator.initialize_from_trajectory('Z',results_variances.Z)
    p_estimator.initialize_from_trajectory('S',results_variances.S)
    p_estimator.initialize_from_trajectory('C',results_variances.C)

    # Scaling for Ipopt can also be provided from the variance estimator's solution
    # these details are elaborated on in the manual
    p_estimator.scale_variables_from_trajectory('Z',results_variances.Z)
    p_estimator.scale_variables_from_trajectory('S',results_variances.S)
    p_estimator.scale_variables_from_trajectory('C',results_variances.C)
    
    # Again we provide options for the solver, this time providing the scaling that we set above
    options = dict()
    options['nlp_scaling_method'] = 'user-scaling'

    # finally we run the optimization
    results_pyomo = p_estimator.run_opt('ipopt',
                                      tee=True,
                                      solver_opts = options,
                                      variances=sigmas)

    # The p_estimator models are needed for the NSD algorithm after C has
    # been found...
    # Try this using a single instance...
    # Initial values are the parameters found using the PE

    # And display the results
    print("The estimated parameters are:")
    for k,v in six.iteritems(results_pyomo.P):
        print(k, v)
        
    # display results
  #  if with_plots:
    results_pyomo.C.plot.line(legend=True)
    plt.xlabel("time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile")

    results_pyomo.S.plot.line(legend=True)
    plt.xlabel("Wavelength (cm)")
    plt.ylabel("Absorbance (L/(mol cm))")
    plt.title("Absorbance  Profile")

    plt.show()
        
    #%%

    # The results from original file:   
        
    # k1 0.22652582570495683
    # k2 1.0
       
    # These are different from the tutorial, for some reason!
        
        
    #%%
def rule_objective(model):
    """This function defines the objective function for the estimability
    
    This is equation 5 from Chen and Biegler 2020. It has the following
    form:
        
    .. math::
        \min J = \frac{1}{2}(\mathbf{w}_m - \mathbf{w})^T V_{\mathbf{w}}^{-1}(\mathbf{w}_m - \mathbf{w})
        
    Originally KIPET was designed to only consider concentration data in
    the estimability, but this version now includes complementary states
    such as reactor and cooling temperatures. If complementary state data
    is included in the model, it is detected and included in the objective
    function.
    
    Args:
        model (pyomo.core.base.PyomoModel.ConcreteModel): This is the pyomo
        model instance for the estimability problem.
            
    Returns:
        obj (pyomo.environ.Objective): This returns the objective function
        for the estimability optimization.
    
    """
    obj = 0

    for k in set(model.mixture_components.value_list) & set(model.measured_data.value_list):
        for t, v in model.C.items():
            obj += 0.5*(model.C[t] - model.Z[t]) ** 2 / model.sigma[k]**2
    
    for k in set(model.complementary_states.value_list) & set(model.measured_data.value_list):
        for t, v in model.U.items():
            obj += 0.5*(model.X[t] - model.U[t]) ** 2 / model.sigma[k]**2      

    return Objective(expr=obj)

    
    #%%
    
# Still does not converge - why?
# Add the sigmas to the problem to get it working!
    
import numpy as np
import pandas as pd

from kipet.library.NestedSchurDecomposition import NestedSchurDecomposition as NSD
#from kipet.examples.Ex_17_CSTR_setup import make_model_dict

# For simplicity, all of the models and simulated data are generated in
# the following function

# models, parameters = make_model_dict() 
models = {1: p_estimator.model}


for k, model in models.items():
    
     model.measured_data = Set(initialize=model.mixture_components)
     model.sigma = Param(model.measured_data, initialize=1)
     model.objective = rule_objective(model)

# This is still needed and should include the union of all parameters in all
# models

#factor = np.random.uniform(low=0.8, high=1.2, size=len(parameters))

p_init = {k: v.value for k, v in models[1].P.items()}

d_init_guess = {k: (1.2*p_init[k], p) for k, p in builder._parameters_bounds.items()}
#d_init_guess = {p.name: (p.init, p.bounds) for i, p in enumerate(parameters)}


# NSD routine

# If there is only one parameter it does not really update - it helps if you
# let the other infomation "seep in" from the other experiments.

# This is very slow and takes a long time for spectra problems
# I need to try this with multiple data sets.

parameter_var_name = 'P'
options = {
    'method': 'trust-constr',
    'use_est_param': False,   # Use this to reduce model based on EP
    'use_scaling' : True,
    'cross_updating' : True,
    }

print(d_init_guess)

nsd = NSD(models, d_init_guess, parameter_var_name, options)
results, od = nsd.nested_schur_decomposition(debug=True)


print(f'\nThe final parameter values are:\n{nsd.parameters_opt}')
        
nsd.plot_paths('')
