#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with unknown variances of spectral data using pyomo discretization 
#
#		\frac{dZ_a}{dt} = -k_1*Z_a*Z_b	                                Z_a(0) = 1
#		\frac{dZ_b}{dt} = -k_1*Z_a*Z_b                   		Z_b(0) = 0.8
#               \frac{dZ_c}{dt} = k_1*Z_a*Z_b-2*k_2*Z_c^2	                Z_c(0) = 0
#               \frac{dZ_d}{dt} = k_2*Z_c^2             	                Z_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 



from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.opt.ParameterEstimator import *
from kipet.opt.VarianceEstimator import *
import matplotlib.pyplot as plt

from kipet.utils.data_tools import *
import inspect
import sys
import os
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
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename =  os.path.join(dataDirectory,'Dij_case51c.txt')
    D_frame = read_spectral_data_from_txt(filename)

    ######################################
    builder = TemplateBuilder()    
    components = {'A':1,'B':0.8,'C':0,'D':0}
    builder.add_mixture_component(components)

    # note the parameter is not fixed
    builder.add_parameter('k1',bounds=(0.0,5.0))
    builder.add_parameter('k2',bounds=(0.0,5.0))
    builder.add_spectral_data(D_frame)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']*m.Z[t,'B']
        exprs['B'] = -m.P['k1']*m.Z[t,'A']*m.Z[t,'B']
        exprs['C'] = m.P['k1']*m.Z[t,'A']*m.Z[t,'B']-2*m.P['k2']*m.Z[t,'C']**2
        exprs['D'] = m.P['k2']*m.Z[t,'C']**2
        return exprs

    builder.set_odes_rule(rule_odes)

    opt_model = builder.create_pyomo_model(0.0,12.0)

    #=========================================================================
    # USER INPUT SECTION - VARIANCE ESTIMATION
    #=========================================================================
   
    v_estimator = VarianceEstimator(opt_model)

    v_estimator.apply_discretization('dae.collocation',nfe=60,ncp=1,scheme='LAGRANGE-RADAU')

    # Provide good initial guess
    
    p_guess = {'k1':4.0,'k2':2.0}
    raw_results = v_estimator.run_lsq_given_P('ipopt',p_guess,tee=False)

    v_estimator.initialize_from_trajectory('Z',raw_results.Z)
    v_estimator.initialize_from_trajectory('S',raw_results.S)
    v_estimator.initialize_from_trajectory('dZdt',raw_results.dZdt)
    v_estimator.initialize_from_trajectory('C',raw_results.C)
    
    # dont push bounds i am giving you a good guess
    options = dict()

    A_set = [l for i,l in enumerate(opt_model.meas_lambdas) if i%4]
    results_variances = v_estimator.run_opt('ipopt',
                                            tee=True,
                                            solver_options=options,
                                            tolerance=1e-5,
                                            max_iter=100,
                                            subset_lambdas=A_set)

    print("\nThe estimated variances are:\n")
    for k,v in six.iteritems(results_variances.sigma_sq):
        print(k, v)

    print("The estimated parameters are:")
    for k,v in six.iteritems(opt_model.P):
        print(k, v.value)
        
    sigmas = results_variances.sigma_sq
    
    #=========================================================================
    # USER INPUT SECTION - PARAMETER ESTIMATION 
    #=========================================================================
    #################################################################################
    opt_model = builder.create_pyomo_model(0.0,12.0)

    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=60,ncp=1,scheme='LAGRANGE-RADAU')
    
    # Provide good initial guess obtained by variance estimation
    p_estimator.initialize_from_trajectory('Z',results_variances.Z)
    p_estimator.initialize_from_trajectory('S',results_variances.S)
    p_estimator.initialize_from_trajectory('C',results_variances.C)

    p_estimator.scale_variables_from_trajectory('Z',results_variances.Z)
    p_estimator.scale_variables_from_trajectory('S',results_variances.S)
    p_estimator.scale_variables_from_trajectory('C',results_variances.C)
    
    # dont push bounds i am giving you a good guess
    options = dict()
    #options['nlp_scaling_method'] = 'user-scaling'
    #options['mu_strategy'] = 'adaptive'
    options['mu_init'] = 1e-6
    options['bound_push'] =1e-6
    results_pyomo = p_estimator.run_opt('ipopt_sens',
                                        tee=True,
                                        solver_opts = options,
                                        variances=sigmas,
                                        with_d_vars=True,
                                        covariance=True)

    
    print("The estimated parameters are:")
    for k,v in six.iteritems(results_pyomo.P):
        print(k, v)

    # display results
    if with_plots:
        results_pyomo.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")

        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
    
        plt.show()
