
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 (From Sawall et.al.)
# Basic simulation of ODE with spectral data using pyomo discretization 
#
#		\frac{dZ_a}{dt} = -k*Z_a	Z_a(0) = 1
#		\frac{dZ_b}{dt} = k*Z_a		Z_b(0) = 0
#
#               C_a(t_i) = Z_a(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 

from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.opt.ParameterEstimator import *
from kipet.opt.VarianceEstimator import *
import matplotlib.pyplot as plt

from kipet.utils.data_tools import *
import os
import sys
import inspect

if __name__ == "__main__":

    # Load spectral data
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), '..','data_sets'))
    filename =  os.path.join(dataDirectory,'Dij_case51b.csv')
    D_frame = read_spectral_data_from_csv(filename)
    
    # build dae block for optimization problems
    #################################################################################    
    builder = TemplateBuilder()    
    components = {'A':1.0,'B':0,'C':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1')
    builder.add_parameter('k2')
    builder.add_P_bounds('k1',(0.0,1.0))
    builder.add_P_bounds('k2',(0.0,1.0))
    builder.add_spectral_data(D_frame)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    builder.set_odes_rule(rule_odes)
    opt_model = builder.create_pyomo_model(0.0,12.0)

    v_estimator = VarianceEstimator(opt_model)
    v_estimator.apply_discretization('dae.collocation',nfe=60,ncp=1,scheme='LAGRANGE-RADAU')

    # Provide good initial guess
    
    p_guess = {'k1':0.5,'k2':0.2}
    raw_results = v_estimator.run_lsq_given_P('ipopt',p_guess,tee=False)
    
    v_estimator.initialize_from_trajectory('Z',raw_results.Z)
    v_estimator.initialize_from_trajectory('S',raw_results.S)
    v_estimator.initialize_from_trajectory('C',raw_results.C)
    

    options = {}
    A_set = [l for i,l in enumerate(opt_model.meas_lambdas) if i%7]
    results_variances = v_estimator.run_opt('ipopt',
                                            tee=True,
                                            solver_options=options,
                                            tolerance=1.0e-5,
                                            max_iter=20,
                                            subset_lambdas=A_set)

    print "\nThe estimated variances are:\n"
    for k,v in results_variances.sigma_sq.iteritems():
        print k,v
        
    sigmas = results_variances.sigma_sq
    
    #################################################################################
    opt_model = builder.create_pyomo_model(0.0,12.0)

    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')
    
    # Provide good initial guess obtained by variance estimation
    p_estimator.initialize_from_trajectory('Z',results_variances.Z)
    p_estimator.initialize_from_trajectory('S',results_variances.S)
    p_estimator.initialize_from_trajectory('C',results_variances.C)

    p_estimator.scale_variables_from_trajectory('Z',results_variances.Z)
    p_estimator.scale_variables_from_trajectory('S',results_variances.S)
    p_estimator.scale_variables_from_trajectory('C',results_variances.C)
    
    # dont push bounds i am giving you a good guess
    options = dict()
    options['nlp_scaling_method'] = 'user-scaling'
    #options['mu_init'] = 1e-6
    #options['bound_push'] =1e-6
    results_pyomo = p_estimator.run_opt('ipopt',
                                      tee=True,
                                      solver_opts = options,
                                      variances=sigmas)

    print "The estimated parameters are:"
    for k,v in results_pyomo.P.iteritems():
        print k,v

    tol = 1e-1
    assert(abs(results_pyomo.P['k1']-0.3)<tol)
    assert(abs(results_pyomo.P['k2']-0.05)<tol)
    
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


