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

from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.data_tools import *
import matplotlib.pyplot as plt

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
    filename =  os.path.join(dataDirectory,'Ad_Sawall_C_data.csv')
    D_frame = read_concentration_data_from_csv(filename)

    ######################################
    builder = TemplateBuilder()    
    components = {'A':1,'B':0.8,'C':0,'D':0}
    builder.add_mixture_component(components)

    # note the parameter is not fixed
    builder.add_parameter('k1',bounds=(0.0,5.0))
    builder.add_parameter('k2',bounds=(0.0,5.0))
    builder.add_concentration_data(D_frame)

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
   
      
    sigmas = {'D':4.095062027416734e-10,
              'A':1.367860752636192e-09,
              'B':0.001595242107179487,
              'C':1.3733014603900695e-09}
 
    #=========================================================================
    # USER INPUT SECTION - PARAMETER ESTIMATION 
    #=========================================================================
    #################################################################################

    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=60,ncp=1,scheme='LAGRANGE-RADAU')
    
    options = dict()
    #options['nlp_scaling_method'] = 'user-scaling'
    #options['mu_strategy'] = 'adaptive'
    options['mu_init'] = 1e-6
    options['bound_push'] =1e-6
    results_pyomo = p_estimator.run_opt('ipopt_sens',
                                        tee=True,
                                        solver_opts = options,
                                        variances=sigmas,
                                        covariance=True)

    
    print("The estimated parameters are:")
    for k,v in six.iteritems(results_pyomo.P):
        print(k, v)

    # display results
    if with_plots:
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
    
        plt.show()
