
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
from kipet.library.EstimabilityAnalyzer import *
import matplotlib.pyplot as plt
import os
import sys
import inspect
import six
import pandas as pd

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
    filename =  os.path.join(dataDirectory,'new_estim_problem_conc.csv')
    D_frame = read_concentration_data_from_csv(filename)

    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_spectral_data
    #################################################################################    
    builder = TemplateBuilder()    
    builder.add_mixture_component('A',0.3)
    builder.add_mixture_component('B',0.0)
    builder.add_mixture_component('C',0.0)
    builder.add_mixture_component('D',0.01)
    builder.add_mixture_component('E',0.0)
    
    #Following this we add the kinetic parameters
    builder.add_parameter('k1',bounds=(0.5,1.5))
    builder.add_parameter('k2',bounds=(0.0,0.3))
    builder.add_parameter('k3',bounds=(0.0,0.4))
    builder.add_parameter('k4',bounds=(0.0,0.6))
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']-m.P['k4']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']-m.P['k3']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']-m.P['k4']*m.Z[t,'C']
        exprs['D'] = m.P['k4']*m.Z[t,'A']-m.P['k3']*m.Z[t,'D']
        exprs['E'] = m.P['k3']*m.Z[t,'B']
        
        return exprs
    builder.add_concentration_data(D_frame)
    #Add these ODEs to our model template
    builder.set_odes_rule(rule_odes)
    opt_model = builder.create_pyomo_model(0.0,20.0)
    #opt_model.C.pprint()


    #=========================================================================
    # USER INPUT SECTION - PARAMETER ESTIMATION 
    #=========================================================================
    # In order to run the paramter estimation we create a pyomo model as described in section 4.3.4

    # and define our parameter estimation problem and discretization strategy
    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')
    
    #p_estimator.initialize_from_trajectory('Z',results_variances.Z)
    #p_estimator.initialize_from_trajectory('S',results_variances.S)
    #p_estimator.initialize_from_trajectory('C',results_variances.C)    
    # Here we use the estimability analysis tools
    e_analyzer = EstimabilityAnalyzer(opt_model)
    # define the uncertainty surrounding each of the parameters
    param_uncertainties = {'k1':0.09,'k2':0.01,'k3':0.02,'k4':0.01}
    # measurement scaling
    meas_uncertainty = 0.001
    e_analyzer.rank_params_yao(meas_scaling = meas_uncertainty, param_scaling = param_uncertainties)

    
    # Again we provide options for the solver, this time providing the scaling that we set above
    options = dict()
    #options['nlp_scaling_method'] = 'gradient-based'
    #options['bound_relax_factor'] = 0
    # finally we run the optimization
    #results_pyomo = p_estimator.run_opt('ipopt_sens',
    #                                    variances=sigmas,
    #                                  tee=True,
    #                                  solver_opts = options,
    #                                  covariance=True)

    # And display the results
    #print("The estimated parameters are:")
    #for k,v in six.iteritems(results_pyomo.P):
    #    print(k, v)

    #C_Dataframe = pd.DataFrame(data=results_pyomo.C)
    #write_concentration_data_to_csv('Ex_1_C_data.csv',C_Dataframe)
        
    # display results
    #if with_plots:
    #    results_pyomo.C.plot.line(legend=True)
    #    plt.xlabel("time (s)")
    #    plt.ylabel("Concentration (mol/L)")
    #    plt.title("Concentration Profile")
        
    #    results_pyomo.Z.plot.line(legend=True)
    #    plt.xlabel("time (s)")
    #    plt.ylabel("Concentration (mol/L)")
    #    plt.title("Concentration Profile")

    
    #    plt.show()
    

