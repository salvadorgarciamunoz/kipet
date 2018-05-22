#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 (From Sawall et.al.)
# First example from WF paper simulation of ODE system using pyomo discretization and IPOPT
#
#		\frac{dZ_a}{dt} = -k_1*Z_a*Z_b	                                Z_a(0) = 1
#		\frac{dZ_b}{dt} = -k_1*Z_a*Z_b                   		Z_b(0) = 1
#               \frac{dZ_c}{dt} = k_1*Z_a*Z_b-2*k_2*Z_c^2	                Z_c(0) = 0
#               \frac{dZ_d}{dt} = k_2*Z_c^2             	                Z_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 

import matplotlib.pyplot as plt
from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.data_tools import *
import inspect
import sys
import os


if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
            
    # read 200*431 spectra matrix D_{i,j}
    # this defines the measurement points t_i and l_j as well
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ),'data_sets'))
    # filename =  os.path.join(dataDirectory,'Dij_case52a.txt')
    filename =  os.path.join(dataDirectory,'Dij_case52a.txt')
    D_frame = read_spectral_data_from_txt(filename)
    
    ######################################
    builder = TemplateBuilder()    
    components = {'A':211.45e-3,'B':180.285e-3,'C':3.187e-3}
    builder.add_mixture_component(components)

    # note the parameter is not fixed
    builder.add_parameter('k1',bounds=(0.0,1.0))
    builder.add_spectral_data(D_frame)

    # define explicit system of ODEs
    def rule_odes2(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']*m.Z[t,'B']
        exprs['B'] = -m.P['k1']*m.Z[t,'A']*m.Z[t,'B']
        exprs['C'] = m.P['k1']*m.Z[t,'A']*m.Z[t,'B']
        return exprs
    
    builder.set_odes_rule(rule_odes2)

    pyomo_model2 = builder.create_pyomo_model(0.0,200.0)

    optimizer = ParameterEstimator(pyomo_model2)

    optimizer.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')

    # Provide good initial guess
    p_guess = {'k1':0.006655}
    raw_results = optimizer.run_lsq_given_P('ipopt',p_guess,tee=False)
    
    optimizer.initialize_from_trajectory('Z',raw_results.Z)
    optimizer.initialize_from_trajectory('S',raw_results.S)
    optimizer.initialize_from_trajectory('dZdt',raw_results.dZdt)
    optimizer.initialize_from_trajectory('C',raw_results.C)

    optimizer.scale_variables_from_trajectory('Z',raw_results.Z)
    optimizer.scale_variables_from_trajectory('S',raw_results.S)
    optimizer.scale_variables_from_trajectory('C',raw_results.C)
    optimizer.scale_variables_from_trajectory('dZdt',raw_results.dZdt)
    
    #CheckInstanceFeasibility(pyomo_model2, 1e-3)
    # dont push bounds i am giving you a good guess
    options = dict()
    # options['mu_strategy'] = 'adaptive'
    # fixes the variances for now
    sigmas = {'device': 1.94554e-5,
              'A': 2.45887e-6,
              'B': 2.45887e-6,
              'C': 3.1296e-11}
    results_pyomo = optimizer.run_opt('ipopt_sens',
                                      tee=True,
                                      solver_options=options,
                                      variances=sigmas,
                                      tolerance=1e-4,
                                      max_iter=40,
                                      covariance=True)
    print("The estimated parameters are:")
    for k,v in six.iteritems(results_pyomo.P):
        print(k, v)
        
    # display results
    if with_plots:
        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")


        results_pyomo.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")

        plt.show()


