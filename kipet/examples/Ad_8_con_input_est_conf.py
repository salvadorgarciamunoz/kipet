#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with known variances of spectral data using pyomo discretization 
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		   Z_b(0) = 0
#       \frac{dZ_c}{dt} = k_2*Z_b	                   Z_c(0) = 0
#       C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#       D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 
from __future__ import print_function
from __future__ import division
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
        
    # read 401x301 spectra matrix D_{i,j}
    # this defines the measurement points t_i and l_j as well
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ),'data_sets'))
    filename =  os.path.join(dataDirectory,'Ad1_C_data.csv')
    D_frame = read_concentration_data_from_csv(filename)

    builder = TemplateBuilder()    
    components = {'A':1.0,'B':0,'C':0}
    builder.add_mixture_component(components)

    # note the parameter is not fixed here
    builder.add_parameter('k1',bounds=(0.0,1.0))
    builder.add_parameter('k2',bounds=(0.0,1.0))
    builder.add_concentration_data(D_frame)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs

    builder.set_odes_rule(rule_odes)

    # Create instance
    pyomo_model = builder.create_pyomo_model(0.0,12.0)

    #=========================================================================
    # USER INPUT SECTION - PARAMETER ESTIMATION 
    #=========================================================================
        
    optimizer = ParameterEstimator(pyomo_model)

    optimizer.apply_discretization('dae.collocation',nfe=30,ncp=1,scheme='LAGRANGE-RADAU')

    solver_options = dict()
    solver_options['mu_strategy'] = 'adaptive'

    # fix the variances
    sigmas = {'device':7.25435e-6,
              'A':4.29616e-6,
              'B':1.11297e-5,
              'C':1.07905e-5}
    
    results_pyomo = optimizer.run_opt('k_aug',
                                      variances=sigmas,
                                      tee=True,
                                      solver_opts = solver_options,
                                      covariance=True)

    print("The estimated parameters are:")
    for k,v in six.iteritems(results_pyomo.P):
        print(k, v)
    #C_Dataframe = pd.DataFrame(data=results_pyomo.C)
    #write_concentration_data_to_csv('Ad1_C_data.csv',C_Dataframe)
    # display results
    if with_plots:
        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        plt.show()


