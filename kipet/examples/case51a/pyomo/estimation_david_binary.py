#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with know variances of spectral data using pyomo discretization
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2*Z_b	                Z_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 
from __future__ import print_function
from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.opt.ParameterEstimator_david import *
import matplotlib.pyplot as plt
from itertools import product
from kipet.utils.data_tools import *
import inspect
import sys
import os

if __name__ == "__main__":
    
    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    # read 300x100 spectra matrix D_{i,j}
    # this defines the measurement points t_i and l_j as well
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), '..','..','data_sets'))
    filename =  os.path.join(dataDirectory,'Dij.txt')
    D_frame = read_spectral_data_from_txt(filename)
    

    ######################################
    builder = TemplateBuilder()    
    components = {'A': 1e-3,'B': 0,'C': 0}
    builder.add_mixture_component(components)
    # note the parameter is not fixed
    builder.add_parameter('k1',bounds=(0.0,5.0))
    builder.add_parameter('k2',bounds=(0.0,1.0))
    builder.add_spectral_data(D_frame)
    
    # define explicit system of ODEs
    def rule_odes2(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs

    builder.set_odes_rule(rule_odes2)
    s = set(['A', 'B', 'C'])
    s2 = set(product(s, s))
    binary = dict.fromkeys(s2, 0)
    binary['A', 'A'] = 1
    binary['B', 'B'] = 1
    binary['C', 'C'] = 1
    print(binary)
    pyomo_model2 = builder.create_pyomo_model(0.0,10.0)
    optimizer = ParameterEstimator(pyomo_model2, binary=binary)
    optimizer.apply_discretization('dae.collocation',nfe=30,ncp=1,scheme='LAGRANGE-RADAU')

    # Provide good initial guess
    p_guess = {'k1':2.0,'k2':0.5}
    raw_results = optimizer.run_lsq_given_P('ipopt',p_guess,tee=False)
    
    optimizer.initialize_from_trajectory('Z',raw_results.Z)
    optimizer.initialize_from_trajectory('S',raw_results.S)
    optimizer.initialize_from_trajectory('C',raw_results.C)
    
    # dont push bounds i am giving you a good guess
    solver_options = dict()
    solver_options['bound_relax_factor'] = 0.0
    solver_options['mu_init'] =  1e-4
    solver_options['bound_push'] = 1e-3
    
    # fixes the variances for now
    sigmas = {'device':1.87309e-6,
              'A':1.55189e-11,
              'B':8.54601e-11,
              'C':6.11854e-11}
    
    results_pyomo = optimizer.run_opt('ipopt',
                                      tee=True,
                                      solver_opts = solver_options,
                                      variances=sigmas)
    optimizer.model.pprint(filename="whatevs")
    print("The estimated parameters are:")
    for k, v in results_pyomo.P.items():
        print(k, v)

    tol = 1e-1
    assert(abs(results_pyomo.P['k1']-2.0)<tol)
    assert(abs(results_pyomo.P['k2']-0.2)<tol)
        
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


