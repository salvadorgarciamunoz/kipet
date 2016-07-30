#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________


# Sample Problem 
# Estimation with known variances of spectral data using pyomo discretization 
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
import matplotlib.pyplot as plt

from kipet.utils.data_tools import *
import inspect
import sys
import os


if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
            
    # read 400x101 spectra matrix D_{i,j}
    # this defines the measurement points t_i and l_j as well
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), '..','..','data_sets'))
    filename =  os.path.join(dataDirectory,'Dij_case51c.txt')
    D_frame = read_spectral_data_from_txt(filename)

    ######################################
    builder = TemplateBuilder()    
    components = {'A':1,'B':0.8,'C':0,'D':0}
    builder.add_mixture_component(components)

    # note the parameter is not fixed
    builder.add_parameter('k1', bounds=(0.0,5.0))
    builder.add_parameter('k2', bounds=(0.0,5.0))
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

    pyomo_model = builder.create_pyomo_model(0.0,10.0)

    optimizer = ParameterEstimator(pyomo_model)

    optimizer.apply_discretization('dae.collocation',nfe=30,ncp=3,scheme='LAGRANGE-RADAU')

    # Provide good initial guess
    p_guess = {'k1':2.0,'k2':1.0}
    raw_results = optimizer.run_lsq_given_P('ipopt',p_guess,tee=False)

    optimizer.initialize_from_trajectory('Z',raw_results.Z)
    optimizer.initialize_from_trajectory('S',raw_results.S)
    optimizer.initialize_from_trajectory('dZdt',raw_results.dZdt)
    optimizer.initialize_from_trajectory('C',raw_results.C)

    #CheckInstanceFeasibility(pyomo_model2, 1e-3)
    # dont push bounds i am giving you a good guess
    solver_options = dict()
    solver_options['bound_relax_factor'] = 0.0
    solver_options['mu_strategy'] = 'adaptive'
    #solver_options['mu_init'] =  1e-4
    #solver_options['bound_push'] = 1e-3

    # fixes the standard deaviations for now
    sigmas = {'device':1.66507e-6,
              'A':4.34682e-6,
              'B':2.62689e-6,
              'C':5.31808e-6,
              'D':2.72712e-6}
    
    results_pyomo = optimizer.run_opt('ipopt',
                                      tee=True,
                                      solver_opts = solver_options,
                                      variances=sigmas,
                                      with_d_vars=True)

    print "The estimated parameters are:"
    for k,v in results_pyomo.P.iteritems():
        print k,v

    tol = 2e-1
    assert(abs(results_pyomo.P['k1']-2.0)<tol)
    assert(abs(results_pyomo.P['k2']-1.0)<tol)
        
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


