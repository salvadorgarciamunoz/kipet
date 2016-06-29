#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 3 (From Sawall et.al.)
# Basic extimation of kinetic paramenter using pyomo discretization 
#
#         min \sum_i^{nt}\sum_j^{nl}( D_{i,j} -\sum_{k}^{nc} C_k(t_i)*S(l_j))**2/\delta
#              + \sum_i^{nt}\sum_k^{nc}(C_k(t_i)-Z_k(t_i))**2/\sigma_k       
#
#		\frac{dZ_a}{dt} = -k*Z_a	Z_a(0) = 1
#		\frac{dZ_b}{dt} = k*Z_a		Z_b(0) = 0
#
#               C_a(t_i) = Z_a(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 


from kipet.model.TemplateBuilder import *
from kipet.opt.Optimizer import *
import matplotlib.pyplot as plt

from kipet.utils.data_tools import *
import sys
import os

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    # read 200x500 spectra matrix D_{i,j}
    # this defines the measurement points t_i and l_j as well
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), '..','..','data_sets'))
    filename = os.path.join(dataDirectory,'Dij_sawall.txt')
    D_frame = read_spectral_data_from_txt(filename)

    # create template model 
    builder = TemplateBuilder()    
    builder.add_mixture_component({'A':1,'B':0})
    builder.add_parameter('k',0.01)
    builder.add_spectral_data(D_frame)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k']*m.Z[t,'A']
        exprs['B'] = m.P['k']*m.Z[t,'A']
        return exprs

    builder.set_odes_rule(rule_odes)
    
    pyomo_sim_model = builder.create_pyomo_model(0.0,200.0)

    simulator = PyomoSimulator(pyomo_sim_model)
    # defines the discrete points wanted in the profiles (does not include measurement points)
    simulator.apply_discretization('dae.collocation',nfe=100,ncp=1,scheme='LAGRANGE-RADAU')
    # simulate
    results_sim = simulator.run_sim('ipopt',tee=True)

    ##########################################################
    
    builder2 = TemplateBuilder()    
    builder2.add_mixture_component({'A':1,'B':0})

    # note the parameter is not fixed
    builder2.add_parameter('k')
    builder2.add_spectral_data(D_frame)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k']*m.Z[t,'A']
        exprs['B'] = m.P['k']*m.Z[t,'A']
        return exprs

    builder2.set_odes_rule(rule_odes)
    
    pyomo_model = builder2.create_pyomo_model(0.0,200.0)
        
    optimizer = Optimizer(pyomo_model)
    
    optimizer.apply_discretization('dae.collocation',nfe=30,ncp=3,scheme='LAGRANGE-RADAU')

    # Provide good initial guess
    optimizer.initialize_from_trajectory('Z',results_sim.Z)
    optimizer.initialize_from_trajectory('S',results_sim.S)
    optimizer.initialize_from_trajectory('C',results_sim.C)

    # dont push bounds i am giving you a good guess
    solver_options = {'mu_init': 1e-10, 'bound_push':  1e-8}
    #solver_options = {'bound_relax_factor':0, 'bound_push':  1e-8}
    # fixes the standard deaviations for now
    sigmas = {'device':1,'A':1,'B':1}
    results_pyomo = optimizer.run_opt('ipopt',
                                      tee=True,
                                      solver_opts = solver_options,
                                      variances=sigmas)

    print "The estimated parameters are:"
    for k,v in results_pyomo.P.iteritems():
        print k,v

    tol =1e-4
    assert(abs(results_pyomo.P['k']-0.01)<tol)
    # display results
    if with_plots:
        results_pyomo.C.plot.line(legend=True)
        plt.plot(results_sim.C.index,results_sim.C['A'],'*',
                 results_sim.C.index,results_sim.C['B'],'*')
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_pyomo.S.plot.line(legend=True)
        plt.plot(results_sim.S.index,results_sim.S['A'],'*',
                 results_sim.S.index,results_sim.S['B'],'*')
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")

        plt.show()

