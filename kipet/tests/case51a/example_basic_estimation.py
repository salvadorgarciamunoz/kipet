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
from kipet.sim.CasadiSimulator import *
import matplotlib.pyplot as plt

from kipet.utils.data_tools import *
import os

import sys
from pyomo_utils import *

if __name__ == "__main__":

    
    # first simulate to provide initial guesses
    # read 100x3 S matrix
    # this defines the measurement lambdas l_j but the t_i still need to be passed
    filename = 'data_sets{}Slk_case51.txt'.format(os.sep)
    S_frame = read_absorption_data_from_txt(filename)

    # create template model
    components = {'A':1,'B':0,'C':0}
    builder = TemplateBuilder()    
    builder.add_mixture_component(components)
    builder.add_parameter('k1',2.0)
    builder.add_parameter('k2',0.2)
    # includes absorption data in the template and defines measurement sets
    builder.add_absorption_data(S_frame)
    builder.add_measurement_times([i*0.0333 for i in range(300)])

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.C[t,'A']
        exprs['B'] = m.P['k1']*m.C[t,'A']-m.P['k2']*m.C[t,'B']
        exprs['C'] = m.P['k2']*m.C[t,'B']
        return exprs

    builder.set_rule_ode_expressions_dict(rule_odes)
    
    # create an instance of a casadi model template
    # the template includes
    #   - C variables indexed over time and components names e.g. m.C[t,'A']
    #   - C_noise variables indexed over measurement t_i and components names e.g. m.C_noise[t_i,'A']
    #   - P parameters indexed over the parameter names m.P['k']
    #   - D spectra data indexed over the t_i, l_j measurement points m.D[t_i,l_j]
    casadi_model = builder.create_casadi_model(0.0,10.0)

    # create instance of simulator
    sim = CasadiSimulator(casadi_model)
    # defines the discrete points wanted in the profiles (does not include measurement points)
    sim.apply_discretization('integrator',nfe=700)
    # simulate
    results_casadi = sim.run_sim("cvodes")

    ##########################################################
    
    builder2 = TemplateBuilder()    
    builder2.add_mixture_component(components)

    # note the parameter is not fixed
    builder2.add_parameter('k1')
    builder2.add_parameter('k2')
    builder2.add_spectral_data(results_casadi.D)

    # define explicit system of ODEs
    def rule_odes2(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.C[t,'A']
        exprs['B'] = m.P['k1']*m.C[t,'A']-m.P['k2']*m.C[t,'B']
        exprs['C'] = m.P['k2']*m.C[t,'B']
        return exprs

    builder2.set_rule_ode_expressions_dict(rule_odes2)

    pyomo_model = builder2.create_pyomo_model(0.0,10.0)
    
    pyomo_model.P['k1'].value = 2.0
    pyomo_model.P['k2'].value = 0.2
    
    optimizer = Optimizer(pyomo_model)

    #pyomo_model.pprint()
    #sys.exit()
    
    optimizer.apply_discretization('dae.collocation',nfe=30,ncp=3,scheme='LAGRANGE-RADAU')

    # Provide good initial guess
    optimizer.initialize_from_trajectory('C',results_casadi.C)
    optimizer.initialize_from_trajectory('dCdt',results_casadi.dCdt)
    optimizer.initialize_from_trajectory('S',results_casadi.S)
    optimizer.initialize_from_trajectory('C_noise',results_casadi.C_noise)

    
    CheckInstanceFeasibility(pyomo_model, 1e-2)
    # dont push bounds i am giving you a good guess
    solver_options = dict()
    solver_options['bound_relax_factor'] = 0.0
    #solver_options['mu_init'] =  1e-10
    solver_options['bound_push'] = 1e-8
    

    # fixes the standard deaviations for now
    sigmas = {'device':1,'A':0.1,'B':1,'C':1}
    
    results_pyomo = optimizer.run_opt('ipopt',
                                      tee=True,
                                      solver_opts = solver_options,
                                      std_deviations=sigmas)

    print "The estimated parameters are:"
    for k,v in results_pyomo.P.iteritems():
        print k,v

    expr = 0.0
    for t in pyomo_model.measurement_times:
        expr += sum((pyomo_model.C_noise[t,k].value-pyomo_model.C[t,k].value)**2/pyomo_model.sigma[k].value**2 for k in pyomo_model.mixture_components)
    print "Concentration term",expr

    expr = 0.0
    for t in pyomo_model.measurement_times:
        for l in pyomo_model.measurement_lambdas:
            current = value(pyomo_model.spectral_data[t,l] - sum(pyomo_model.C_noise[t,k]*pyomo_model.S[l,k] for k in pyomo_model.mixture_components))
            expr+= value(current**2/pyomo_model.device_std_dev**2)
    print "Spectra term",expr
            
    # display results
    plt.plot(results_pyomo.C)
    plt.plot(results_casadi.C_noise.index,results_casadi.C_noise['A'],'*',
             results_casadi.C_noise.index,results_casadi.C_noise['B'],'*',
             results_casadi.C_noise.index,results_casadi.C_noise['C'],'*')
    plt.xlabel("time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile")

    plt.figure()

    plt.plot(results_pyomo.S)
    plt.plot(results_casadi.S.index,results_casadi.S['A'],'*',
             results_casadi.S.index,results_casadi.S['B'],'*',
             results_casadi.S.index,results_casadi.S['C'],'*')
    plt.xlabel("Wavelength (cm)")
    plt.ylabel("Absorbance (L/(mol cm))")
    plt.title("Absorbance  Profile")
    
    plt.show()
    
