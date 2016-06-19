#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 (From Sawall et.al.)
# Basic simulation of ODE with spectral data using pyomo discretization 
#
#		\frac{dC_a}{dt} = -k_1*C_a	                C_a(0) = 1
#		\frac{dC_b}{dt} = k_1*C_a - k_2*C_b		C_b(0) = 0
#               \frac{dC_c}{dt} = k_2*C_b	                C_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 

from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.opt.Optimizer import *
import matplotlib.pyplot as plt

from kipet.utils.data_tools import *
import os

if __name__ == "__main__":

    # read 401x301 spectra matrix D_{i,j}
    # this defines the measurement points t_i and l_j as well

    filename = 'data_sets{}Dij_case51b.csv'.format(os.sep)
    D_frame = read_spectral_data_from_csv(filename)
    
    # create template model 
    builder = TemplateBuilder()    
    components = {'A':1.0,'B':0,'C':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1',0.30)
    builder.add_parameter('k2',0.05)
    # includes spectra data in the template and defines measurement sets
    builder.add_spectral_data(D_frame)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.C[t,'A']
        exprs['B'] = m.P['k1']*m.C[t,'A']-m.P['k2']*m.C[t,'B']
        exprs['C'] = m.P['k2']*m.C[t,'B']
        return exprs

    builder.set_rule_ode_expressions_dict(rule_odes)
    
    # create an instance of a pyomo model template
    # the template includes
    #   - C variables indexed over time and components names e.g. m.C[t,'A']
    #   - C_noise variables indexed over measurement t_i and components names e.g. m.C_noise[t_i,'A']
    #   - P parameters indexed over the parameter names m.P['k']
    #   - D spectra data indexed over the t_i, l_j measurement points m.D[t_i,l_j]
    pyomo_model = builder.create_pyomo_model(0.0,12.0)

    # create instance of simulator
    simulator = PyomoSimulator(pyomo_model)
    # defines the discrete points wanted in the profiles (does not include measurement points)
    simulator.apply_discretization('dae.collocation',nfe=30,ncp=1,scheme='LAGRANGE-RADAU')
    # simulate
    results_sim = simulator.run_sim('ipopt',tee=True)

    ######################################
    builder2 = TemplateBuilder()    
    builder2.add_mixture_component(components)

    # note the parameter is not fixed
    builder2.add_parameter('k1')
    builder2.add_parameter('k2')
    builder2.add_spectral_data(D_frame)

    # define explicit system of ODEs
    def rule_odes2(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.C[t,'A']
        exprs['B'] = m.P['k1']*m.C[t,'A']-m.P['k2']*m.C[t,'B']
        exprs['C'] = m.P['k2']*m.C[t,'B']
        return exprs

    builder2.set_rule_ode_expressions_dict(rule_odes2)

    pyomo_model2 = builder2.create_pyomo_model(0.0,12.0)
    
    #pyomo_model2.P['k1'].value = 2.0
    #pyomo_model2.P['k2'].value = 0.2
    #pyomo_model2.P['k1'].fixed = True
    #pyomo_model2.P['k2'].fixed = True
    
    optimizer = Optimizer(pyomo_model2)

    optimizer.apply_discretization('dae.collocation',nfe=30,ncp=1,scheme='LAGRANGE-RADAU')

    # Provide good initial guess
    optimizer.initialize_from_trajectory('C',results_sim.C)
    optimizer.initialize_from_trajectory('dCdt',results_sim.dCdt)
    optimizer.initialize_from_trajectory('S',results_sim.S)
    optimizer.initialize_from_trajectory('C_noise',results_sim.C_noise)

    #CheckInstanceFeasibility(pyomo_model2, 1e-3)
    # dont push bounds i am giving you a good guess
    solver_options = dict()
    #solver_options['bound_relax_factor'] = 0.0
    #solver_options['mu_init'] =  1e-4
    #solver_options['bound_push'] = 1e-3

    # fixes the variances for now
    sigmas = {'device':7.25435e-6,
              'A':4.29616e-6,
              'B':1.11297e-5,
              'C':1.07905e-5}
    
    results_pyomo = optimizer.run_opt('ipopt',
                                      tee=True,
                                      solver_opts = solver_options,
                                      std_deviations=sigmas)

    print "The estimated parameters are:"
    for k,v in results_pyomo.P.iteritems():
        print k,v

    # display results
    plt.plot(results_pyomo.C)
    plt.plot(results_sim.C_noise.index,results_sim.C_noise['A'],'*',
             results_sim.C_noise.index,results_sim.C_noise['B'],'*',
             results_sim.C_noise.index,results_sim.C_noise['C'],'*')
    plt.xlabel("time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile")

    plt.figure()

    plt.plot(results_pyomo.S)
    plt.plot(results_sim.S.index,results_sim.S['A'],'*',
             results_sim.S.index,results_sim.S['B'],'*',
             results_sim.S.index,results_sim.S['C'],'*')
    plt.xlabel("Wavelength (cm)")
    plt.ylabel("Absorbance (L/(mol cm))")
    plt.title("Absorbance  Profile")
    
    plt.show()


