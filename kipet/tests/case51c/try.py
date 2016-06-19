#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 (From Sawall et.al.)
# Basic simulation of ODE with spectral data using pyomo discretization 
#
#		\frac{dC_a}{dt} = -k_1*C_a*C_b	                                C_a(0) = 1
#		\frac{dC_b}{dt} = -k_1*C_a*C_b                   		C_b(0) = 0.8
#               \frac{dC_c}{dt} = k_1*C_a*C_b-2*k_2*C_c^2	                C_c(0) = 0
#               \frac{dC_d}{dt} = k_2*C_c^2             	                C_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 



from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.opt.Optimizer import *
import matplotlib.pyplot as plt

from kipet.utils.data_tools import *
import os
from pyomo_utils import *

if __name__ == "__main__":

    # read 400x101 spectra matrix D_{i,j}
    # this defines the measurement points t_i and l_j as well
    filename = 'data_sets{}Dij_case51c.txt'.format(os.sep)
    D_frame = read_spectral_data_from_txt(filename)

    #plot_spectral_data(D_frame,dimension='3D')
    #plt.show()
    
    # create template model 
    builder = TemplateBuilder()
    components = {'A':1,'B':0.8,'C':0,'D':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1',2.0)
    builder.add_parameter('k2',1.0)

    # includes spectra data in the template and defines measurement sets
    builder.add_spectral_data(D_frame)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.C[t,'A']*m.C[t,'B']
        exprs['B'] = -m.P['k1']*m.C[t,'A']*m.C[t,'B']
        exprs['C'] = m.P['k1']*m.C[t,'A']*m.C[t,'B']-2*m.P['k2']*m.C[t,'C']**2
        exprs['D'] = m.P['k2']*m.C[t,'C']**2
        return exprs
    
    builder.set_rule_ode_expressions_dict(rule_odes)
    
    # create an instance of a pyomo model template
    # the template includes
    #   - C variables indexed over time and components names e.g. m.C[t,'A']
    #   - C_noise variables indexed over measurement t_i and components names e.g. m.C_noise[t_i,'A']
    #   - P parameters indexed over the parameter names m.P['k']
    #   - D spectra data indexed over the t_i, l_j measurement points m.D[t_i,l_j]
    pyomo_model = builder.create_pyomo_model(0.0,10.0)

    # create instance of simulator
    simulator = PyomoSimulator(pyomo_model)
    # defines the discrete points wanted in the profiles (does not include measurement points)
    simulator.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')
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
        exprs['A'] = -m.P['k1']*m.C[t,'A']*m.C[t,'B']
        exprs['B'] = -m.P['k1']*m.C[t,'A']*m.C[t,'B']
        exprs['C'] = m.P['k1']*m.C[t,'A']*m.C[t,'B']-2*m.P['k2']*m.C[t,'C']**2
        exprs['D'] = m.P['k2']*m.C[t,'C']**2
        return exprs

    builder2.set_rule_ode_expressions_dict(rule_odes2)

    pyomo_model2 = builder2.create_pyomo_model(0.0,10.0)
    
    pyomo_model2.P['k1'].value = 2.0
    pyomo_model2.P['k2'].value = 1.0
    #pyomo_model2.P['k1'].fixed = True
    #pyomo_model2.P['k2'].fixed = True
    
    optimizer = Optimizer(pyomo_model2)

    optimizer.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')

    # Provide good initial guess
    optimizer.initialize_from_trajectory('C',results_sim.C)
    optimizer.initialize_from_trajectory('dCdt',results_sim.dCdt)
    optimizer.initialize_from_trajectory('S',results_sim.S)
    optimizer.initialize_from_trajectory('C_noise',results_sim.C_noise)

    #CheckInstanceFeasibility(pyomo_model2, 1e-3)
    # dont push bounds i am giving you a good guess
    solver_options = dict()
    solver_options['bound_relax_factor'] = 0.0
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
                                      std_deviations=sigmas)

    print "The estimated parameters are:"
    for k,v in results_pyomo.P.iteritems():
        print k,v

    """
    expr = 0.0
    for t in pyomo_model2.measurement_times:
        expr += sum((pyomo_model2.C_noise[t,k].value-pyomo_model2.C[t,k].value)**2/pyomo_model2.sigma_sq[k].value for k in pyomo_model2.mixture_components)
    print "Concentration term",expr

    expr = 0.0
    for t in pyomo_model2.measurement_times:
        for l in pyomo_model2.measurement_lambdas:
            current = value(pyomo_model2.spectral_data[t,l] - sum(pyomo_model2.C_noise[t,k]*pyomo_model2.S[l,k] for k in pyomo_model2.mixture_components))
            expr+= value(current**2/pyomo_model2.device_variance)
    print "Spectra term",expr
    """     
    # display results
    results_pyomo.C_noise.plot.line(legend=True)
    plt.plot(results_sim.C_noise.index,results_sim.C_noise['A'],'*',
             results_sim.C_noise.index,results_sim.C_noise['B'],'*',
             results_sim.C_noise.index,results_sim.C_noise['C'],'*',
             results_sim.C_noise.index,results_sim.C_noise['D'],'*')
    plt.xlabel("time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile")

    
    results_pyomo.S.plot.line(legend=True)

    plt.plot(results_sim.S.index,results_sim.S['A'],'*',
             results_sim.S.index,results_sim.S['B'],'*',
             results_sim.S.index,results_sim.S['C'],'*',
             results_sim.S.index,results_sim.S['D'],'*')
    plt.xlabel("Wavelength (cm)")
    plt.ylabel("Absorbance (L/(mol cm))")
    plt.title("Absorbance  Profile")
    
    plt.show()


