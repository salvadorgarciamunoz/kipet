
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 (From Sawall et.al.)
# Basic simulation of ODE with spectral data using pyomo discretization 
#
#		\frac{dZ_a}{dt} = -k*Z_a	Z_a(0) = 1
#		\frac{dZ_b}{dt} = k*Z_a		Z_b(0) = 0
#
#               C_a(t_i) = Z_a(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 

from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.opt.ParameterEstimator import *
from kipet.opt.VarianceEstimator import *
import matplotlib.pyplot as plt

from kipet.utils.data_tools import *
import os
import sys
import inspect

def Lorentzian_parameters():
    params_a = dict()
    params_a['alphas'] = [2.0,1.3]
    params_a['betas'] = [50.0,200.0]
    params_a['gammas'] = [30000.0,1000.0]

    params_b = dict()
    params_b['alphas'] = [1.0,0.2,2.0,1.0]
    params_b['betas'] = [150.0,170.0,200.0,250.0]
    params_b['gammas'] = [100.0,30000.0,100.0,100.0]

    return {'A':params_a,'B':params_b}

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
            
    # read 500x2 S matrix
    wl_span = np.arange(180,230,10)
    S_parameters = Lorentzian_parameters()
    S_frame = generate_absorbance_data(wl_span,S_parameters)

    # components
    concentrations = {'A':1,'B':0}
    # create template model 
    builder = TemplateBuilder()    
    builder.add_mixture_component(concentrations)
    builder.add_parameter('k',0.1)
    builder.add_absorption_data(S_frame)
    builder.add_measurement_times([i for i in range(0,50,10)])
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k']*m.Z[t,'A']
        exprs['B'] = m.P['k']*m.Z[t,'A']
        return exprs

    builder.set_odes_rule(rule_odes)
    sim_model = builder.create_pyomo_model(0.0,40.0)
    
    # create instance of simulator
    simulator = PyomoSimulator(sim_model)
    simulator.apply_discretization('dae.collocation',nfe=4,ncp=1,scheme='LAGRANGE-RADAU')
    
    # simulate
    sigmas = {'device':1e-10,
              'A':1e-5,
              'B':1e-7}    
    results_sim = simulator.run_sim('ipopt',tee=True,variances=sigmas, seed=123453256)

    results_sim.C.plot.line()

    results_sim.S.plot.line()
    plt.show()
    
    sys.exit()
    
    #################################################################################
    builder = TemplateBuilder()    
    builder.add_mixture_component(concentrations)
    builder.add_parameter('k')
    builder.add_P_bounds('k',(0.0,1.0))
    builder.add_spectral_data(results_sim.D)
    builder.set_odes_rule(rule_odes)
    
    opt_model = builder.create_pyomo_model(0.0,200.0)
    
    v_estimator = VarianceEstimator(opt_model)
    v_estimator.apply_discretization('dae.collocation',nfe=100,ncp=1,scheme='LAGRANGE-RADAU')
    
    v_estimator.initialize_from_trajectory('Z',results_sim.Z)
    v_estimator.initialize_from_trajectory('S',results_sim.S)

    v_estimator.scale_variables_from_trajectory('Z',results_sim.Z)
    v_estimator.scale_variables_from_trajectory('S',results_sim.S)

    options = {}#{'mu_init': 1e-6, 'bound_push':  1e-6}
    results_variances = v_estimator.run_opt('ipopt',
                                            tee=True,
                                            solver_options=options,
                                            tolerance=1e-6)

    print "\nThe estimated variances are:\n"
    for k,v in results_variances.sigma_sq.iteritems():
        print k,v
    sigmas = results_variances.sigma_sq

    #################################################################################
    
    builder = TemplateBuilder()    
    builder.add_mixture_component(concentrations)
    builder.add_parameter('k')
    builder.add_P_bounds('k',(0.0,10.0))
    builder.add_spectral_data(results_sim.D)
    builder.set_odes_rule(rule_odes)
    
    opt_model = builder.create_pyomo_model(0.0,200.0)

    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=100,ncp=1,scheme='LAGRANGE-RADAU')
    
    p_estimator.initialize_from_trajectory('Z',results_variances.Z)
    p_estimator.initialize_from_trajectory('S',results_variances.S)

    p_estimator.scale_variables_from_trajectory('Z',results_variances.Z)
    p_estimator.scale_variables_from_trajectory('S',results_variances.S)
    
    results_p = p_estimator.run_opt('ipopt',
                                    tee=True,
                                    variances=sigmas)

    print "The estimated parameters are:"
    for k,v in opt_model.P.iteritems():
        print k,v.value

    
    if with_plots:
        # display concentration and absorbance results
        results_p.C.plot.line(legend=True)
        plt.plot(results_sim.C.index,results_sim.C['A'],'*',
                 results_sim.C.index,results_sim.C['B'],'*')
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_p.S.plot.line(legend=True)
        plt.plot(results_sim.S.index,results_sim.S['A'],'*',
                 results_sim.S.index,results_sim.S['B'],'*')
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")
        plt.show()
    
