
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with generated absorbance data and one non-absorbing species (the product)
# This data is used to simulate a problem, generate a new D-matrix and then we use this 
# D-matrix to do variance and parameter estimation
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		Z_b(0) = 0
#       \frac{dZ_c}{dt} = k_2*Z_b	                Z_c(0) = 0
#       C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#        D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 

from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.opt.ParameterEstimator import *
from kipet.opt.VarianceEstimator import *
import matplotlib.pyplot as plt

from kipet.utils.data_tools import *
import os
import sys
import inspect

    #=========================================================================
    #USER INPUT SECTION - Parameters for the absorption generation from Lorentzian parameters
    #=========================================================================
   

def Lorentzian_parameters():
    params_a = dict()
    params_a['alphas'] = [2.0,1.3]
    params_a['betas'] = [50.0,200.0]
    params_a['gammas'] = [30000.0,1000.0]

    params_b = dict()
    params_b['alphas'] = [1.0,0.2,2.0,1.0]
    params_b['betas'] = [150.0,170.0,200.0,250.0]
    params_b['gammas'] = [100.0,30000.0,100.0,100.0]
    
   #This is not really needed as C is non-absorbing 
    params_c = dict()
    params_c['alphas'] = [1.0,0.2,2.0,1.0]
    params_c['betas'] = [150.0,170.0,200.0,250.0]
    params_c['gammas'] = [100.0,30000.0,100.0,100.0]

    return {'A':params_a,'B':params_b, 'C':params_c}

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False

    #=========================================================================
    #USER INPUT SECTION - MODEL BUILDING - See user guide for how this section functions
    #=========================================================================
               
    # read 500x2 S matrix
    wl_span = np.arange(180,230,10)
    S_parameters = Lorentzian_parameters()
    S_frame = generate_absorbance_data(wl_span,S_parameters)

    # components
    concentrations = {'A':1,'B':0,'C':0}
    # create template model 
    builder = TemplateBuilder()    
    builder.add_mixture_component(concentrations)
    builder.add_parameter('k1',0.2)
    builder.add_parameter('k2',0.1)
    builder.add_absorption_data(S_frame)
    builder.add_measurement_times([i for i in range(0,50,5)])
    
    write_absorption_data_to_txt('Sij_small.txt',S_frame)
    # define explicit system of ODEs
    # define explicit system of ODEs
    def rule_odes(m, t):
        exprs = dict()
        exprs['A'] = -m.P['k1'] * m.Z[t, 'A']
        exprs['B'] = m.P['k1'] * m.Z[t, 'A'] - m.P['k2'] * m.Z[t, 'B']
        exprs['C'] = m.P['k2'] * m.Z[t, 'B']
        return exprs


    builder.set_odes_rule(rule_odes)
    sim_model = builder.create_pyomo_model(0.0, 50.0)

    #################################################################################
    #: non absorbing species.

    non_abs = ['C']
    builder.set_non_absorbing_species(sim_model, non_abs)
    #################################################################################
    #=========================================================================
    #USER INPUT SECTION - SIMULATOR - See user guide for how this section functions
    #=========================================================================
     
    # create instance of simulator
    simulator = PyomoSimulator(sim_model)
    simulator.apply_discretization('dae.collocation',nfe=8,ncp=3,scheme='LAGRANGE-RADAU')
    
    # simulate with fixed variances
    sigmas = {'device':1e-8,
              'A':1e-6,
              'B':1e-7}    
    results_sim = simulator.run_sim('ipopt',tee=True,variances=sigmas, seed=123453256)

    results_sim.C.plot.line()

    results_sim.S.plot.line()
    plt.show()
    
    #=========================================================================
    #USER INPUT SECTION - Using the D matrix generated we re-run the system to estimate variance and parameters
    #=========================================================================
         
    #################################################################################
    builder = TemplateBuilder()    
    builder.add_mixture_component(concentrations)
    builder.add_parameter('k1',bounds=(0.0,1.0))
    builder.add_parameter('k2',bounds=(0.0,5.0))
    builder.add_spectral_data(results_sim.D)

    builder.set_odes_rule(rule_odes)
    
    opt_model = builder.create_pyomo_model(0.0,50.0)
        
    #################################################################################
    #: non absorbing species.

    builder.set_non_absorbing_species(opt_model, non_abs)
    #################################################################################
  
    v_estimator = VarianceEstimator(opt_model)
    v_estimator.apply_discretization('dae.collocation',nfe=8,ncp=3,scheme='LAGRANGE-RADAU')
    
    v_estimator.initialize_from_trajectory('Z',results_sim.Z)
    v_estimator.initialize_from_trajectory('S',results_sim.S)

    #v_estimator.scale_variables_from_trajectory('Z',results_sim.Z)
    #v_estimator.scale_variables_from_trajectory('S',results_sim.S)
    
    options = {}#{'mu_init': 1e-6, 'bound_push':  1e-6}
    results_variances = v_estimator.run_opt('ipopt',
                                            tee=True,
                                            solver_options=options,
                                            tolerance=1e-6)

    print("\nThe estimated variances are:\n")
    for k,v in six.iteritems(results_variances.sigma_sq):
        print(k, v)
    sigmas = results_variances.sigma_sq#results_variances.sigma_sq

    #################################################################################

    opt_model = builder.create_pyomo_model(0.0,50.0)
    builder.set_non_absorbing_species(opt_model, non_abs)
    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=8,ncp=3,scheme='LAGRANGE-RADAU')
    
    p_estimator.initialize_from_trajectory('Z',results_variances.Z)
    p_estimator.initialize_from_trajectory('S',results_variances.S)

    p_estimator.scale_variables_from_trajectory('Z',results_variances.Z)
    p_estimator.scale_variables_from_trajectory('S',results_variances.S)
    
    results_p = p_estimator.run_opt('ipopt',
                                    tee=True,
                                    variances=sigmas)

    print("The estimated parameters are:")
    for k,v in six.iteritems(opt_model.P):
        print(k, v.value)

    
    if with_plots:
        # display concentration and absorbance results
        results_p.C.plot.line(legend=True)
        plt.plot(results_sim.C.index,results_sim.C['A'],'*',
                 results_sim.C.index,results_sim.C['B'],'*',
                 results_sim.C.index,results_sim.C['C'],'*')
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_p.S.plot.line(legend=True)
        plt.plot(results_sim.S.index,results_sim.S['A'],'*',
                 results_sim.S.index,results_sim.S['B'],'*',
                 results_sim.S.index,results_sim.S['C'],'*')
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")
        plt.show()
    
