#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 (From Sawall et.al.)
# First example from WF paper simulation of ODE system using pyomo discretization and IPOPT
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2*Z_b	                Z_c(0) = 0
from __future__ import print_function
from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.opt.ParameterEstimator import *
from kipet.utils.data_tools import *
import matplotlib.pyplot as plt
import numpy as np
import sys

import pickle


if __name__ == "__main__":


    #filename =  'original_data.csv'
    filename =  'trimmed.csv'
    D_frame = read_spectral_data_from_csv(filename)
    print("Dimensions of D:", D_frame.shape)
    
    """
    D_frame.T.plot(legend=False)
    plt.show()
    sys.exit()
    """
    # create template model 
    builder = TemplateBuilder()    

    # components
    components = dict()
    components['AH']   = 0.395555
    components['B']    = 0.0351202
    components['C']    = 0.0
    components['BH+']  = 0.0
    components['A-']   = 0.0
    components['AC-']  = 0.0
    components['P']    = 0.0
        
    builder.add_mixture_component(components)

    # add algebraics
    algebraics = [0,1,2,3,4] # the indices of the rate rxns
    builder.add_algebraic_variable(algebraics)

    """
    # add parameters
    params = dict()
    params['k0'] = 5.0
    params['k1'] = 5.0
    params['k2'] = 1.0
    params['k3'] = 5.0
    params['k4'] = 1.0
    builder.add_parameter(params)
    """

    params = dict()
    params['k0'] = 49.7796
    params['k1'] = 8.93156
    params['k2'] = 1.31765
    params['k3'] = 0.310870
    params['k4'] = 3.87809
    
    builder.add_parameter(params)
    """
    builder.add_parameter('k0',bounds=(0.0,100.0))
    builder.add_parameter('k1',bounds=(0.0,20.0))
    builder.add_parameter('k2',bounds=(0.0,10.0))
    builder.add_parameter('k3',bounds=(0.0,20.0))
    builder.add_parameter('k4',bounds=(0.0,10.0))
    """
    builder.add_spectral_data(D_frame)
    
    # add additional state variables
    extra_states = dict()
    extra_states['V'] = 0.0629418
    
    builder.add_complementary_state_variable(extra_states)

    gammas = dict()
    gammas['AH']   = [-1, 0, 0,-1, 0]    
    gammas['B']    = [-1, 0, 0, 0, 1]   
    gammas['C']    = [ 0,-1, 1, 0, 0]  
    gammas['BH+']  = [ 1, 0, 0, 0,-1]   
    gammas['A-']   = [ 1,-1, 1, 1, 0] 
    gammas['AC-']  = [ 0, 1,-1,-1,-1]
    gammas['P']    = [ 0, 0, 0, 1, 1]
        
    def rule_algebraics(m,t):
        r = list()
        r.append(m.Y[t,0]-m.P['k0']*m.Z[t,'AH']*m.Z[t,'B'])
        r.append(m.Y[t,1]-m.P['k1']*m.Z[t,'A-']*m.Z[t,'C'])
        r.append(m.Y[t,2]-m.P['k2']*m.Z[t,'AC-'])
        r.append(m.Y[t,3]-m.P['k3']*m.Z[t,'AC-']*m.Z[t,'AH'])
        r.append(m.Y[t,4]-m.P['k4']*m.Z[t,'AC-']*m.Z[t,'BH+'])
        return r

    builder.set_algebraics_rule(rule_algebraics)
    
    def rule_odes(m,t):
        exprs = dict()
        eta = 1e-4
        step = 0.5*((t+1)/((t+1)**2+eta**2)**0.5+(210.0-t)/((210.0-t)**2+eta**2)**0.5)
        exprs['V'] = 7.27609e-05*step
        V = m.X[t,'V']
        # mass balances
        for c in m.mixture_components:
            exprs[c] = sum(gammas[c][j]*m.Y[t,j] for j in m.algebraics) - exprs['V']/V*m.Z[t,c]
            if c=='C':
                exprs[c] += 0.02247311828/(m.X[t,'V']*210)*step
        return exprs

    builder.set_odes_rule(rule_odes)
    

    model = builder.create_pyomo_model(0.0,1400)    

    #model.pprint()
    
    opt = ParameterEstimator(model)
    # defines the discrete points wanted in the concentration profile
    opt.apply_discretization('dae.collocation',nfe=80,ncp=1,scheme='LAGRANGE-RADAU')

    # good initialization
    initialization = pd.read_csv("init_Z.csv",index_col=0)
    opt.initialize_from_trajectory('Z',initialization)
    initialization = pd.read_csv("init_X.csv",index_col=0)
    opt.initialize_from_trajectory('X',initialization)
    
    
    p_guess = dict()
    p_guess['k0'] = 5.0
    p_guess['k1'] = 5.0
    p_guess['k2'] = 1.0
    p_guess['k3'] = 5.0
    p_guess['k4'] = 1.0
    
    raw_results = opt.run_lsq_given_P('ipopt',p_guess,tee=True,with_bounds=True)

    opt.initialize_from_trajectory('Z',raw_results.Z)
    opt.initialize_from_trajectory('S',raw_results.S)
    opt.initialize_from_trajectory('C',raw_results.C)
    opt.initialize_from_trajectory('Y',raw_results.Y)

    raw_results.S.plot()
    raw_results.C.plot()
    plt.show()
    solver_options = dict()
    #solver_options['bound_relax_factor'] = 0.0
    #solver_options['mu_init'] =  1e-4
    #solver_options['bound_push'] = 1e-3
    
    # fixes the variances for now
    sigmas = {'device':1.11569e-9,
              'AH':2.29718e-6,
              'B':2.37659e-3,
              'C':2.60090e-6,
              'P':3.51299e-8}
    
    results = opt.run_opt('ipopt',
                          tee=True,
                          solver_opts = solver_options,
                          variances=sigmas)

    
    # display concentration results    
    results.C.plot.line(legend=True)
    plt.xlabel("time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile")

    results.Y.plot.line()
    plt.xlabel("time (s)")
    plt.ylabel("rxn rates (mol/L*s)")
    plt.title("Rates of rxn")

    results.S.plot.line(legend=True)
    plt.xlabel("Wavelength (cm)")
    plt.ylabel("Absorbance (L/(mol cm))")
    plt.title("Absorbance  Profile")
    
    plt.show()

