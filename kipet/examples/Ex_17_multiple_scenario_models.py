"""
Multiple Scenarios using a nested Schur decomposition strategy (models)

This uses a scipy.optimize.minimize method to control the outer problem and
uses Ipopt in the normal manner to solve the inner problems. These are solved
in the ubiquitous way that Kipet solves parameter fitting problems.

This is the first version of this code and uses three simulated examples all 
with different initial values. Each case has added noise. Two scenarios have
three random measurement points while the third has six. This is to show that
it works not only for different conditions, but also for different measuremnet
times.

This example requires the nested_schur module in the kipet library directory
in order to work. Once the scenarios are entered into Kipet, only a single
function call is needed to start the parameter fitting using decomposition.

TODO:  - Place into the MultipleExperimentsEstimator class
       - Enable it to use parallelization
       - Check if the TR approach is the best or if a simple line search is enough
       - Find out why the mixture warnings are occuring!
"""
import copy
import sys

import numpy as np
import pandas as pd

from kipet.library.data_tools import add_noise_to_signal
from kipet.library.NestedSchurDecompositionModelsPyNumero import NestedSchurDecomposition as NSD    

from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.TemplateBuilder import TemplateBuilder

def run_simulation(simulator, nfe=50, ncp=3, use_only_FE=True):
    """This is not necessary, but is used for generating data used in the
    estimation algorithm
    """
    simulator.apply_discretization('dae.collocation',
                                   ncp = ncp,
                                   nfe = nfe,
                                   scheme = 'LAGRANGE-RADAU')

    options = {'solver_opts' : dict(linear_solver='ma57')}
    
    for k, v in simulator.model.P.items():
        simulator.model.P[k].fix()
        
    results_pyomo = simulator.run_sim('ipopt_sens',
                                      tee=False,
                                      solver_options=options,
                                      )

    if with_plots:
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (h)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        
        # results_pyomo.X.plot.line(legend=True)
        # plt.xlabel("time (h)")
        # plt.ylabel("Temperature (K)")       
        # plt.title("Temperature Profiles")
        
        plt.show()
    
    Z_data = pd.DataFrame(results_pyomo.Z)
    #X_data = pd.DataFrame(results_pyomo.X)
    X_data = None
    
    if use_only_FE:
        
        t = np.linspace(0, ncp*nfe, nfe+1).astype(int)
        
        Z_data = Z_data.iloc[t, :]
        #X_data = X_data.iloc[t, :]
        
        Z_data.drop(index=0, inplace=True)
        #X_data.drop(index=0, inplace=True)
        
    return Z_data, X_data, results_pyomo

def add_exp_data_and_make_model(scneario):

    exp_data = pd.DataFrame(scenario[1])
    model_builder = copy.copy(scenario[0])
 
    conc_state_headers = model_builder._component_names & set(exp_data.columns)
    if len(conc_state_headers) > 0:
        model_builder.add_concentration_data(pd.DataFrame(exp_data[conc_state_headers].dropna()))
    
    comp_state_headers = model_builder._complementary_states & set(exp_data.columns)
    if len(comp_state_headers) > 0:
        model_builder.add_complementary_states_data(pd.DataFrame(exp_data[comp_state_headers].dropna()))
        
    model = model_builder.create_pyomo_model()
    
    return model

if __name__ == "__main__":

    with_plots = False
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
    # Set up the problem - here are the odes to describe the reaction
 
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    
    # This is to test the robustness - A to B, no C
    def rule_odes_2(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']#-m.P['k2']*m.Z[t,'B']
        #exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    times = (0.0, 10.0)

    # Set up the scenarios - this needs to be made easier for the user
    # Each case is made using simulated data with noise added and with
    # different initial conditions (hence the four experiments)
    # Note: the fourth experiment has only one reaction and one parameter
    
    models = {}
    
    # Declare all global variables used in simulation
    d_init = {'k1' : (2.5, (0.0, 5.0)),
              'k2' : (0.8, (0.0, 1.0)),
              }
    
    # Declare all global variables as the initial guess
    d_init_guess = {'k1' : (2.5, (0.0, 5.0)),
                    'k2' : (0.5, (0.0, 1.0)),
                   }


    # Scenario 1 #############################################################
    builder = TemplateBuilder()
    build = builder    
    components = {'A':1e-3,'B':0,'C':0}
    build.add_mixture_component(components)
    
    for d, dv in d_init.items():
        build.add_parameter(d, init=dv[0], bounds=dv[1])
        
    build.set_odes_rule(rule_odes)
    build.set_model_times(times)
    
    # Simulate data for scenario 1 
    pyomo_model = build.create_pyomo_model()
    simulator = PyomoSimulator(pyomo_model)
    Z_data, X_data, results = run_simulation(simulator)
    
    conc_measurement_index = [7, 57, 99]
    Z_data = results.Z.iloc[conc_measurement_index, :]
    #Z_data = add_noise_to_signal(Z_data, 1e-5)
    
    # This is in a working state, but shows that it works!
    #build.add_global_parameters({k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())})
    build._parameters_init = {k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())}
    scenario = (build, Z_data)
    models[0] = add_exp_data_and_make_model(scenario)
    
    # Scenario 2 #############################################################
    builder2 = TemplateBuilder()    
    build = builder2
    components = {'A':0.5e-3,'B':1e-4,'C':0}
    build.add_mixture_component(components)
    
    for d, dv in d_init.items():
        build.add_parameter(d, init=dv[0], bounds=dv[1])
   
    build.set_odes_rule(rule_odes)
    build.set_model_times(times)
    
    # Simulate data for scenario 2 
    pyomo_model2 = build.create_pyomo_model()
    simulator2 = PyomoSimulator(pyomo_model2)
    Z_data2, X_data2, results2 = run_simulation(simulator2)
    
    conc_measurement_index2 = [25, 28, 80]
    Z_data2 = results2.Z.iloc[conc_measurement_index2, :]
    #Z_data2 = add_noise_to_signal(Z_data2, 1e-5)
    #build.add_global_parameters({k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())})
    build._parameters_init = {k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())}
    scenario = (build, Z_data2)
    models[1] = add_exp_data_and_make_model(scenario)
    
    # Scenario 3 #############################################################
    builder3 = TemplateBuilder()   
    build = builder3
    components = {'A':1.5e-3,'B':4e-4,'C':1e-4}
    build.add_mixture_component(components)
    
    for d, dv in d_init.items():
        build.add_parameter(d, init=dv[0], bounds=dv[1])
    
    build.set_odes_rule(rule_odes)
    build.set_model_times(times)
    
    # Simulate data for scenario 3
    pyomo_model3 = build.create_pyomo_model()
    simulator3 = PyomoSimulator(pyomo_model3)
    Z_data3, X_data3, results3 = run_simulation(simulator3)
    
    conc_measurement_index3 = [12, 30, 50, 70, 82, 140]
    Z_data3 = results3.Z.iloc[conc_measurement_index3, :]
    #Z_data3 = add_noise_to_signal(Z_data3, 1e-5)
    
    #build.add_global_parameters({k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())})
    build._parameters_init = {k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())}
    scenario = (build, Z_data3)
    models[2] = add_exp_data_and_make_model(scenario)
    
    # Scenario 4 #############################################################
    builder4 = TemplateBuilder()   
    build = builder4
    components = {'A':1.5e-3,'B':4e-4}
    build.add_mixture_component(components)
    build.add_parameter('k1', init=d_init['k1'][0], bounds=d_init['k1'][1])
    build.set_odes_rule(rule_odes_2)
    build.set_model_times(times)
    
    # Simulate data for scenario 4
    pyomo_model4 = build.create_pyomo_model()
    simulator4 = PyomoSimulator(pyomo_model4)
    Z_data4, X_data4, results4 = run_simulation(simulator4)
    
    conc_measurement_index4 = [12, 30, 50, 70, 82, 140]
    Z_data4 = results4.Z.iloc[conc_measurement_index4, :]
    #Z_data4 = add_noise_to_signal(Z_data4, 1e-5)
    
    #build.add_global_parameters({k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())})
    build._parameters_init = {k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())}
    scenario = (build, Z_data4)
    models[3] = add_exp_data_and_make_model(scenario)
    
    # Perform the NSD parameter optimization
    
    # specify the parameter name in your model (required!)
    options = {'parameter_var_name': 'P',
               'method': 'newton', #'trust-constr',
         #      'objective_name' : 'name_of_objective_attribute', # I will find a cleaner method than this! (not tested)
               }
    
    nsd = NSD(models, d_init_guess, options)
    results, od = nsd.nested_schur_decomposition()
    
    print(f'\nThe final parameter values are:\n{nsd.parameters_opt}')
