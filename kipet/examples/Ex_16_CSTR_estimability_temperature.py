"""
Kipet: Kinetic parameter estimation toolkit
Copyright (c) 2016 Eli Lilly.
 
Example from Chen and Biegler, Reduced Hessian Based Parameter Selection and
    Estimation with Simultaneous Collocation Approach (AIChE 2020) paper with
    a CSTR for a simple reaction.
    
    This example uses reactor temperature as the known output data.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyomo.core as pyomo

from kipet.library.data_tools import add_noise_to_signal
from kipet.library.EstimationPotential import EstimationPotential
from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.TemplateBuilder import (
        TemplateBuilder,
        Component,
        KineticParameter,
        )

def run_simulation(simulator, nfe=50, ncp=3, use_only_FE=True):
    """This is not necessary, but is used for generating data used in the
    estimation algorithm
    """
    simulator.apply_discretization('dae.collocation',
                                   ncp = ncp,
                                   nfe = nfe,
                                   scheme = 'LAGRANGE-RADAU')

    options = {'solver_opts' : dict(linear_solver='ma57')}
    
    results_pyomo = simulator.run_sim('ipopt_sens',
                                      tee=False,
                                      solver_options=options,
                                      )

    if with_plots:
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (h)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        
        results_pyomo.X.plot.line(legend=True)
        plt.xlabel("time (h)")
        plt.ylabel("Temperature (K)")       
        plt.title("Temperature Profiles")
        
        plt.show()
    
    Z_data = pd.DataFrame(results_pyomo.Z)
    X_data = pd.DataFrame(results_pyomo.X)
    
    if use_only_FE:
        
        t = np.linspace(0, ncp*nfe, nfe+1).astype(int)
        
        Z_data = Z_data.iloc[t, :]
        X_data = X_data.iloc[t, :]
        
        Z_data.drop(index=0, inplace=True)
        X_data.drop(index=0, inplace=True)
        
    return Z_data, X_data, results_pyomo


if __name__ == "__main__":

    with_plots = False
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    builder = TemplateBuilder()  
    
    components = [
            Component('A', 1000, 0.1),
            ]
        
    parameters = [
            KineticParameter('Tf',   (250, 400), 293.15, 0.09),
            KineticParameter('Cfa',  (0, 5000), 2500, 0.01),
            KineticParameter('rho',  (800, 2000), 1025, 0.01),
            KineticParameter('delH', (0.0, 400), 160, 0.01),
            KineticParameter('ER',   (0.0, 500), 255, 0.01),
            KineticParameter('k',    (0.0, 10), 2.5, 0.01),
            KineticParameter('Tfc',  (250, 400), 283.15, 0.01),
            KineticParameter('rhoc', (0.0, 2000), 1000, 0.01),
            KineticParameter('h',    (0.0, 5000), 3600, 0.01),
            ]
    
    constants = {
            'F' : 0.1, # m^3/h
            'Fc' : 0.15, # m^3/h
            'Ca0' : 1000, # mol/m^3
            'V' : 0.2, # m^3
            'Vc' : 0.055, # m^3
            'A' : 4.5, # m^2
            'Cpc' : 1.2, # kJ/kg/K
            'Cp' : 1.55, #6 kJ/kg/K
            }
    
    # Make it easier to use the constants
    C = constants
    
    builder.add_complementary_state_variable('T',  293.15)
    builder.add_complementary_state_variable('Tc', 293.15)

    # Prepare components
    for com in components:
        builder.add_mixture_component(com.name, com.init)
    
    factor = 1.0
    
    # Prepare parameters
    for param in parameters:
        builder.add_parameter(param.name, param.init*factor)
      
    def rule_odes(m,t):
        
        Ra = m.P['k']*pyomo.exp(-m.P['ER']/m.X[t,'T'])*m.Z[t,'A']
        exprs = dict()
        exprs['A'] = C['F']/C['V']*(m.P['Cfa']-m.Z[t,'A']) - Ra
        exprs['T'] = C['F']/C['V']*(m.P['Tf']-m.X[t,'T']) + m.P['delH']/(m.P['rho'])/C['Cp']*Ra - m.P['h']*C['A']/(m.P['rho'])/C['Cp']/C['V']*(m.X[t,'T'] - m.X[t,'Tc'])
        exprs['Tc'] = C['Fc']/C['Vc']*(m.P['Tfc']-m.X[t,'Tc']) + m.P['h']*C['A']/(m.P['rhoc'])/C['Cpc']/C['Vc']*(m.X[t,'T'] - m.X[t,'Tc'])
        return exprs

    builder.set_odes_rule(rule_odes)
    times = (0.0, 5.0)
    builder.set_model_times(times)
    pyomo_model = builder.create_pyomo_model()
    simulator = PyomoSimulator(pyomo_model)
    Z_data, X_data, results = run_simulation(simulator)

    # End of the simulation stages - this is not necessary when using actual data

    noise = {
            'T' : 0.25,
            }

    X_data['T'] = add_noise_to_signal(X_data['T'], noise['T'])
    exp_data = pd.DataFrame(X_data['T'])

    # This is the 20% increase from the original parameter values - not needed in real cases
    factor = 1.2
    
    builder_est = TemplateBuilder() 
    
    builder_est.add_complementary_state_variable('T',  293.15)
    builder_est.add_complementary_state_variable('Tc', 293.15)
    
    builder_est.set_odes_rule(rule_odes)

    for com in components:
        builder_est.add_mixture_component(com.name, com.init)
    
    for param in parameters:
        builder_est.add_parameter(param.name, bounds=param.bounds, init=param.init*factor)
   
    # New method to add state variance to the model builder
    builder_est.add_state_variance(noise)
    
    # New method to add times like everything else
    builder_est.set_model_times(times)
    
    est_param = EstimationPotential(builder_est, exp_data, simulation_data=results, verbose=True)
    est_param.estimate()
    est_param.plot_results()