"""
 
Kipet: Kinetic parameter estimation toolkit
Copyright (c) 2016 Eli Lilly.
 
Example from Chen and Biegler, Reduced Hessian Based Parameter Selection and
    Estimation with Simultaneous Collocation Approach (AIChE 2020) paper with
    the simplified Dow Problem.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from kipet.library.data_tools import (
    add_noise_to_signal,
    read_concentration_data,
    )

from kipet.library.EstimationPotential import EstimationPotential
from kipet.library.FESimulator import FESimulator
from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.TemplateBuilder import (
        TemplateBuilder,
        Component,
        KineticParameter,
        )

def run_simulation(simulator):

    options = {'solver_opts' : dict(linear_solver='ma57')}

    results_pyomo = simulator.run_sim('ipopt',
                                      tee=True,
                                      solver_options=options,
                                      )

    if with_plots:
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (h)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        
        plt.show()
        
    Z_data = pd.DataFrame(results_pyomo.Z)
    X_data = pd.DataFrame(results_pyomo.X)
    return Z_data, X_data, results_pyomo


if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================
    
    # Create Template model (Section 4.2.1 of documentation)
    builder = TemplateBuilder()  
    
    components = [
            Component('y1', 1.6497, 4.09e-4),
            Component('y2', 8.2262, 1.04e-3),
            Component('y3', 0.0, 4.69e-4),
            Component('y4', 0, 3.28e-4),
            ]
        
    kinetic_parameters = [
            KineticParameter('k1',  (0.1, 5), 1.8934, 0.09),
            KineticParameter('k2',  (0.0, 5), 2.7585, 0.01),
            KineticParameter('k-1', (0.0, 3000), 1754.0, 0.01),
            KineticParameter('B1',  (0.0, 2), 6.1894e-3, 0.01),
            KineticParameter('B2',  (0.0, 2), 0.0048, 0.01),
            ]
    
    # kinetic_parameters = [
    #         KineticParameter('k1',  (0.1, 2), 1.726, 0.09),
    #         KineticParameter('k2',  (0.0, 5), 2.312, 0.01),
    #         KineticParameter('k-1', (0.0, 300), 240.5, 0.01),
    #         KineticParameter('B1',  (0.0, 2), 0.006, 0.01),
    #         KineticParameter('B2',  (0.0, 2), 0.004, 0.01),
    #         ]
    
    # Prepare components
    for com in components:
        builder.add_mixture_component(com.name, com.init)
    
    # Prepare parameters
    for param in kinetic_parameters:
        builder.add_parameter(param.name, param.init)

    # define explicit system of ODEs
    def rule_odes(m,t):
        
        #x1 = 340.15
        x2 = 0.0131
        x3 = 1.6497
        x4 = 8.2262
        
        S = (-2*x3 + x4 + 2*m.Z[t,'y1'] - m.Z[t,'y2'] + m.Z[t,'y3'])/(m.Z[t,'y1'] + m.P['B1']*(-x3 + x4 + m.Z[t,'y1'] - m.Z[t,'y2']) + m.P['B2']*m.Z[t,'y3'])
        term1 = (x2 + 2*x3 - x4 -2*m.Z[t,'y1'] + m.Z[t,'y2'] - m.Z[t,'y3'])
        
        r1 = m.P['k2']*m.Z[t,'y1']*m.Z[t,'y2']*S
        r2 = -m.P['k1']*m.Z[t,'y2']*term1 - r1 + m.P['k-1']*m.P['B1']*(-x3 + x4 + m.Z[t,'y1'] - m.Z[t,'y2'])*S 
        r3 = m.P['k1']*(x3 - m.Z[t,'y1'] - m.Z[t,'y3'])*term1 + r1 - 0.5*m.P['k-1']*m.P['B2']*m.Z[t,'y3']*S
        r4 = r1 - r3
        
        exprs = dict()
        exprs['y1'] = -r1
        exprs['y2'] = r2
        exprs['y3'] = r3      
        exprs['y4'] = r4 
        return exprs

    #Add these ODEs to our model template
    builder.set_odes_rule(rule_odes)

    # create an instance of a pyomo model template
    # the template includes
    #      - Z variables indexed over time and components names e.g. m.Z[t,'A']
    #      - P parameters indexed over the parameter names e.g. m.P['k']
    # The arguments here are the start and end time of the simulation
    #pyomo_model = builder.create_pyomo_model(0.0, 10.0)

    #=========================================================================
    # USER INPUT SECTION - SPECIFIC USE SECTION
    #=========================================================================
    # Since in this example we wish to simulate the reaction system defined above,
    # we call the PyomoSimulator class as described in Section 4.2.2 of the documentation

    
    # create instance of simulator with the created pyomo_model as input
    times = (0.0, 300.0)
    builder.set_model_times(times)
    pyomo_model = builder.create_pyomo_model()
    simulator = FESimulator(pyomo_model)
    
    simulator.apply_discretization('dae.collocation',
                                   ncp = 5,
                                   nfe = 2000,
                                   scheme = 'LAGRANGE-RADAU')
    
    simulator.call_fe_factory()
    
    Z_data, X_data, results = run_simulation(simulator)

    #Make some data points
    # ncp = 3
    # nfe = 20
    # measurements = 11
    
    # t = np.linspace(0, ncp*nfe, measurements).astype(int)
    # Z_data = Z_data.iloc[t, :]
    # Z_data.drop(index=0, inplace=True)
    
    # Use the experimental data from the paper: too stiff at the moment - need a good initialization
        
    data_directory = 'data_sets'
    data_filename = 'dow_data.csv'
    dataDirectory = Path(Path(__file__).resolve().parent, data_directory)
    filename = dataDirectory.joinpath(data_filename)
    Z_data = read_concentration_data(filename)
    
    factor = 1.2
    builder_est = TemplateBuilder() 
    builder_est.set_odes_rule(rule_odes)
    
    noise = {c.name : c.sigma*1 for c in components}

    for c in components:
        Z_data[c.name] = add_noise_to_signal(Z_data[c.name], np.sqrt(noise[c.name]))
    
    builder_est.add_state_variance({k : np.sqrt(v) for k, v in noise.items()})

    for com in components:
        builder_est.add_mixture_component(com.name, com.init)
    
    for param in kinetic_parameters:
        builder_est.add_parameter(param.name, bounds=param.bounds, init=param.init*factor)
   
    # New method to add state variance to the model builder
    builder_est.add_state_variance(noise)
    # New method to add times like everything else
    builder_est.set_model_times(times)
    
    est_param = EstimationPotential(builder_est, Z_data, simulation_data=results, verbose=True)
    est_param.estimate()
    est_param.plot_results()