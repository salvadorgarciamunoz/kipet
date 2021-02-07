"""Example 5: Simulation with FESimulator with new KipetModel

"""
# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports
import pandas as pd
from pyomo.environ import exp

# Kipet library imports
from kipet import KipetModel


if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
     
    # This holds the model
    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('simulation')
    
    # components
    components = dict()
    components['AH'] = 0.395555
    components['B'] = 0.0351202
    components['C'] = 0.0
    components['BHp'] = 0.0
    components['Am'] = 0.0
    components['ACm'] = 0.0
    components['P'] = 0.0

    for comp, init_value in components.items():
        r1.add_component(comp, value=init_value)
    
    r1.add_alg_var('y0', value=0, description='Reaction 0')
    r1.add_alg_var('y1', value=0, description='Reaction 1')
    r1.add_alg_var('y2', value=0, description='Reaction 2')
    r1.add_alg_var('y3', value=0, description='Reaction 3')
    r1.add_alg_var('y4', value=0, description='Reaction 4')

    params = dict()
    params['k0'] = 49.7796
    params['k1'] = 8.93156
    params['k2'] = 1.31765
    params['k3'] = 0.310870
    params['k4'] = 3.87809

    for param, init_value in params.items():
        r1.add_parameter(param, value=init_value)

    # add additional state variables
    r1.add_state('V', state='state', value=0.0629418)

    # stoichiometric coefficients
    stoich_coeff = dict()
    stoich_coeff['AH'] = [-1, 0, 0, -1, 0]
    stoich_coeff['B'] = [-1, 0, 0, 0, 1]
    stoich_coeff['C'] = [0, -1, 1, 0, 0]
    stoich_coeff['BHp'] = [1, 0, 0, 0, -1]
    stoich_coeff['Am'] = [1, -1, 1, 1, 0]
    stoich_coeff['ACm'] = [0, 1, -1, -1, -1]
    stoich_coeff['P'] = [0, 0, 0, 1, 1]
    
    r1.add_step('V_step', time=210, fixed=True, switch='off')   
 
    r1.add_constant('V_flow', value=7.27609e-5)   
 
    # Get the model variables
    c = r1.get_model_vars()
    
    # Algebraics (written as y0 = k0*AH*B)
    AE = {}
    AE['y0'] = c.k0*c.AH*c.B
    AE['y1'] = c.k1*c.Am*c.C
    AE['y2'] = c.k2*c.ACm
    AE['y3'] = c.k3*c.ACm*c.AH
    AE['y4'] = c.k4*c.ACm*c.BHp
    
    r1.add_algebraics(AE)
    
    # Rate Equations
    reaction_vars = r1.algebraics.names
    
    RE = r1.reaction_block(stoich_coeff, reaction_vars)
    RE['C'] += 0.02247311828 / (c.V * 210) * c.V_step
    RE['V'] = c.V_flow * c.V_step

    r1.add_odes(RE)
    
    r1.check_model_units()
    # Add dosing points 
    r1.add_dosing_point('AH', 100, 0.3)
    r1.add_dosing_point('Am', 300, 0.9)

    r1.set_times(0, 600)
    r1.settings.collocation.nfe = 50
    r1.settings.simulator.method = 'fe'
    
    r1.simulate()
    
    if with_plots:
        r1.plot()