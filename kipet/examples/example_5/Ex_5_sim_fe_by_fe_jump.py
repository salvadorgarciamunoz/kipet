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
    AH = r1.component('AH', value= 0.395555)
    B = r1.component('B', value= 0.0351202)
    C = r1.component('C', value= 0.0)
    BHp = r1.component('BHp', value= 0.0)
    Am = r1.component('Am', value= 0.0)
    ACm = r1.component('ACm', value= 0.0)
    P = r1.component('P', value= 0.0)
    
    y0 = r1.algebraic('y0', value=0, description='Reaction 0')
    y1 = r1.algebraic('y1', value=0, description='Reaction 1')
    y2 = r1.algebraic('y2', value=0, description='Reaction 2')
    y3 = r1.algebraic('y3', value=0, description='Reaction 3')
    y4 = r1.algebraic('y4', value=0, description='Reaction 4')

    k0 = r1.parameter('k0', value=49.7796)
    k1 = r1.parameter('k1', value=8.93156)
    k2 = r1.parameter('k2', value=1.31765)
    k3 = r1.parameter('k3', value=0.31087)
    k4 = r1.parameter('k4', value=3.87809)
    
    # add additional state variables
    V = r1.state('V', state='state', value=0.0629418)

    # stoichiometric coefficients
    stoich_coeff = dict()
    stoich_coeff['AH'] = [-1, 0, 0, -1, 0]
    stoich_coeff['B'] = [-1, 0, 0, 0, 1]
    stoich_coeff['C'] = [0, -1, 1, 0, 0]
    stoich_coeff['BHp'] = [1, 0, 0, 0, -1]
    stoich_coeff['Am'] = [1, -1, 1, 1, 0]
    stoich_coeff['ACm'] = [0, 1, -1, -1, -1]
    stoich_coeff['P'] = [0, 0, 0, 1, 1]
    
    V_step = r1.step('V_step', time=210, fixed=True, switch='off')   
 
    V_flow = r1.constant('V_flow', value=7.27609e-5)   
 
    # Algebraics (written as y0 = k0*AH*B)
    AE = {}
    AE['y0'] = k0*AH*B
    AE['y1'] = k1*Am*C
    AE['y2'] = k2*ACm
    AE['y3'] = k3*ACm*AH
    AE['y4'] = k4*ACm*BHp
    
    r1.add_algebraics(AE)
    
    # Rate Equations
    reaction_vars = r1.algebraics.names
    
    RE = r1.reaction_block(stoich_coeff, reaction_vars)
    RE['C'] += 0.02247311828 / (V * 210) * V_step
    RE['V'] = V_flow * V_step

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