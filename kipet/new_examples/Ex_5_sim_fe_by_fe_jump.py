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
    components['BH+'] = 0.0
    components['A-'] = 0.0
    components['AC-'] = 0.0
    components['P'] = 0.0

    for comp, init_value in components.items():
        r1.add_component(comp, state='concentration', init=init_value)

    # add algebraics
    algebraics = [0, 1, 2, 3, 4, 5]  # the indices of the rate rxns
    # note the fifth component. Which basically works as an input

    r1.add_algebraic_variables(algebraics)

    params = dict()
    params['k0'] = 49.7796
    params['k1'] = 8.93156
    params['k2'] = 1.31765
    params['k3'] = 0.310870
    params['k4'] = 3.87809

    for param, init_value in params.items():
        r1.add_parameter(param, init=init_value)

    # add additional state variables
    r1.add_component('V', state='state', init=0.0629418)

    # stoichiometric coefficients
    gammas = dict()
    gammas['AH'] = [-1, 0, 0, -1, 0]
    gammas['B'] = [-1, 0, 0, 0, 1]
    gammas['C'] = [0, -1, 1, 0, 0]
    gammas['BH+'] = [1, 0, 0, 0, -1]
    gammas['A-'] = [1, -1, 1, 1, 0]
    gammas['AC-'] = [0, 1, -1, -1, -1]
    gammas['P'] = [0, 0, 0, 1, 1]

    def rule_algebraics(m, t):
        r = list()
        r.append(m.Y[t, 0] - m.P['k0'] * m.Z[t, 'AH'] * m.Z[t, 'B'])
        r.append(m.Y[t, 1] - m.P['k1'] * m.Z[t, 'A-'] * m.Z[t, 'C'])
        r.append(m.Y[t, 2] - m.P['k2'] * m.Z[t, 'AC-'])
        r.append(m.Y[t, 3] - m.P['k3'] * m.Z[t, 'AC-'] * m.Z[t, 'AH'])
        r.append(m.Y[t, 4] - m.P['k4'] * m.Z[t, 'AC-'] * m.Z[t, 'BH+'])
        return r
    #: there is no AE for Y[t,5] because step equn under rule_odes functions as the switch for the "C" equation

    r1.add_algebraics(rule_algebraics)
 
    def rule_odes(m, t):
        exprs = dict()
        eta = 1e-2
        step = 0.5 * ((m.Y[t, 5] + 1) / ((m.Y[t, 5] + 1) ** 2 + eta ** 2) ** 0.5 + (210.0 - m.Y[t,5]) / ((210.0 - m.Y[t, 5]) ** 2 + eta ** 2) ** 0.5)
        exprs['V'] = 7.27609e-05 * step
        V = m.X[t, 'V']
        
        # mass balances
        for c in m.mixture_components:
            exprs[c] = sum(gammas[c][j] * m.Y[t, j] for j in m.algebraics if j != 5) - exprs['V'] / V * m.Z[t, c]
            if c == 'C':
                exprs[c] += 0.02247311828 / (m.X[t, 'V'] * 210) * step
        return exprs

    r1.add_equations(rule_odes)
    
    # Declare dosing algebraic
    r1.set_dosing_var(5)
    # Add dosing points 
    r1.add_dosing_point('AH', 100, 0.3)
   # r1.add_dosing_point('A-', 300, 0.9)

    r1.set_times(0, 600)
      
    r1.simulate()
    
    if with_plots:
        r1.results.plot('Z')
        r1.results.plot('X')
        r1.results.plot('Y')