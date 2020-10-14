"""Example 5: Simulation with FESimulator with new KipetModel

"""
# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports
import pandas as pd
from pyomo.environ import exp

# Kipet library imports
from kipet.kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
                    
    # Declare the model:
    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('simulation')
    
    # When the category is trajectory, it is used for initializations, scaling,
    # or fixing variables. It is not added to the pyomo model
    filename = 'Tempvalues.csv'
    
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
        
    # add additional state variables
    r1.add_component('V', state='state', init=0.0629418)
    # Add components that are to be fixed to a trajectory here as well
    r1.add_component('Temp', state='trajectory')

    r1.add_dataset(file=filename)  

    # add algebraics
    algebraics = ['0', '1', '2', '3', '4', '5', 'k4T', 'Temp']  # the indices of the rate rxns
    # note the fifth, sixth and seventh components. Which basically work as inputs

    r1.add_algebraic_variables(algebraics)
    
    params = dict()
    params['k0'] = 49.7796
    params['k1'] = 8.93156
    params['k2'] = 1.31765
    params['k3'] = 0.310870
    params['k4Tr'] = 3.87809
    params['E'] = 20.

    for param, init_value in params.items():
        r1.add_parameter(param, init=init_value)


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
        r.append(m.Y[t, '0'] - m.P['k0'] * m.Z[t, 'AH'] * m.Z[t, 'B'])
        r.append(m.Y[t, '1'] - m.P['k1'] * m.Z[t, 'A-'] * m.Z[t, 'C'])
        r.append(m.Y[t, '2'] - m.P['k2'] * m.Z[t, 'AC-'])
        r.append(m.Y[t, '3'] - m.P['k3'] * m.Z[t, 'AC-'] * m.Z[t, 'AH'])
        r.append(m.Y[t, '4'] - m.Y[t, 'k4T'] * m.Z[t, 'AC-'] * m.Z[t, 'BH+'])#6 is k4T
        Tr=303.15
        #add temperature dependence via Arrhenius law
        r.append(m.Y[t, 'k4T'] - m.P['k4Tr']*exp(m.P['E'] (1 / m.Y[t, 'Temp'] - 1 / Tr)))
        return r
    #: there is no ae for Y[t,5] because step equn under rule_odes functions as the switch for the "C" equation

    r1.add_algebraics(rule_algebraics)

    def rule_odes(m, t):
        exprs = dict()
        eta = 1e-2
        #: @dthierry: This thingy is now framed in terms of `m.Y[t, 5]`
        step = 0.5 * ((m.Y[t, '5'] + 1) / ((m.Y[t, '5'] + 1) ** 2 + eta ** 2) ** 0.5 + (210.0 - m.Y[t,'5']) / ((210.0 - m.Y[t, '5']) ** 2 + eta ** 2) ** 0.5)
        exprs['V'] = 7.27609e-05 * step
        V = m.X[t, 'V']
        # mass balances
        for c in m.mixture_components:
            exprs[c] = gammas[c][0]*m.Y[t, '0'] + gammas[c][1]*m.Y[t, '1'] + gammas[c][2]*m.Y[t, '2'] + gammas[c][3]*m.Y[t, '3'] + gammas[c][4]*m.Y[t,'4'] - exprs['V']/V*m.Z[t, c]
            if c == 'C':
                exprs[c] += 0.02247311828 / (m.X[t, 'V'] * 210) * step
        return exprs

    r1.add_equations(rule_odes)
    
    # Declare dosing algebraic
    r1.set_dosing_var('5')
    # Add dosing points or other jumps
    r1.add_dosing_point('AH', 101.035, 0.3)
    r1.add_dosing_point('A-', 400, 0.1)
    r1.add_dosing_point('V', 303.126, 0.01)

    # Fix certain variables to given data
    r1.fix_from_trajectory('Y',  'Temp', 'Traj')    

    # Make the model
    r1.set_times(0, 600)
    
    # Run the simulation
    r1.simulate()
    
    # Plot the results
    if with_plots:
        r1.results.plot('Z', show_exp=False)
        r1.results.plot('X')
