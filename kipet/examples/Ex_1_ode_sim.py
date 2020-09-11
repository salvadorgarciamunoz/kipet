"""Example 1: ODE Simulation with new KipetModel"""

# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports
import numpy as np

# Kipet library imports
from kipet.kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================
    
    kipet_model = KipetModel(intent='simulation')
    
    # Add the model parameters
    kipet_model.add_parameter('k1', 2)
    kipet_model.add_parameter('k2', 0.2)
    
    # Declare the components and give the initial values
    kipet_model.add_component('A', state='concentration', init=1)
    kipet_model.add_component('B', state='concentration', init=0.0)
    kipet_model.add_component('C', state='concentration', init=0.0)
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs

    # Add the rules to the model
    kipet_model.add_equations(rule_odes)
    
    # create the model - simulations require times
    kipet_model.create_pyomo_model(0, 10)
    
    # simulate with default options
    kipet_model.simulate()
    
    if with_plots:
        # Plot the results using results['sim'] and simulation=True
        kipet_model.results['sim'].plot(simulation=True)