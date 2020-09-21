"""Example 1: ODE Simulation with new KipetModel"""

# Standard library imports
import sys

# Third party imports

# Kipet library imports
from kipet.kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    kipet_model = KipetModel()
    
    # Add the model parameters
    kipet_model.add_parameter('k1', 2)
    kipet_model.add_parameter('k2', 0.2)
    
    # Declare the components and give the initial values
    kipet_model.add_component('A', state='concentration', init=1)
    kipet_model.add_component('B', state='concentration', init=0.0)
    kipet_model.add_component('C', state='concentration', init=0.0)
    
    # Define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs

    # Add the rules to the model
    kipet_model.add_equations(rule_odes)
    
    # Create the model - simulations require times
    kipet_model.create_pyomo_model(0, 10)
    
    # Simulate with default options
    kipet_model.simulate()
    
    if with_plots:
        kipet_model.results.plot()