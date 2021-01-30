"""Example 1: ODE Simulation with new KipetModel"""

# Standard library imports
import sys

# Third party imports

# Kipet library imports
from kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('reaction-1')
    
    # Add the model parameters
    r1.add_parameter('k1', 2, units='1/min')
    r1.add_parameter('k2', 0.2, units='1/min')
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=1, units='mol/L')
    r1.add_component('B', state='concentration', init=0.0, units='mol/L')
    r1.add_component('C', state='concentration', init=0.0, units='mol/L')
    
    # Define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs

    # Add the rules to the model
    r1.add_odes(rule_odes)
  
    # Add dosing points 
    #r1.add_dosing_point('A', 3, 0.3)
    
    # Create the model - simulations require times
    r1.set_times(0, 10)
    
    # Simulate with default options
    r1.simulate()
    
    if with_plots:
        r1.results.plot()