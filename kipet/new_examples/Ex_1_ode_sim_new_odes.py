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
    r1.add_parameter('k1', 2)
    r1.add_parameter('k2', 0.2)
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=1)
    r1.add_component('B', state='concentration', init=0.0)
    r1.add_component('C', state='concentration', init=0.0)
    
    
    # New way of writing ODEs - only after declaring components, algebraics,
    # and parameters
    c = r1.get_model_vars()
    
    # globals()['A'] = c.A
    
    # c now holds of all of the pyomo variables needed to define the equations
    rates = {}
    rates['A'] = -c.k1 * c.A
    rates['B'] = c.k1 * c.A - c.k2 * c.B
    rates['C'] = c.k2 * c.B
    
    r1.add_odes(rates)

    # Add dosing points 
    r1.add_dosing_point('A', 3, 0.3)
    
    # Create the model - simulations require times
    r1.set_times(0, 10)
    r1.create_pyomo_model()
    
    # print(r1.model.odes.display())
    
    r1.settings.collocation.ncp = 3
    r1.settings.collocation.nfe = 50
    # r1.settings.simulator.method = 'fe'
    
    #Simulate with default options
    r1.simulate()
    
    if with_plots:
        r1.results.plot()