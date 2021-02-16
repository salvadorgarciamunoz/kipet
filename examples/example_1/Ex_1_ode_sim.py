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
    k1 = r1.parameter('k1', value=2, units='1/s')
    k2 = r1.parameter('k2', value=0.2, units='1/s')
    
    # Declare the components and give the initial values
    A = r1.component('A', value=1.0, units='M')
    B = r1.component('B', value=0.0, units='M')
    C = r1.component('C', value=0.0, units='M')
    
    rA = r1.add_expression('rA', k1*A)
    rB = r1.add_expression('rB', k2*B)
    
    r1.add_ode('A', -rA )
    r1.add_ode('B', rA - rB )
    r1.add_ode('C', rB )

    # # Option to check the units of your models
    r1.check_model_units()
    # # Add dosing points 
    #r1.add_dosing_point('A', 3, 0.3)
    
    # # Create the model - simulations require times
    r1.set_times(0, 10)
    # r1.create_pyomo_model()
    
    r1.settings.collocation.ncp = 3
    r1.settings.collocation.nfe = 50

    #Simulate with default options
    r1.simulate()
    
    if with_plots:
        r1.plot()