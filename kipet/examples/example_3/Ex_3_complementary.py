"""Example 3: Simulation using complementatry states with new KipetModel"""

# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports
from pyomo.core import exp

# Kipet library imports
from kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False

    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('reaction-1')
    
    # Declare the components and give the initial values
    A = r1.component('A', value=1.0)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0)
    
    T = r1.state('T', value=290, description='Temperature')
    V = r1.state('V', value=100, description='Volumne')

    # Define the ODEs
    k1 = 1.25*exp((9500/1.987)*(1/320.0 - 1/T))
    k2 = 0.08*exp((7000/1.987)*(1/290.0 - 1/T))
    ra = -k1*A
    rb = 0.5*k1*A - k2*B
    rc = 3*k2*B
    cao = 4.0
    vo = 240
    T1 = 35000*(298 - T)
    T2 = 4*240*30.0*(T-305.0)
    T3 = V*(6500.0*k1*A - 8000.0*k2*B)
    Den = (30*A + 60*B + 20*C)*V + 3500.0
    
    r1.add_ode('A', ra + (cao - A)/V )
    r1.add_ode('B', rb - B*vo/V )
    r1.add_ode('C', rc - C*vo/V )
    r1.add_ode('T', (T1 + T2 + T3)/Den )
    r1.add_ode('V', vo )
    
    r1.set_times(0.0, 2.0)
    
    r1.settings.collocation.nfe = 20
    r1.settings.collocation.ncp = 1

    r1.simulate()  

    if with_plots:
        r1.plot()