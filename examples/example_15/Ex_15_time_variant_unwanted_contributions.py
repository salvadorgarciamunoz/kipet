"""Example 15: Time variant unwanted contributions with the new KipetModel
 
"""
# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports

# Kipet library imports
from kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
    # Define the general model
    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('reaction-1')
    
    # Add the model parameters
    k1 = r1.parameter('k1', value=1.4, bounds=(0.0, 2.0))
    k2 = r1.parameter('k2', value=0.25, bounds=(0.0, 0.5))
    
    # Declare the components and give the initial values
    A = r1.component('A', value=1.0e-2)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0)
    
    # define explicit system of ODEs
    rates = {}
    rates['A'] = -k1 * A
    rates['B'] = k1 * A - k2 * B
    rates['C'] = k2 * B
  
    r1.add_odes(rates)
   
    # Add the data
    r1.add_data(category='spectral', file='data/Dij_tv_G.txt')
    
    # Settings
    r1.settings.general.initialize_pe = False
    r1.settings.general.no_user_scaling = True
    r1.settings.collocation.nfe = 100
    # r1.settings.parameter_estimator.G_contribution = 'time_variant_G'
    
    r1.unwanted_contribution('time_variant_G')

    # Run KIPET
    r1.run_opt()

    if with_plots:
        r1.plot()
