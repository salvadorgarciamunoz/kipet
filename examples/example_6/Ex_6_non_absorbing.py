"""Example 6: Estimation with non-absorbing species with new KipetModel"""

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
       
    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('reaction-1')
    
    # Add the model parameters
    k1 = r1.parameter('k1', value=2, bounds=(0.0, 5.0))
    k2 = r1.parameter('k2', value=0.2, bounds=(0.0, 2.0))
    
    # Declare the components and give the initial values
    A = r1.component('A', value=7.5e-5)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0, absorbing=False)
    
    # Use this function to replace the old filename set-up
    r1.add_data(category='spectral', file='data/Dij.txt')
    
    # define explicit system of ODEs
    rA = k1*A
    rB = k2*B
    
    r1.add_ode('A', -rA)
    r1.add_ode('B', -rA + rB)
    r1.add_ode('C', rB)

    # Settings
    r1.settings.collocation.ncp = 1
    r1.settings.collocation.nfe = 60
    r1.settings.variance_estimator.use_subset_lambdas = True
    r1.settings.variance_estimator.max_iter = 5
    r1.settings.variance_estimator.tolerance = 1e-4
    r1.settings.parameter_estimator.tee = False
    
    r1.run_opt()
    
    # Display the results
    r1.results.show_parameters
    
        # New plotting methods
    if with_plots:
        r1.plot()
