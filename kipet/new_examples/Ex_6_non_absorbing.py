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
    r1.add_parameter('k1', init=2, bounds=(0.1, 5.0))
    r1.add_parameter('k2', init=0.2, bounds=(0.01, 2.0))
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=1e-3)
    r1.add_component('B', state='concentration', init=0.0)
    r1.add_component('C', state='concentration', init=0.0)
    
    # Use this function to replace the old filename set-up
    filename = r1.set_directory('Dij.txt')
    r1.add_dataset('D_frame', category='spectral', file=filename)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    r1.add_equations(rule_odes)
    
    # If no times are given to the builder, it will use the times in the data
    r1.set_non_absorbing_species(['C'])

    # Settings
    r1.settings.collocation.ncp = 1
    r1.settings.collocation.nfe = 60
    r1.settings.variance_estimator.use_subset_lambdas = True
    r1.settings.variance_estimator.max_iter = 5
    r1.settings.variance_estimator.tolerance = 1e-4
    r1.settings.parameter_estimator.tee = False
    
    # This is all you need to run KIPET!
    r1.run_opt()
    
    # Display the results
    r1.results.show_parameters
    
       # New plotting methods
    if with_plots:
        r1.results.plot()