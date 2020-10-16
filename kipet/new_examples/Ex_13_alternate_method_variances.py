"""Example 13: Alternate method for variance estimation with new KipetModel

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
   
    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('reaction-1')

    # Add the model parameters
    r1.add_parameter('k1', init=1.2, bounds=(0.5, 5.0))
    r1.add_parameter('k2', init=0.2, bounds=(0.005, 5.0))
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=1e-2)
    r1.add_component('B', state='concentration', init=0.0)
    r1.add_component('C', state='concentration', init=0.0)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs

    r1.add_equations(rule_odes)
    r1.bound_profile(var='S', bounds=(0, 100))
    
    # Add data (after components)
    r1.add_dataset(category='spectral', file='varest.csv', remove_negatives=True)

    # Settings
    r1.settings.general.no_user_scaling = True
    r1.settings.variance_estimator.tolerance = 1e-10
    r1.settings.parameter_estimator.tee = False
    r1.settings.parameter_estimator.solver = 'ipopt_sens'
    
    # Additional settings for the alternate method
    r1.settings.variance_estimator.method = 'alternate'
    r1.settings.variance_estimator.secant_point = 5e-4
    r1.settings.variance_estimator.initial_sigmas = 5e-5
    
    # This is all you need to run KIPET!
    r1.run_opt()
    
    # Display the results
    r1.results.show_parameters
    
       # New plotting methods
    if with_plots:
        r1.results.plot()
