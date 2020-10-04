"""Example 13: Alternate method for variance estimation with new KipetModel

"""

# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports

# Kipet library imports
from kipet.kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
   
    kipet_model = KipetModel()
    
    # Use this function to replace the old filename set-up
    filename = kipet_model.set_directory('varest.csv')
    kipet_model.add_dataset('D_frame', category='spectral', file=filename, remove_negatives=True)

    # Add the model parameters
    kipet_model.add_parameter('k1', init=1.2, bounds=(0.5, 5.0))
    kipet_model.add_parameter('k2', init=0.2, bounds=(0.005, 5.0))
    
    # Declare the components and give the initial values
    kipet_model.add_component('A', state='concentration', init=1e-2)
    kipet_model.add_component('B', state='concentration', init=0.0)
    kipet_model.add_component('C', state='concentration', init=0.0)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs

    kipet_model.add_equations(rule_odes)
    kipet_model.bound_profile(var='S', bounds=(0, 100))

    # Settings
    kipet_model.settings.general.no_user_scaling = True
    kipet_model.settings.variance_estimator.tolerance = 1e-10
    kipet_model.settings.parameter_estimator.tee = False
    kipet_model.settings.parameter_estimator.solver = 'ipopt_sens'
    
    # Additional settings for the alternate method
    kipet_model.settings.variance_estimator.method = 'alternate'
    kipet_model.settings.variance_estimator.secant_point = 5e-4
    kipet_model.settings.variance_estimator.initial_sigmas = 5e-5
    
    # This is all you need to run KIPET!
    kipet_model.run_opt()
    
    # Display the results
    kipet_model.results.show_parameters
    
       # New plotting methods
    if with_plots:
        kipet_model.results.plot()
