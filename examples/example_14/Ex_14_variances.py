"""Example 14: Estimation with new KipetModel

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
    
    A = r1.component('A', value=3e-2)
    B = r1.component('B', value=4e-2)
    C = r1.component('C', value=0.0)
    D = r1.component('D', value=2e-2)
    E = r1.component('E', value=0.0)
    F = r1.component('F', value=0.0)

    filename = 'data/varest3.csv'
    r1.add_data(category='spectral', file=filename, remove_negatives=True)
    full_data = kipet_model.read_data_file(filename)
    
    # r1.spectra.decrease_wavelengths(2)

    k1 = r1.parameter('k1', value=1.5, bounds=(0.5, 2.0)) 
    k2 = r1.parameter('k2', value=28.0, bounds=(1, 30))
    k3 = r1.parameter('k3', value=0.3, bounds=(0.001, 0.5))

    r1.add_ode('A', -k1*A )
    r1.add_ode('B', -k1*A - k3*B )
    r1.add_ode('C', -k2*C*D + k1*A )
    r1.add_ode('D', -2*k2*C*D )
    r1.add_ode('E', k2*C*D )
    r1.add_ode('F', k3*B )
    
    # r1.add_equations(rule_odes)
    r1.bound_profile(var='S', bounds=(0, 20))

    # Settings
    r1.settings.general.no_user_scaling = True
    r1.settings.variance_estimator.tolerance = 1e-10
    r1.settings.parameter_estimator.tee = False
    r1.settings.parameter_estimator.solver = 'ipopt_sens'
    
    # Additional settings for the alternate method
    use_alternate_method = False
    if use_alternate_method:
        r1.settings.variance_estimator.method = 'alternate'
        r1.settings.variance_estimator.secant_point = 6e-8
        r1.settings.variance_estimator.initial_sigmas = 1e-8
    
    # This is all you need to run KIPET!
    r1.run_opt()
    
    # Display the results
    r1.results.show_parameters
    
    # New plotting methods
    if with_plots:
        r1.plot()
