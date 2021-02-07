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
    
    components = dict()
    components['A'] = 3e-2
    components['B'] = 4e-2
    components['C'] = 0.0
    components['D'] = 2e-2
    components['E'] = 0.0
    components['F'] = 0.0
    
    for comp, init_value in components.items():
        r1.add_component(comp, value=init_value)

    filename = 'example_data/varest3.csv'
    r1.add_data(category='spectral', file=filename, remove_negatives=True)
    full_data = kipet_model.read_data_file(filename)
    
    # r1.spectra.decrease_wavelengths(2)

    r1.add_parameter('k1', value=1.5, bounds=(0.5, 2.0)) 
    r1.add_parameter('k2', value=28.0, bounds=(1, 30))
    r1.add_parameter('k3', value=0.3, bounds=(0.001, 0.5))

    c = r1.get_model_vars()

    r1.add_ode('A', -c.k1*c.A )
    r1.add_ode('B', -c.k1*c.A - c.k3*c.B )
    r1.add_ode('C', -c.k2*c.C*c.D + c.k1*c.A )
    r1.add_ode('D', -2*c.k2*c.C*c.D )
    r1.add_ode('E', c.k2*c.C*c.D )
    r1.add_ode('F', c.k3*c.B )
    
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
