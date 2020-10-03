"""Example 14: Estimation with new KipetModel

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
    
    components = dict()
    components['A'] = 3e-2
    components['B'] = 4e-2
    components['C'] = 0.0
    components['D'] = 2e-2
    components['E'] = 0.0
    components['F'] = 0.0
    
    for comp, init_value in components.items():
        kipet_model.add_component(comp, state='concentration', init=init_value)
    
    # Use this function to replace the old filename set-up
    filename = kipet_model.set_directory('varest3.csv')
    kipet_model.add_dataset('D_frame', category='spectral', file=filename, remove_negatives=True)

    kipet_model.add_parameter('k1', init=1.5, bounds=(0.5, 2.0)) 
    kipet_model.add_parameter('k2', init=28.0, bounds=(1, 30))
    kipet_model.add_parameter('k3', init=0.3, bounds=(0.001, 0.5))

    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = -m.P['k1']*m.Z[t,'A']-m.P['k3']*m.Z[t,'B']
        exprs['C'] = -m.P['k2']*m.Z[t,'C']*m.Z[t,'D'] +m.P['k1']*m.Z[t,'A']
        exprs['D'] = -2*m.P['k2']*m.Z[t,'C']*m.Z[t,'D']
        exprs['E'] = m.P['k2']*m.Z[t,'C']*m.Z[t,'D']
        exprs['F'] = m.P['k3']*m.Z[t,'B']
        return exprs

    kipet_model.add_equations(rule_odes)
    kipet_model.bound_profile(var='S', bounds=(0, 20))

    # Settings
    kipet_model.settings.general.no_user_scaling = True
    kipet_model.settings.variance_estimator.tolerance = 1e-10
    kipet_model.settings.parameter_estimator.tee = False
    kipet_model.settings.parameter_estimator.solver = 'ipopt_sens'
    
    # Additional settings for the alternate method
    use_alternate_method = False
    if use_alternate_method:
        kipet_model.settings.variance_estimator.method = 'alternate'
        kipet_model.settings.variance_estimator.secant_point = 6e-8
        kipet_model.settings.variance_estimator.initial_sigmas = 1e-8
    
    # This is all you need to run KIPET!
    kipet_model.run_opt()
    
    # Display the results
    kipet_model.results.show_parameters
    
       # New plotting methods
    if with_plots:
        kipet_model.results.plot()
