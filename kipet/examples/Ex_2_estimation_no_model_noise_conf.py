"""Example 2: Estimation with new KipetModel

No Model Noise with parameter confidence intervals
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
    
    # Add the model parameters
    kipet_model.add_parameter('k1', init=2, bounds=(0.0, 5.0))
    kipet_model.add_parameter('k2', init=0.2, bounds=(0.0, 2.0))
    
    # Declare the components and give the initial values
    kipet_model.add_component('A', state='concentration', init=1e-3)
    kipet_model.add_component('B', state='concentration', init=0.0)
    kipet_model.add_component('C', state='concentration', init=0.0)
    
    # Use this function to replace the old filename set-up
    filename = kipet_model.set_directory('Dij.txt')
    kipet_model.add_dataset('D_frame', category='spectral', file=filename)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    kipet_model.add_equations(rule_odes)
    kipet_model.builder.bound_profile(var='S', bounds=(0, 200))
    # If no times are given to the builder, it will use the times in the data
    kipet_model.create_pyomo_model()
    
    # Display the KipetTemplate object attributes
    print(kipet_model)

    # Settings
    kipet_model.settings.general.initialize_pe = False
    kipet_model.settings.general.scale_pe = False
    
    kipet_model.settings.collocation.ncp = 1
    kipet_model.settings.collocation.nfe = 100
    
    kipet_model.settings.variance_estimator.use_subset_lambdas = False
    kipet_model.settings.variance_estimator.max_device_variance = True

    kipet_model.settings.variance_estimator.tee = False
    kipet_model.settings.parameter_estimator.solver = 'k_aug'

    
    # Show the KipetModel settings
    kipet_model._update_related_settings()
    print(kipet_model.settings)
    
    # This is all you need to run KIPET!
    kipet_model.run_opt()
    
    # Display the results
    print("The estimated parameters are:")
    kipet_model.results.parameters
    
    # New plotting methods
    if with_plots:
        kipet_model.results.plot('Z', show_exp=False)
