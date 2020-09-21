"""Example 2: Estimation using alternate variance method with new KipetModel"""

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
       
    kipet_model = KipetModel(name='Ex-2')
    
    # Add the model parameters
    kipet_model.add_parameter('k1', init=2, bounds=(1.0, 5.0))
    kipet_model.add_parameter('k2', init=0.2, bounds=(0.0, 1.0))
    
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
    
    # Any calls that need to be made to the TemplateBuilder can be accessed
    # using the builder attribute
    kipet_model.builder.bound_profile(var='S', bounds=(0, 200))
    # If no times are given to the builder, it will use the times in the data
    kipet_model.create_pyomo_model()

    # Display the KipetModel object attributes
    print(kipet_model)
    
    # Change some of the default settings
    # Settings can be set like this:
    # options = {}
    # options['linear_solver'] = 'ma57'
    # options['max_iter'] = 2000
    # kipet_model.settings.collocation.update({'ncp': 3, 'nfe': 100})
    # kipet_model.settings.variance_estimator.solver_options.update(options)
    # kipet_model.settings.variance_estimator.update({'method': 'alternate', 'tee': False})
    # kipet_model.settings.parameter_estimator.update({'solver': 'ipopt_sens', 'covariance': True})
    
    # Or like this:
    kipet_model.settings.collocation.ncp = 3
    kipet_model.settings.collocation.nfe = 100
    kipet_model.settings.variance_estimator.solver_options.linear_solver = 'ma57'
    kipet_model.settings.variance_estimator.solver_options.max_iter = 2000
    kipet_model.settings.variance_estimator.method = 'alternate'
    kipet_model.settings.parameter_estimator.solver = 'ipopt_sens'
    kipet_model.settings.parameter_estimator.covariance = True
    kipet_model.settings.parameter_estimator.tee = False
    
    # Show the KipetModel settings
    print(kipet_model.settings)
    
    # This is all you need to run KIPET!
    kipet_model.run_opt()
    
    # Display the results
    print("The estimated parameters are:")
    kipet_model.results.parameters
    
    # New plotting method
    if with_plots:
        kipet_model.results.plot()