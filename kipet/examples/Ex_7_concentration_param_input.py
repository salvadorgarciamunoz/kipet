"""Example 7: New data and component inputs"""

# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports

# Kipet library imports
from kipet.kipet import KipetModel # The only thing you need for using Kipet

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
    kipet_model = KipetModel()   
 
    # New method for adding parameters
    kipet_model.add_parameter('k1', bounds=(0.0, 5.0))
    kipet_model.add_parameter('k2', bounds=(0.0, 2.0))
    
    # A similar method exists for the Components
    kipet_model.add_component('A', state='concentration', init=1e-3, variance=1e-10)
    kipet_model.add_component('B', state='concentration', init=0.0, variance=1e-11)
    kipet_model.add_component('C', state='concentration', init=0.0, variance=1e-10)
    
    # Use this function to replace the old filename set-up
    filename = kipet_model.set_directory('Ex_7_C_data_short.csv')
    #filename = model.set_directory('Ex_1_C_data.csv')
    
    kipet_model.add_dataset('C_data', category='concentration', file=filename)
    
    # Display the inputs
    print(kipet_model.parameters)
    print(kipet_model.components)
    print(kipet_model.datasets)  
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    kipet_model.add_equations(rule_odes)
    
    # If no times are given to the builder, it will use the times in the data
    kipet_model.create_pyomo_model(scale_parameters=True)
    
    # If you have lots of data, this will take a while
    # This calls the EstimationPotential module to reduce the model using the
    # reduced hessian parameter selection method - the model attribute is then
    # set as the reduced model. If all parameters are estimable, this has no
    # effect except for changing the parameter values.
    # TODO: if no reduction takes place, report this
    kipet_model.reduce_model()
    
    # Display the KipetTemplate object attributes
    print(kipet_model)
    
    # Again we provide options for the solver, this time providing the scaling that we set above
    options = dict()
    options['nlp_scaling_method'] = 'user-scaling'

    # Run the optimization
    kipet_model.create_parameter_estimator()
    kipet_model.run_pe_opt('ipopt',
                     variances=kipet_model.components.variances,
                     tee=True,
                     solver_opts=options)

    # Optimal parameters can be shown using parameters property of the results object
    print("The estimated parameters are:")
    kipet_model.results['p_estimator'].parameters

    # display results
    if with_plots:
        kipet_model.results['p_estimator'].plot()