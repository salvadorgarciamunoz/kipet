"""Example 2: Estimation with new KipetModel"""

# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports

# Kipet library imports
from kipet.kipet import KipetModel, KipetModelBlock

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=====================================================`====================
       
    kipet_model = KipetModel(name='Ex-2')
    
    # Add the model parameters
    kipet_model.add_parameter('k1', init=2, bounds=(0.0, 5.0))
    kipet_model.add_parameter('k2', init=0.2, bounds=(0.0, 2.0))
    
    # Declare the components and give the initial values
    kipet_model.add_component('A', state='concentration', init=1)
    kipet_model.add_component('B', state='concentration', init=0.0, variance=1)
    kipet_model.add_component('C', state='concentration', init=0.0, variance=1)
    
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
    
    # Big change? Make the initial component values part of the data and not
    # an attribute for the user to enter unless really necessary
    
    # Any calls that need to be made to the TemplateBuilder can be accessed
    # using the builder attribute
    kipet_model.builder.bound_profile(var='S', bounds=(0, 200))
    # If no times are given to the builder, it will use the times in the data
    kipet_model.create_pyomo_model()

    # Display the KipetModel object attributes
    print(kipet_model)
    
    # Make a new KipetModel using the model components of the previous
    # Note that you can add the new initial values here if necessary
    # Why is this not part of the data?
    
    new_kipet_model = kipet_model.clone(init={'A':1.2,'B': 0.1,'C': 0})
    
    print(new_kipet_model)
    
    # This block is used for multiple experiments - it is not important for
    # single models
    
    # MEE could just take this block and add some constraints after solving
    # each one individually - simply copy the models as blocks, right?
    kipet_model_set = KipetModelBlock()
    kipet_model_set.add_model(kipet_model)
    kipet_model_set.add_model(new_kipet_model)
    
    print(kipet_model_set)
    
# #%%
#     #=========================================================================
#     #USER INPUT SECTION - VARIANCE ESTIMATION 
#     #=========================================================================
    
#     # This creates the VarianceEstimator instance
#     kipet_model.create_variance_estimator(ncp=1, nfe=60)
    
#     # It is often requried for larger problems to give the solver some direct instructions
#     # These must be given in the form of a dictionary
#     options = {}
#     # While this problem should solve without changing the deault options, example code is 
#     # given commented out below. See Section 5.6 for more options and advice.
#     # options['bound_push'] = 1e-8
#     # options['tol'] = 1e-9
    
#     # Data reduction to improve speed
#     # The set A_set is then decided. This set, explained in Section 4.3.3 is used to make the
#     # variance estimation run faster and has been shown to not decrease the accuracy of the variance 
#     # prediction for large noisey data sets.
#     A_set = kipet_model.reduce_spectra_data_set()

#     # Finally we run the variance estimatator using the arguments shown in Section 4.3.3
#     # New method run_ve_opt to call VarianceEstimator.run_opt()
    
#     kipet_model.run_ve_opt('ipopt',
#                      tee=True,
#                      solver_options=options,
#                      tolerance=1e-5,
#                      max_iter=15,
#                      #method='alternate',
#                      subset_lambdas=A_set
#                      )

#     # Variances can then be displayed 
#     print("\nThe estimated variances are:\n")
#     kipet_model.results['v_estimator'].variances

#     # New method to create and discretize the ParameterEstimation instance
#     # The resulting model stored as p_model and the p_estimator is now an attribute
#     kipet_model.create_parameter_estimator(ncp=1, nfe=60)
    
#     # Certain problems may require initializations and scaling and these can be provided from the 
#     # varininace estimation step. This is optional.
#     # Initialize methods make it simpler to use the variance trajectories
#     kipet_model.initialize_from_trajectory(source=kipet_model.results['v_estimator'])
#     kipet_model.scale_variables_from_trajectory(source=kipet_model.results['v_estimator'])
    
#     # Again we provide options for the solver
#     options = dict()
#     options['nlp_scaling_method'] = 'user-scaling'
    
#     # New method run_pe_opt to call ParameterEstimator.run_opt()
#     kipet_model.run_pe_opt('ipopt',
#                            tee=True,
#                            solver_opts=options,
#                            variances=kipet_model.results['v_estimator'].sigma_sq)


    #%%
    kipet_model.run_opt(options={}, ncp=1, nfe=60, scale_variances=False)
    
    #%%

    # And display the results
    print("The estimated parameters are:")
    kipet_model.results['p_estimator'].parameters
    
    # New plotting methods
    if with_plots:
        kipet_model.results['p_estimator'].plot('C')
        kipet_model.results['p_estimator'].plot('S')