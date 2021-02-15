"""Example 10: Wavelength subset selection using lack of fit

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
    k1 = r1.parameter('k1', value=2, bounds=(0.0, 5.0))
    k2 = r1.parameter('k2', value=0.2, bounds=(0.0, 2.0))
    
    # Declare the components and give the initial values
    A = r1.component('A', value=1)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0)
    
    # Use this function to replace the old filename set-up
    r1.add_data('D_frame', category='spectral', file='data/Dij.txt')

    rates = {}    
    rates['A'] = -k1 * A
    rates['B'] = k1 * A - k2 * B
    rates['C'] = k2 * B
    
    r1.add_odes(rates)
    
    r1.bound_profile(var='S', bounds=(0, 200))

    # Settings
    r1.settings.collocation.ncp = 1
    r1.settings.collocation.nfe = 60
    r1.settings.variance_estimator.use_subset_lambdas = True
    r1.settings.parameter_estimator.tee = False
    r1.settings.general.no_user_scaling = True
    r1.settings.parameter_estimator.solver = 'ipopt_sens'
    r1.settings.solver.mu_init = 1e-4
    r1.settings.general.initialize_pe = True
    r1.settings.general.scale_pe = True
    r1.settings.parameter_estimator.sim_init = False
    r1.settings.solver.linear_solver = 'ma57'
    
    r1.run_opt()
   
    # Display the results
    r1.results.show_parameters
        
    if with_plots:
        r1.plot()
    
    """Wavelength subset selection methods"""
    
    # See the tutorial for more info: Tutorial 13
    lof = r1.lack_of_fit()
    correlations = r1.wavelength_correlation()
    #r1.run_lof_analysis()
    
    # Follow the tutorial for this next step:
    # A specific range is selected and smaller steps are made
    #r1.run_lof_analysis(step_size = 0.01, search_range = (0, 0.12))
    
    # It seems like this is a good cut-off point
    subset = r1.wavelength_subset_selection(n=0.095) 
    
    # Solve the ParameterEstimator using the wavelength subset
    subset_results = r1.run_opt_with_subset_lambdas(subset) 
    
    # Display the new results
    subset_results.show_parameters
    
    # display results
    #if with_plots:
    # subset_results.plot(show_plot=with_plots)
