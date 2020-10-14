"""Example 10: Wavelength subset selection using lack of fit

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
    
    r1 = kipet_model.new_reaction('reaction-1')
    
    # Add the model parameters
    r1.add_parameter('k1', init=4.0, bounds=(0.0, 5.0))
    r1.add_parameter('k2', init=0.5, bounds=(0.0, 1.0))
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=1e-3)
    r1.add_component('B', state='concentration', init=0.0)
    r1.add_component('C', state='concentration', init=0.0)
    
    # Use this function to replace the old filename set-up
    filename = r1.set_directory('Dij.txt')
    r1.add_dataset('D_frame', category='spectral', file=filename)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    r1.add_equations(rule_odes)
    
    #r1.bound_profile(var='S', bounds=(0, 200))

    # Display the KipetTemplate object attributes
    print(r1)

    # Settings
    r1.settings.collocation.ncp = 3
    r1.settings.collocation.nfe = 60
    r1.settings.variance_estimator.use_subset_lambdas = True
    r1.settings.variance_estimator.tolerance = 1e-5
    r1.settings.parameter_estimator.tee = False
    
    # Show the KipetModel settings
    print(r1.settings)
    
    # This is all you need to run KIPET!
    #r1.set_times(0, 10)
    r1.run_opt()
    
    # Display the results
    r1.results.show_parameters
        
    r1.results.plot(show_plot=with_plots)
    
   
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
    subset_results.plot(show_plot=with_plots)
