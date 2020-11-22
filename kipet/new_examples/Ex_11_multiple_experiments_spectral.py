"""Example 11: Multiple Experimental Datasets with the new KipetModel
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
 
    # Define the general model
    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction(name='reaction-1')
    
    # Add the model parameters
    r1.add_parameter('k1', init=1.0, bounds=(0.0, 10.0))
    r1.add_parameter('k2', init=0.224, bounds=(0.0, 10.0))
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=1e-3)
    r1.add_component('B', state='concentration', init=0.0)
    r1.add_component('C', state='concentration', init=0.0)
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    r1.add_equations(rule_odes)
    
    # Add the dataset for the first model
    r1.add_dataset(file='Dij_exp1.txt', category='spectral')

    # Repeat for the second model - the only difference is the dataset    
    r2 = kipet_model.new_reaction(name='reaction_2', model_to_clone=r1, items_not_copied='datasets')

    # Add the dataset for the second model
    r2.add_dataset(file='Dij_exp3_reduced.txt', category='spectral')

    kipet_model.settings.general.use_wavelength_subset = True
    kipet_model.settings.general.freq_wavelength_subset = 3
    kipet_model.settings.collocation.nfe = 100
    
    # If you provide your variances, they need to added directly to run_opt
    #user_provided_variances = {'A':1e-10,'B':1e-10,'C':1e-11,'device':1e-6}
    """Using confidence intervals - uncomment the following three lines"""
    
    kipet_model.settings.parameter_estimator.solver = 'ipopt_sens'
    #kipet_model.settings.parameter_estimator.covariance = True
    
    # If it is not solving properly, try scaling the variances
    #kipet_model.settings.parameter_estimator.scaled_variance = True
    
    """End of confidence interval section"""
    
    # Create the MultipleExperimentsEstimator and perform the parameter fitting
    kipet_model.run_opt()

    # Plot the results
    for model, results in kipet_model.results.items():
        results.show_parameters
        results.plot(show_plot=with_plots)