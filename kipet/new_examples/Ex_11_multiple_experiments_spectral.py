"""Example 11: Multiple Experimental Datasets with the new KipetModel
 
"""
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
 
    # Define the general model
    kipet_model = KipetModel()
    
    # Add the model parameters
    kipet_model.add_parameter('k1', init=1.0, bounds=(0.0, 10.0))
    kipet_model.add_parameter('k2', init=0.224, bounds=(0.0, 10.0))
    
    # Declare the components and give the initial values
    kipet_model.add_component('A', state='concentration', init=1e-3)
    kipet_model.add_component('B', state='concentration', init=0.0)
    kipet_model.add_component('C', state='concentration', init=0.0)
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    kipet_model.add_equations(rule_odes)
    
    # For clarity, the first model is cloned although this is not necessary
    kipet_model1 = kipet_model.clone(name='Model-1')
  
    # Add the dataset for the first model
    filename1 = kipet_model1.set_directory('Dij_exp1.txt')
    kipet_model1.add_dataset('D_frame1', category='spectral', file=filename1)

    # Repeat for the second model - the only difference is the dataset    
    kipet_model2 = kipet_model.clone(name='Model-2')

    # Add the dataset for the second model
    filename2 = kipet_model.set_directory('Dij_exp3_reduced.txt')
    kipet_model2.add_dataset('D_frame2', category='spectral', file=filename2)

    kipet_model2.datasets['D_frame2'].data = kipet_model2.add_noise_to_data(kipet_model2.datasets['D_frame2'].data, 0.000000011) 
    
    # Create the KipetModelBlock instance to hold the KipetModels
    model_block = KipetModelBlock()
    model_block.add_model(kipet_model1)
    model_block.add_model(kipet_model2)
    
    model_block.settings.general.use_wavelength_subset = True
    model_block.settings.general.freq_wavelength_subset = 3
    model_block.settings.collocation.nfe = 100
    
    # If you provide your variances, they need to added directly to run_opt
    user_provided_variances = {'A':1e-10,'B':1e-10,'C':1e-11,'device':1e-6}
    
    """Using confidence intervals - uncomment the following three lines"""
    
    #model_block.settings.parameter_estimator.solver = 'ipopt_sens'
    #model_block.settings.parameter_estimator.covariance = True
    
    # If it is not solving properly, try scaling the variances
    #model_block.settings.parameter_estimator.scaled_variance = True
    
    """End of confidence interval section"""
    
    # Create the MultipleExperimentsEstimator and perform the parameter fitting
    model_block.run_opt(variances=user_provided_variances)

    # Plot the results
    for model, results in model_block.results.items():
        results.show_parameters
        results.plot(show_plot=with_plots)
