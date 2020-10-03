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
    filename1 = kipet_model1.set_directory('Ex_1_C_data.txt')
    kipet_model1.add_dataset('C_frame1', category='concentration', file=filename1)

    # Repeat for the second model - the only difference is the dataset    
    kipet_model2 = kipet_model.clone(name='Model-2')

    # Simulated second dataset with noise
    noised_data = kipet_model1.add_noise_to_data(kipet_model1.datasets['C_frame1'].data, 0.0001) 
    
    # Add the dataset for the second model
    kipet_model2.add_dataset('C_frame2', category='concentration', data=noised_data)

    # Create the KipetModelBlock instance to hold the KipetModels
    model_block = KipetModelBlock()
    model_block.add_model(kipet_model1)
    model_block.add_model(kipet_model2)
    
    """Using confidence intervals - uncomment the following two lines"""
    # If using knonw variances, add them to the create method
    # This will skip the variance estimation step
    
    user_provided_variances = {'A':1e-10,'B':1e-10,'C':1e-10}
    
    # If you want the confidence intervals, change the default solver
    model_block.settings.parameter_estimator.solver = 'ipopt_sens'
    model_block.settings.parameter_estimator.covariance = True
    
    #model_block.settings.parameter_estimator.spectra_problem = False
    
    """End of confidence interval section"""
    
    # Create the MultipleExperimentsEstimator and perform the parameter fitting
    model_block.create_multiple_experiments_estimator(variances=user_provided_variances)
    model_block.run_multiple_experiments_estimator()

    # Plot the results
    for model, results in model_block.results.items():
        results.show_parameters
        results.plot()
