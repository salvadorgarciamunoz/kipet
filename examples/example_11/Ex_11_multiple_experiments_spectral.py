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
    
    r1 = kipet_model.new_reaction('reaction-1')

    # Add the model parameters
    k1 = r1.parameter('k1', value=1.0, bounds=(0.0, 10.0))
    k2 = r1.parameter('k2', value=0.224, bounds=(0.0, 10.0))
    
    # Declare the components and give the initial values
    A = r1.component('A', value=1e-3)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0)
    
    # Use this function to replace the old filename set-up
    r1.add_data(category='spectral', file='data/Dij_exp1.txt')
    
    # Preprocessing!
    #r1.spectra.msc()
    #r1.spectra.decrease_wavelengths(A_set=2)

 
    # define explicit system of ODEs
    rates = {}
    rates['A'] = -k1 * A
    rates['B'] = k1 * A - k2 * B
    rates['C'] = k2 * B
    
    r1.add_odes(rates)
    
    # Settings
    r1.settings.collocation.ncp = 1
    r1.settings.collocation.nfe = 100
    r1.settings.parameter_estimator.scaled_variance = False
    r1.settings.parameter_estimator.solver = 'ipopt'
  
    r1.spectra.decrease_wavelengths(4)
    r1.run_opt()
   
    # Repeat for the second model - the only difference is the dataset    
    r2 = kipet_model.new_reaction(name='reaction_2', model=r1)

    # Add the dataset for the second model
    r2.add_data(file='data/Dij_exp3_reduced.txt', category='spectral')
    r2.spectra.decrease_wavelengths(4)
    r2.run_opt() 

    """Using confidence intervals - uncomment the following three lines"""
    kipet_model.settings.solver.solver = 'ipopt_sens'
    kipet_model.settings.general.shared_spectra=True
    
    # Create the MultipleExperimentsEstimator and perform the parameter fitting
    kipet_model.run_opt()

    # Plot the results
    if with_plots:    
        for name, model in kipet_model.models.items():
            kipet_model.results[name].show_parameters
            model.plot()
