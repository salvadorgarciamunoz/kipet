"""Example 12: Multiple Experimental Datasets with the new KipetModel
 
This examples uses two reactions with concentration data where the second data
set is noisy
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
    
    # Add the parameters
    k1 = r1.parameter('k1', value=1.0, bounds=(0.0, 10.0))
    k2 = r1.parameter('k2', value=0.224, bounds=(0.0, 10.0))
    
    # Declare the components and give the initial values
    A = r1.component('A', value=1.0e-3)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0)
    
    # define explicit system of ODEs
    rates = {}
    rates['A'] = -k1 * A
    rates['B'] = k1 * A - k2 * B
    rates['C'] = k2 * B
    
    r1.add_odes(rates)
   
    # Add the dataset for the first model
    r1.add_data('C_data', file='data/Ex_1_C_data.txt')
    
    # Add the known variances
    r1.variances = {'A':1e-10,'B':1e-10,'C':1e-10}
    
    r2 = kipet_model.new_reaction(name='reaction-2', model=r1)
   
    # Add the dataset for the first model
    noised_data = kipet_model.add_noise_to_data(r1.data['C_data'], 0.0001) 
    r2.add_data('C_data', data=noised_data[::10])
    
    # Add the known variances
    r2.components.update('variance', {'A':1e-4,'B':1e-4,'C':1e-4})
    # # Create the MultipleExperimentsEstimator and perform the parameter fitting
    kipet_model.run_opt()

    # Plot the results
    if with_plots:     
        for name, model in kipet_model.models.items():
            kipet_model.results[name].show_parameters
            model.plot()
