"""Example 7: Estimation using measured concentration data with new KipetModel"""

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
 
    full_data = kipet_model.read_data_file('example_data/Ex_1_C_data.txt')

    # r1.add_dataset(data=full_data[['A']], remove_negatives=True)   
 
    # Add the model parameters
    r1.add_parameter('k1', value=2.0, bounds=(0.0, 5.0))
    r1.add_parameter('k2', value=0.2, bounds=(0.0, 2.0), fixed=False)
    
    # Declare the components and give the initial values
    r1.add_component('A', value=0.001, variance=1e-10, known=False, bounds=(0.0, 3))
    r1.add_component('B', value=0.0, variance=1e-11)
    r1.add_component('C', value=0.0, variance=1e-8)
   
    # Use this function to replace the old filename set-up
    filename = 'example_data/Ex_1_C_data.txt'
    
    r1.add_data(data=full_data.iloc[::10, :], remove_negatives=True)   
    
    c = r1.get_model_vars()
    
    # Define the reaction model
    r1.add_ode('A', -c.k1 * c.A )
    r1.add_ode('B', c.k1 * c.A - c.k2 * c.B )
    r1.add_ode('C', c.k2 * c.B )
    
    # Settings
    r1.settings.collocation.nfe = 60
    r1.settings.parameter_estimator.solver = 'k_aug'
    
    # Run KIPET
    r1.run_opt()  
    
    # Display the results
    r1.results.show_parameters

    # if with_plots:
    #     r1.plot()