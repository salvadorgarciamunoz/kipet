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
 
    # Develop the KipetModel with all components
    k = KipetModel()
    
    # Add the model parameters
    k.parameter('k1', value=2.0, bounds=(0.0, 5.0))
    k.parameter('k2', value=0.2, bounds=(0.0, 2.0), fixed=False)
    k.parameter('k3', value=5, bounds=(0, 10))

    # Declare the components and give the initial values
    k.component('A', value=0.001, variance=1e-10, known=False, bounds=(0.0, 3))
    k.component('B', value=0.0, variance=1e-11)
    k.component('C', value=0.0, variance=1e-8)
    
    c = k.get_model_vars()
    
    # Starting building the reaction model (odes, algs, data)
    
    # Use this function to replace the old filename set-up
    filename = 'example_data/Ex_1_C_data.txt'
    
    r1 = k.new_reaction('reaction-1')   
    full_data = k.read_data_file('example_data/Ex_1_C_data.txt')
    r1.add_data(data=full_data.iloc[::10, :], remove_negatives=True)   
    
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
   # r1.results.show_parameters

    # if with_plots:
    #     r1.plot()