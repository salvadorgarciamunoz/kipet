"""Example 7: Estimation using measured concentration data with new KipetModel"""

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
 
    kipet_model = KipetModel(name='Ex-7')
    
    # Add the model parameters
    kipet_model.add_parameter('k1', init=2.0, bounds=(0.0, 5.0))
    kipet_model.add_parameter('k2', init=0.2, bounds=(0.0, 2.0))
    
    # Declare the components and give the initial values
    kipet_model.add_component('A', state='concentration', init=0.001)#, variance=1e-10)
    kipet_model.add_component('B', state='concentration', init=0.0)#, variance=1e-11)
    kipet_model.add_component('C', state='concentration', init=0.0)#, variance=1e-10)
   
    # Use this function to replace the old filename set-up
    filename = kipet_model.set_directory('missing_data.txt')
    
    kipet_model.add_dataset('C_data', category='concentration', file=filename)
    
    # Define the reaction model
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs 
    
    kipet_model.add_equations(rule_odes)
    
    # Settings
    kipet_model.settings.collocation.nfe = 60
    
    # Run KIPET
    kipet_model.run_opt()  
    
    # Display the results
    kipet_model.results.show_parameters

    if with_plots:
        kipet_model.results.plot()