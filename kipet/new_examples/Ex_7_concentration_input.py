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
 
    # Add the model parameters
    r1.add_parameter('k1', init=2.0, bounds=(0.0, 5.0))
    r1.add_parameter('k2', init=0.2, bounds=(0.0, 2.0))
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=0.001, variance=1e-10)
    r1.add_component('B', state='concentration', init=0.0, variance=1e-11)
    r1.add_component('C', state='concentration', init=0.0, variance=1e-8)
   
    # Use this function to replace the old filename set-up
    filename = 'example_data/Ex_1_C_data.txt'
    r1.add_dataset('C_data', category='concentration', file=filename)
    
    # Define the reaction model
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs 
    
    r1.add_equations(rule_odes)
    
    # Settings
    r1.settings.collocation.nfe = 60
    r1.settings.parameter_estimator.solver = 'k_aug'
    
    # Run KIPET
    r1.run_opt()  
    
    # Display the results
    r1.results.show_parameters

    r1.results.plot('Z', 
                    show_plot=with_plots,
                    description={'title': 'Example 7',
                                  'xaxis': 'Time [s]',
                                  'yaxis': 'Concentration [mol/L]'})