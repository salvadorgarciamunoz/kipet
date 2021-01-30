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
    filename = 'example_data/delayed_data.csv'
    full_data = kipet_model.read_data_file(filename)
    
    data_set = full_data.iloc[::3]
    r1.add_dataset('C_data', category='concentration', data=data_set)
    
    # Add step functions to the template (m.step[t, name])
    r1.add_step('b1', time=2, fixed=False, switch='on')
    r1.add_step('b2', time=2.1, fixed=False, switch='on')
    
    # Define the reaction model
    def rule_odes(m,t):
        exprs = dict()
        
        # Multiply each reaction by the step function
        R1 = m.step[t, 'b1']*(m.P['k1']*m.Z[t,'A'])
        R2 = m.step[t, 'b2']*(m.P['k2']*m.Z[t,'B'])
        
        # ODEs defined using reactions
        exprs['A'] = -R1
        exprs['B'] = R1 - R2
        exprs['C'] = R2
        return exprs 
    
    r1.add_odes(rule_odes)
    
    # Settings
    r1.settings.collocation.nfe = 60
    r1.settings.collocation.ncp = 3
    
    r1.settings.parameter_estimator.solver = 'mumps'
    r1.settings.parameter_estimator.sim_init = True
    
    # # Run KIPET
    r1.run_opt()  
    
    # Display the results
    r1.results.show_parameters

    r1.results.plot('Z', 
                    show_plot=with_plots,
                    description={'title': 'Example 7',
                                  'xaxis': 'Time [s]',
                                  'yaxis': 'Concentration [mol/L]'})