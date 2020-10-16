"""
Advanced Demonstration 13: How to solve problems with unknown initial conditions
"""
# Standard library imports
import sys

# Third party imports

# KIPET library imports
from kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    kipet_model = KipetModel()
 
    r1 = kipet_model.new_reaction('reaction-1')   
 
    # Add the model parameters
    r1.add_parameter('k1', init=1.0, bounds=(0.0, 10.0))
    r1.add_parameter('k2', init=0.224, bounds=(0.0, 10.0))
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=0.001, known=False, bounds=(0.0, 0.1))
    r1.add_component('B', state='concentration', init=0.0)
    r1.add_component('C', state='concentration', init=0.0)
   
    # Use this function to replace the old filename set-up
    filename = 'Ex_1_C_data_withoutA.csv'
    
    r1.add_dataset(file=filename)
    
    # Define the reaction model
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs 
    
    r1.add_equations(rule_odes)


    # Repeat for the second model - the only difference is the dataset    
    r2 = kipet_model.new_reaction(name='reaction-2', model_to_clone=r1, items_not_copied='datasets')
    # Simulated second dataset with noise
    noised_data = kipet_model.add_noise_to_data(r1.datasets['C_data'].data, 0.0001) 
    
    # Add the dataset for the second model
    r2.add_dataset(data=noised_data)
    
    kipet_model.settings.collocation.nfe = 60
    kipet_model.settings.parameter_estimator.solver = 'ipopt_sens'

    r1.variances = {'device':1e-10,'A':1e-10,'B':1e-10,'C':1e-10}
    r2.variances = {'device':1e-10,'A':1e-10,'B':1e-10,'C':1e-10}
    
    kipet_model.run_opt()
    
    # Plot the results
    for model, results in kipet_model.results.items():
        results.show_parameters
        results.plot(show_plot=with_plots)    