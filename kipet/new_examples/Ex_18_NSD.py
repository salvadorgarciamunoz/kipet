"""
NSD Example using the new KIPET format
"""
# Standard library imports
import sys

# Third party imports

# KIPET library imports
from kipet import KipetModel
from kipet.library.nsd_funs.NSD_TrustRegion import NSD
   
def generate_models():

    kipet_model = KipetModel()
 
    r1 = kipet_model.new_reaction('reaction-1')   
 
    factor = 1.2   
 
    # Add the model parameters
    r1.add_parameter('k1', init=0.25*factor, bounds=(0.0, 10.0))
    r1.add_parameter('k2', init=1*factor, bounds=(0.0, 10.0))
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=0.001, variance=1) #, known=False, bounds=(0.0, 0.1))
    r1.add_component('B', state='concentration', init=0.0, variance=1)
    r1.add_component('C', state='concentration', init=0.0, variance=1)
   
    # Use this function to replace the old filename set-up
    filename = 'example_data/Ex_1_C_data_withoutA.csv'
    full_data = kipet_model.read_data_file(filename)
    
    r1.add_dataset(data=full_data)#.iloc[0::6])
    
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
    
    kipet_model.settings.collocation.nfe = 50
    kipet_model.settings.parameter_estimator.solver = 'ipopt'

    #r1.variances = {'device':1e-10,'A':1e-10,'B':1e-10,'C':1e-10}
    #r2.variances = {'device':1e-10,'A':1e-10,'B':1e-10,'C':1e-10}
    
    r1.create_pyomo_model()
    r2.create_pyomo_model()

    models = [r1, r2]
    
    return models
    

if __name__ == '__main__':

    # Generate the ReactionModels (at some point KipetModel)
    models = generate_models()   

    # Create the NSD object using the ReactionModels list
    nsd = NSD(models)
    
    # Set the initial values
    nsd.set_initial_value({'k1' : 0.3,
                            'k2' : 1.2}
                         )
    
    # Runs the TR Method
    results = nsd.trust_region(scaled=False)
    # Plot the results using ReactionModel format
    nsd.plot_results()
    # Plot the parameter value paths
    nsd.plot_paths()



