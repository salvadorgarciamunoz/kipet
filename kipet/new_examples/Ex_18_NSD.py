"""
NSD Example using the new KIPET format
"""
# Standard library imports
import sys

# Third party imports

# KIPET library imports
from kipet import KipetModel
from kipet.library.nsd_funs.NSD_TrustRegion_Ipopt import NSD
   
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
    
def generate_models_cstr():

    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('cstr')
  
    # Perturb the initial parameter values by some factor
    factor = 1.2
    
    # Add the model parameters
    r1.add_parameter('Tf', init=293.15*factor, bounds=(250, 350))
    r1.add_parameter('Cfa', init=2500*factor, bounds=(100, 5000))
    r1.add_parameter('rho', init=1025*factor, bounds=(800, 1100))
    r1.add_parameter('delH', init=160*factor, bounds=(10, 400))
    r1.add_parameter('ER', init=255*factor, bounds=(10, 500))
    r1.add_parameter('k', init=2.5*factor, bounds=(0.1, 10))
    r1.add_parameter('Tfc', init=283.15*factor, bounds=(250, 300))
    r1.add_parameter('rhoc', init=1000*factor, bounds=(800, 2000))
    r1.add_parameter('h', init=3600*factor, bounds=(10, 5000))
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=1000, variance=0.001)
    r1.add_component('T', state='state', init=293.15, variance=0.0625)
    r1.add_component('Tc', state='state', init=293.15, variance=0.001)
   
    # Change this to a clearner method
    full_data = kipet_model.read_data_file('example_data/sim_chen.csv') #'cstr_t_and_c.csv')
    
    constants = {
            'F' : 0.1, # m^3/h
            'Fc' : 0.15, # m^3/h
            'Ca0' : 1000, # mol/m^3
            'V' : 0.2, # m^3
            'Vc' : 0.055, # m^3
            'A' : 4.5, # m^2
            'Cpc' : 1.2, # kJ/kg/K
            'Cp' : 1.55, # kJ/kg/K
            }
    
    # Make it easier to use the constants in the ODEs
    C = constants
      
    
    from pyomo.environ import exp
    # Define the model ODEs
    def rule_odes(m,t):
        
        Ra = m.P['k']*exp(-m.P['ER']/m.X[t,'T'])*m.Z[t,'A']
        exprs = dict()
        exprs['A'] = C['F']/C['V']*(m.P['Cfa']-m.Z[t,'A']) - Ra
        exprs['T'] = C['F']/C['V']*(m.P['Tf']-m.X[t,'T']) + m.P['delH']/(m.P['rho'])/C['Cp']*Ra - m.P['h']*C['A']/(m.P['rho'])/C['Cp']/C['V']*(m.X[t,'T'] - m.X[t,'Tc'])
        exprs['Tc'] = C['Fc']/C['Vc']*(m.P['Tfc']-m.X[t,'Tc']) + m.P['h']*C['A']/(m.P['rhoc'])/C['Cpc']/C['Vc']*(m.X[t,'T'] - m.X[t,'Tc'])
        return exprs

    r1.add_equations(rule_odes)
    
    r1.settings.solver.print_level = 5
    r1.settings.collocation.ncp = 3
    r1.settings.collocation.nfe = 50
    
    r1.add_dataset(data=full_data[['T']].iloc[0::3])
    r1.add_dataset(data=full_data[['A']].loc[[3.9, 2.6, 1.115505]])

    #r1.variances = {'device':1e-10,'A':1e-10,'B':1e-10,'C':1e-10}
    #r2.variances = {'device':1e-10,'A':1e-10,'B':1e-10,'C':1e-10}
    
    r1.create_pyomo_model()
    #r2.create_pyomo_model()

    models = [r1]#, r2]
    
    return models


if __name__ == '__main__':

    # Generate the ReactionModels (at some point KipetModel)
    models = generate_models()   

    # Create the NSD object using the ReactionModels list
    kwargs = {'kipet': True,
              'objective_multiplier': 1e6}
    
    nsd = NSD(models, kwargs=kwargs)
    
    # Set the initial values
    nsd.set_initial_value({'k1' : 0.6,
                            'k2' : 1.2}
                          )
    
    # Runs the TR Method
    results = nsd.ipopt_method(scaled=False)
    #results = nsd.trust_region(scaled=False)
    
    # Plot the results using ReactionModel format
    nsd.plot_results()
    # Plot the parameter value paths
    nsd.plot_paths()



