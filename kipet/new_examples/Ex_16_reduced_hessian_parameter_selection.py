"""
Example 16: Parameter Selection Using the Reduced Hessian

Kipet: Kinetic parameter estimation toolkit
Copyright (c) 2016 Eli Lilly.
 
Example from Chen and Biegler, Reduced Hessian Based Parameter Selection and
    Estimation with Simultaneous Collocation Approach (AIChE 2020) paper with
    a CSTR for a simple reaction.
    
This example uses reactor temperature as the known output data as well as some
concentration data.
"""
# Standard library imports
import sys

# Third party imports
from pyomo.environ import exp

# Kipet library imports
import kipet

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    kipet_model = kipet.KipetModel()
    
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

    # global is somewhat faster than fixed
    #rh_method = 'fixed'
    rh_method = 'global'
 
    results = r1.reduce_model(calc_method=rh_method)

    # results is a standard ResultsObject
    results.plot('Z', show_plot=with_plots)
    results.plot('X', show_plot=with_plots)