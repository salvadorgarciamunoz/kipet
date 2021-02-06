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
    kipet_model.ub.TIME_BASE = 'hour'
    
    r1 = kipet_model.new_reaction('cstr')
    
    # Perturb the initial parameter values by some factor
    factor = 1.2
    
    # Add the model parameters
    r1.add_parameter('Tf', value=293.15*factor, bounds=(250, 350), units='K')
    r1.add_parameter('Cfa', value=2500*factor, bounds=(100, 5000), units='mol/m**3')
    r1.add_parameter('rho', value=1025*factor, bounds=(800, 1100), units='kg/m**3')
    r1.add_parameter('delH', value=160*factor, bounds=(10, 400), units='')
    r1.add_parameter('ER', value=255*factor, bounds=(10, 500), units='')
    r1.add_parameter('k', value=2.5*factor, bounds=(0.1, 10), units='')
    r1.add_parameter('Tfc', value=283.15*factor, bounds=(250, 300), units='K')
    r1.add_parameter('rhoc', value=1000*factor, bounds=(800, 2000), units='kg/m**3')
    r1.add_parameter('h', value=3600*factor, bounds=(10, 5000), units='')
    
    # Declare the components and give the valueial values
    r1.add_component('A', value=1000, variance=0.001, units='mol/m**3')
    r1.add_state('T', value=293.15, variance=0.0625,  units='K')
    r1.add_state('Tc', value=293.15, variance=0.001, units='K')
   
    # Change this to a clearner method
    full_data = kipet_model.read_data_file('example_data/sim_chen.csv') #'cstr_t_and_c.csv')
    
    r1.add_constant('F', value=0.1, units='m**3/hour')
    r1.add_constant('Fc', value=0.15, units='m**3/hour')
    r1.add_constant('Ca0', value=1000, units='mol/m**3')
    r1.add_constant('V', value=0.2, units='m**3')
    r1.add_constant('Vc', value=0.055, units='m**3')
    r1.add_constant('Area', value=4.5, units='m**2')
    r1.add_constant('Cpc', value=1.2, units='kJ/kg/K')
    r1.add_constant('Cp', value=1.55, units='kJ/kg/K')
    
    r1.add_data('T_data', data=full_data[['T']].iloc[0::3])
    r1.add_data('A_data', data=full_data[['A']].loc[[3.9, 2.6, 1.115505]])
    
    r1.add_alg_var('rA')
    
    c = r1.get_model_vars()
    
    r1.add_algebraic('rA', c.k*exp(-c.ER/c.T)*c.A )
    
    r1.add_ode('A', c.F/c.V*(c.Cfa - c.A) - c.rA )
    r1.add_ode('T', c.F/c.V *(c.Tf - c.T) + c.delH/c.rho/c.Cp*c.rA - c.h*c.Area/c.rho/c.Cp/c.V*(c.T -c.Tc) )
    r1.add_ode('Tc', c.Fc/c.Vc *(c.Tfc - c.Tc) + c.h*c.Area/c.rhoc/c.Cpc/c.Vc*(c.T -c.Tc) )
    
    
    r1.settings.solver.print_level = 5
    r1.settings.collocation.ncp = 3
    r1.settings.collocation.nfe = 50
    
    r1.set_times(0, 5)
    #r1.simulate()

    r1.run_opt()

    # rh_method = 'global'
    # results = r1.rhps_method(calc_method=rh_method)

    # # results is a standard ResultsObject
    r1.plot()
    # results.plot('X', show_plot=with_plots)