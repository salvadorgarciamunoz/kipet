"""
Example 17: Using custom data in the objective

This example uses data representing the ratio of component B to the total of
B and C (B/(B+C)). This data is entered into the model with the category
custom. The ratio between B and C is then entered as an algebraic rule and the
variable describing this ratio, y, is given as a custom objective. The least
squares difference between y and the provided data is added to the objective
function.

"""
import sys

import kipet

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    kipet_model = kipet.KipetModel()
    kipet_model.settings.units.time = 's'
    kipet_model.reset_base_units()
    
    full_data = kipet_model.read_data_file('example_data/ratios.csv')
    r1 = kipet_model.new_reaction('reaction-1')
    
    # Add the model parameters
    r1.add_parameter('k1', value=2, bounds=(0.0, 10.0), units='1/s')
    r1.add_parameter('k2', value=0.2, bounds=(0.0, 10.0), units='1/s')
    
    # Declare the components and give the valueial values
    r1.add_component('A', value=1.0, units='mol/l')
    r1.add_component('B', value=0.0, units='mol/l')
    r1.add_component('C', value=0.0, units='mol/l')
    
    r1.add_alg_var('y', description='Ratio of B to B + C')
    
    r1.add_data('C_data', data=full_data[['A']], remove_negatives=True)
    r1.add_data('y_data', data=full_data[['y']])
    
    # r1.add_data('y_data', file='example_data/ratios.csv')
    r1.add_data('traj', file='example_data/extra_states.txt')
    
    c = r1.get_model_vars()
    
    r1.add_ode('A', -c.k1 * c.A )
    r1.add_ode('B', c.k1 * c.A - c.k2 * c.B )
    r1.add_ode('C', c.k2 * c.B )
    
    r1.add_algebraic('y', (c.B)/(c.B + c.C) )
    
    r1.check_model_units()
    # Add the custom objective varibale to the model using the following method:
    r1.add_objective_from_algebraic('y')
     
    r1.settings.general.no_user_scaling = True
    r1.settings.parameter_estimator.sim_init = False
    r1.settings.solver.linear_solver = 'ma57'
    
    r1.run_opt()
    
    r1.results.show_parameters
    
    if with_plots:
        r1.plot()