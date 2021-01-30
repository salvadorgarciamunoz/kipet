"""
Example 17: Using custom data in the objective

This example uses data representing the ratio of component B to the total of
B and C (B/(B+C)). This data is entered into the model with the category
custom. The ratio between B and C is then entered as an algebraic rule and the
variable describing this ratio, y, is given as a custom objective. The least
squares difference between y and the provided data is added to the objective
function.

"""
import kipet

kipet_model = kipet.KipetModel()
full_data = kipet_model.read_data_file('example_data/ratios.csv')
r1 = kipet_model.new_reaction('reaction-1')

# Add the model parameters
r1.add_parameter('k1', init=2.0, bounds=(0.0, 10.0))
r1.add_parameter('k2', init=0.2, bounds=(0.0, 10.0))

# Declare the components and give the initial values
r1.add_component('A', state='concentration', init=1.0)
r1.add_component('B', state='concentration', init=0.0)
r1.add_component('C', state='concentration', init=0.0)

r1.add_dataset(data=full_data[['A']], remove_negatives=True)
r1.add_dataset('y_data', category='custom', data=full_data[['y']])

r1.add_algebraic_variables('y', init=0.2, bounds=(0.0, 1.0))

c = r1.get_model_vars()

rates = {}
rates['A'] = -c.k1 * c.A
rates['B'] = c.k1 * c.A - c.k2 * c.B
rates['C'] = c.k2 * c.B

r1.add_odes(rates)

AE = {}
AE['y'] = (c.B + 1e-12)/(c.B + c.C + 1e-12)

r1.add_algebraics(AE)

# Add the custom objective varibale to the model using the following method:
r1.add_objective_from_algebraic('y')
 
r1.settings.general.no_user_scaling = True
r1.settings.solver.linear_solver = 'mumps'

r1.run_opt()
r1.results.show_parameters
r1.results.plot('Z')
r1.results.plot('Y')