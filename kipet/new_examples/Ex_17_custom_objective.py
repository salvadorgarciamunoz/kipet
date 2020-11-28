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
r1.add_parameter('k1', init=5.0, bounds=(0.0, 10.0))
r1.add_parameter('k2', init=5.0, bounds=(0.0, 10.0))

# Declare the components and give the initial values
r1.add_component('A', state='concentration', init=1.0)
r1.add_component('B', state='concentration', init=0.0)
r1.add_component('C', state='concentration', init=0.0)

r1.add_dataset(data=full_data[['A']], remove_negatives=True)
r1.add_dataset('y_data', category='custom', data=full_data[['y']])

# Define the reaction model
def rule_odes(m,t):
    exprs = dict()
    exprs['A'] = -m.P['k1']*m.Z[t,'A']
    exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
    exprs['C'] = m.P['k2']*m.Z[t,'B']
    return exprs 

r1.add_equations(rule_odes)

# To use custom objective terms for special data, define the variable as an
# algegraic and provide the relationship between components
r1.add_algebraic_variables('y', init = 0.0, bounds = (0.0, 1.0))

def rule_algebraics(m, t):
    r = list()
    r.append(m.Y[t, 'y']*(m.Z[t, 'B'] + m.Z[t, 'C']) - m.Z[t, 'B'])
    return r

r1.add_algebraics(rule_algebraics)

# Add the custom objective varibale to the model using the following method:
r1.add_objective_from_algebraic('y')
 
r1.run_opt()
r1.results.show_parameters
r1.results.plot('Z')
r1.results.plot('Y')