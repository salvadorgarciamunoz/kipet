"""
Example 20: Fedbatch example using step function for adding a component
"""
# Standard library imports
import sys

# Third party imports

# Kipet library imports
from kipet import KipetModel


if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False

    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('fed_batch_parest')
    
    r1.add_parameter('k1', init = 0.1)
    
    r1.add_component('A', state = 'concentration', init = 2.0, units='mol/L')
    r1.add_component('B', state = 'concentration', init = 0.0, units='mol/L')
    r1.add_component('C', state = 'concentration', init = 0.0, units='mol/L')
    
    r1.add_component('V', state = 'state', init = 1.0)
    
    filename = 'example_data/abc_fedbatch.csv'
    r1.add_dataset('C_data', category='concentration', file = filename)
    
    # Step function for B feed
    r1.add_step('s_Qin_B', coeff=1, time=15, switch='off')
    
    # Volumetric flow rate for B feed
    r1.add_constant('Qin_B', value=0.1, units='L/min')
    
    # Concentration of B in feed
    r1.add_constant('Cin_B', value=2.0, units='mol/L')
    
    c = r1.get_model_vars()

    # c now holds of all of the pyomo variables needed to define the equations
    # Using this object allows for a much simpler construction of expressions
    R1 = c.k1*c.A*c.B
    Qin_B = c.Qin_B*c.s_Qin_B
    QV = Qin_B/c.V
    
    rates = {}
    rates['A'] = -c.A*QV - R1
    rates['B'] = (c.Cin_B - c.B)*QV - R1
    rates['C'] = -c.C*QV + R1
    rates['V'] = Qin_B
    
    r1.add_equations(rates)
    
    r1.settings.solver.linear_solver = 'ma27'
    r1.settings.parameter_estimator.sim_init = True
    
    r1.run_opt()
    
    r1.results.plot(show_plot=with_plots)