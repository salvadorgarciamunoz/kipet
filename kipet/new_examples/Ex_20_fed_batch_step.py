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
    
    r1.add_component('A', state = 'concentration', init = 2.0)
    r1.add_component('B', state = 'concentration', init = 0.0)
    r1.add_component('C', state = 'concentration', init = 0.0)
    
    r1.add_component('V', state = 'state', init = 1.0)
    
    filename = 'example_data/abc_fedbatch.csv'
    r1.add_dataset('C_data', category='concentration', file = filename)
    
    cin_B = 2
    # Add the step function for the B feed
    # on = False means turn it off at time = 15, True would mean turning it on at 15
    r1.add_step('Qin_B', time=15, switch='off')
    
    def rule_odes(m,t):
        exprs = dict()
        # Use the step variable here
        Qin_B = 0.1*m.step[t, 'Qin_B']
        exprs['A'] = -m.Z[t,'A'] * Qin_B/m.X[t,'V']-m.P['k1']*m.Z[t,'A']*m.Z[t,'B']
        exprs['B'] = (cin_B*Qin_B - m.Z[t,'B'] * Qin_B)/m.X[t,'V']- m.P['k1']*m.Z[t,'A']*m.Z[t,'B']
        exprs['C'] = -m.Z[t,'C'] * Qin_B/m.X[t,'V'] + m.P['k1']*m.Z[t,'A']*m.Z[t,'B']
        exprs['V'] = Qin_B
        return exprs
    
    r1.add_equations(rule_odes)
    
    r1.settings.solver.linear_solver = 'ma27'
    r1.settings.parameter_estimator.sim_init = True
    
    r1.run_opt()
    
    r1.results.plot(show_plot=with_plots)