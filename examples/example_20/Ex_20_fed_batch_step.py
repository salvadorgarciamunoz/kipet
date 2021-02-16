"""
Example 20: Fedbatch example using step function for adding a component

Uses mixed units to show the unit checking feature
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
    
    # Set the base time unit (match data)
    kipet_model.ub.TIME_BASE = 'minute'
    
    r1 = kipet_model.new_reaction('fed_batch_parest')
    
    # Reaction rate constant (parameter to fit)
    k1 = r1.parameter('k1', value = 0.05, units='ft**3/mol/min')

    # Components
    A = r1.component('A', value=2.0, units='mol/L')
    B = r1.component('B', value=0.0, units='mol/L')
    C = r1.component('C', value=0.0, units='mol/L')
    
    # Reactor volume
    V = r1.state('V', value = 0.264172, units='gal')
    
    # Step function for B feed - steps can be added
    s_Qin_B = r1.step('s_Qin_B', coeff=1, time=15, switch='off')
    
    # Volumetric flow rate of the feed
    Qin_B = r1.constant('Qin_B', value=6, units='L/hour')
    
    # Concentration of B in feed
    Cin_B = r1.constant('Cin_B', value=2.0, units='mol/L')
    
    # Add the data
    filename = 'data/abc_fedbatch.csv'
    r1.add_data('C_data', file=filename, units='mol/L', remove_negatives=True)
    
    # Convert your model components to a common base
    # KIPET assumes that the provided data has the same units and will be
    # converted as well - be careful!
    r1.check_component_units()
    
    R1 = k1*A*B
    Qin_B = Qin_B*(s_Qin_B)
    QV = Qin_B/V
    
    r1.add_ode('A', -A*QV - R1 )
    r1.add_ode('B', (Cin_B - B)*QV - R1 )
    r1.add_ode('C', -C*QV + R1)
    r1.add_ode('V', Qin_B)
    
    # Check for consistant units in the model equations
    r1.check_model_units(display=True)

    r1.settings.solver.linear_solver = 'ma57'
    r1.settings.parameter_estimator.sim_init = True
    
    r1.run_opt()
    
    if with_plots:
        r1.plot()
