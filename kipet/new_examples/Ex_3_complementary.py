"""Example 3: Simulation using complementatry states with new KipetModel"""

# Standard library imports
import sys # Only needed for running the example from the command line

# Third party imports
from pyomo.core import exp

# Kipet library imports
from kipet import KipetModel

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False

    kipet_model = KipetModel()
    
    r1 = kipet_model.new_reaction('reaction-1')
    
    # Declare the components and give the initial values
    r1.add_component('A', state='concentration', init=1.0)
    r1.add_component('B', state='concentration', init=0.0)
    r1.add_component('C', state='concentration', init=0.0)
    r1.add_component('T', state='state', init=290)
    r1.add_component('V', state='state', init=100)
    
    # Define the ODEs
    def rule_odes(m,t):
        k1 = 1.25*exp((9500/1.987)*(1/320.0-1/m.X[t,'T']))
        k2 = 0.08*exp((7000/1.987)*(1/290.0-1/m.X[t,'T']))
        ra = -k1*m.Z[t,'A']
        rb = 0.5*k1*m.Z[t,'A']-k2*m.Z[t,'B']
        rc = 3*k2*m.Z[t,'B']
        cao = 4.0
        vo = 240
        T1 = 35000*(298-m.X[t,'T'])
        T2 = 4*240*30.0*(m.X[t,'T']-305.0)
        T3 = m.X[t,'V']*(6500.0*k1*m.Z[t,'A']-8000.0*k2*m.Z[t,'B'])
        Den = (30*m.Z[t,'A']+60*m.Z[t,'B']+20*m.Z[t,'C'])*m.X[t,'V']+3500.0
        exprs = dict()
        exprs['A'] = ra+(cao-m.Z[t,'A'])/m.X[t,'V']
        exprs['B'] = rb-m.Z[t,'B']*vo/m.X[t,'V']
        exprs['C'] = rc-m.Z[t,'C']*vo/m.X[t,'V']
        exprs['T'] = (T1+T2+T3)/Den
        exprs['V'] = vo
        return exprs
    
    r1.add_equations(rule_odes)
    r1.set_times(0.0, 2.0)
    
    r1.settings.collocation.nfe = 20
    r1.settings.collocation.ncp = 1

    r1.simulate()  

    if with_plots:
        r1.results.plot()