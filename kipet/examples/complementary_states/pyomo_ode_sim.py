#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 
# Simulation of ODE system using pyomo discretization and IPOPT
#
#		\frac{dZ_a}{dt} = -k1(T)*Z_	           Z_a(0) = 1.0
#		\frac{dZ_b}{dt} = 0.5*k1(T)*Z_a-k2(T)*Z_b  Z_b(0) = 0.0
#		\frac{dZ_c}{dt} = 3*k(T)*Z_b               Z_c(0) = 0.0
#               \frac{dT}{dt} = heat tranference           T(0) = 290.0
#               \frac{dV}{dt} = 100+240*t                  V(0) = 100.0

from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
import matplotlib.pyplot as plt
from pyomo.core.base.expr import Expr_if
from pyomo.core import *

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False

    
    # create template model 
    builder = TemplateBuilder()    
    builder.add_mixture_component('A',1.0)
    builder.add_mixture_component('B',0.0)
    builder.add_mixture_component('C',0.0)

    builder.add_complementary_state_variable('T',290.0)
    builder.add_complementary_state_variable('V',100.0)
    
    
    # define explicit system of ODEs
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

    builder.set_odes_rule(rule_odes)
    
    # create an instance of a pyomo model template
    # the template includes
    #      - Z variables indexed over time and components names e.g. m.Z[t,'A']
    #      - P parameters indexed over the parameter names e.g. m.P['k']
    pyomo_model = builder.create_pyomo_model(0.0,2.0)

    pyomo_model.pprint()
    # create instance of simulator
    simulator = PyomoSimulator(pyomo_model)
    # defines the discrete points wanted in the concentration profile
    simulator.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')

    # simulate
    results_pyomo = simulator.run_sim('ipopt',tee=True)

    if with_plots:
        # display concentration results
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        
        results_pyomo.X.plot.line(legend=True)
        plt.ylim([250,450])
        plt.xlabel("time (s)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature Profile")
        
        plt.show()
