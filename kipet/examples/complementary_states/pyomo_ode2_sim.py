#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 
# Simulation of ODE system using pyomo discretization and IPOPT
#
#		\frac{dZ_a}{dt} = -k(T)*Z_a*Z_b	        Z_a(0) = 6.7
#		\frac{dZ_b}{dt} = -k(T)*Z_a*Z_b		Z_b(0) = 20.2
#		\frac{dZ_c}{dt} = 2*k(T)*Z_a*Z_b        Z_c(0) = 0.0
#               \frac{dT}{dt} = Tedot + Tsdot           T(0) = 293.2



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
    builder.add_mixture_component('A',6.7)
    builder.add_mixture_component('B',20.2)
    builder.add_mixture_component('C',0.0)
    
    builder.add_complementary_state_variable('T',290.0)
    
    builder.add_parameter('k_p',3.734e7)

    # define explicit system of ODEs
    def rule_odes(m,t):
        r = -m.P['k_p']*exp(-15400.0/(1.987*m.X[t,'T']))*m.Z[t,'A']*m.Z[t,'B']
        T1 = 45650.0*(-r*0.01)/28.0
        T2 = Expr_if(IF=m.X[t,'T']>328.0, THEN=0.0, ELSE=2.0)#ca.if_else(m.X[t,'T']>328.0,0.0,2.0)
        exprs = dict()
        exprs['A'] = r
        exprs['B'] = 0.0
        exprs['C'] = -r
        exprs['T'] = T1+T2
        return exprs

    builder.set_odes_rule(rule_odes)
    
    # create an instance of a pyomo model template
    # the template includes
    #      - Z variables indexed over time and components names e.g. m.Z[t,'A']
    #      - P parameters indexed over the parameter names e.g. m.P['k']
    pyomo_model = builder.create_pyomo_model(0.0,20.0)

    pyomo_model.pprint()
    # create instance of simulator
    simulator = PyomoSimulator(pyomo_model)
    # defines the discrete points wanted in the concentration profile
    simulator.apply_discretization('dae.collocation',nfe=100,ncp=3,scheme='LAGRANGE-RADAU')

    # simulate
    results_pyomo = simulator.run_sim('ipopt',tee=True)

    if with_plots:
        # display concentration results
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        
        results_pyomo.X.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature Profile")
        
        plt.show()
