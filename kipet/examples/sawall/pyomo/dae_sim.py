#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 1 (From Sawall et.al.)
# Basic simulation of ODE system using pyomo discretization and IPOPT
#
#		\frac{dZ_a}{dt} = -k*Z_a	Z_a(0) = 1
#		\frac{dZ_b}{dt} = k*Z_a		Z_b(0) = 0


from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

    
    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False

    
    # create template model 
    builder = TemplateBuilder()    
    builder.add_mixture_component('A',1)
    builder.add_mixture_component('B',0)
    builder.add_algebraic_variable('ra')
    builder.add_parameter('k',0.01)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.Y[t,'ra']
        exprs['B'] = m.Y[t,'ra']
        return exprs

    builder.set_odes_rule(rule_odes)

    def rule_algebraics(m,t):
        algebraics = list()
        algebraics.append(m.Y[t,'ra']-m.P['k']*m.Z[t,'A'])
        return algebraics
    builder.set_algebraics_rule(rule_algebraics)

    
    # create an instance of a pyomo model template
    # the template includes
    #      - Z variables indexed over time and components names e.g. m.Z[t,'A']
    #      - P parameters indexed over the parameter names e.g. m.P['k']
    pyomo_model = builder.create_pyomo_model(0.0,200.0)

    pyomo_model.pprint()
    # create instance of simulator
    simulator = PyomoSimulator(pyomo_model)
    # defines the discrete points wanted in the concentration profile
    simulator.apply_discretization('dae.collocation',nfe=200,ncp=3,scheme='LAGRANGE-RADAU')

    # simulate
    results_pyomo = simulator.run_sim('ipopt',tee=True)

    if with_plots:
        # display concentration results
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_pyomo.Y.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Algebraics")
        plt.title("Algebraics")
        
        plt.show()


