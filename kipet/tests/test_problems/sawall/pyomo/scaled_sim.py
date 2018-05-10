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
    builder.add_parameter('k',0.01)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k']*m.Z[t,'A']
        exprs['B'] = m.P['k']*m.Z[t,'A']
        return exprs

    builder.set_odes_rule(rule_odes)

    # create an instance of a pyomo model template
    # the template includes
    #      - Z variables indexed over time and components names e.g. m.Z[t,'A']
    #      - P parameters indexed over the parameter names e.g. m.P['k']
    pyomo_model = builder.create_pyomo_model(0.0,200.0)
    
    # create instance of simulator
    simulator = PyomoSimulator(pyomo_model)
    # defines the discrete points wanted in the concentration profile
    simulator.apply_discretization('dae.collocation',nfe=30,ncp=1,scheme='LAGRANGE-RADAU')

    
    # simulate
    results_pyomo = simulator.run_sim('ipopt',tee=True)

    # second model to scale and initialize
    scaled_model = builder.create_pyomo_model(0.0,200.0)
    scaled_sim = PyomoSimulator(scaled_model)
    scaled_sim.apply_discretization('dae.collocation',nfe=30,ncp=1,scheme='LAGRANGE-RADAU')

    scaled_sim.initialize_from_trajectory('Z',results_pyomo.Z)
    scaled_sim.initialize_from_trajectory('dZdt',results_pyomo.dZdt)

    scaled_sim.scale_variables_from_trajectory('Z',results_pyomo.Z)
    scaled_sim.scale_variables_from_trajectory('dZdt',results_pyomo.dZdt)
    
    solver_options = dict()
    solver_options['nlp_scaling_method'] = 'user-scaling'
    #solver_options['bound_relax_factor'] = 0.0
    #solver_options['mu_init'] =  1e-6
    #solver_options['bound_push'] = 1e-6
    
    # simulate scaled model
    results_scaled = scaled_sim.run_sim('ipopt',
                                        tee=True,
                                        solver_opts = solver_options)
    
    if with_plots:
        # display concentration results
        results_scaled.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        # display concentration results
        results_scaled.dZdt.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("dZdt (mol/L)")
        plt.title("Concentration derivative Profile")
        plt.show()
