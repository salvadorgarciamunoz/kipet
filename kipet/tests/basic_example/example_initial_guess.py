from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.sim.CasadiSimulator import *
import matplotlib.pyplot as plt


if __name__ == "__main__":


    builder = TemplateBuilder()    
    builder.add_mixture_component('A',1)
    builder.add_mixture_component('B',0)
    builder.add_parameter('k',0.01)

    casadi_model = builder.create_casadi_model(0.0,200.0)
    
    casadi_model.diff_exprs['A'] = -casadi_model.P['k']*casadi_model.C['A']
    casadi_model.diff_exprs['B'] = casadi_model.P['k']*casadi_model.C['A']

    sim = CasadiSimulator(casadi_model)
    sim.apply_discretization('integrator',nfe=700)
    results_casadi = sim.run_sim("cvodes")
    
    ##########################################################
    
    builder2 = TemplateBuilder()    
    builder2.add_mixture_component('A',1)
    builder2.add_mixture_component('B',0)
    builder2.add_parameter('k',0.01)
    
    pyomo_model = builder2.create_pyomo_model(0.0,200.0)
    
    def diff_exprs(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k']*m.C[t,'A']
        exprs['B'] = m.P['k']*m.C[t,'A']
        return exprs

    def rule_mass_balances(m,t,k):
        exprs = diff_exprs(m,t)
        if t == m.start_time:
            return Constraint.Skip
        else:
            return m.dCdt[t,k] == exprs[k] 
    pyomo_model.mass_balances = Constraint(pyomo_model.time,
                                     pyomo_model.mixture_components,
                                     rule=rule_mass_balances)
    
    
    simulator = PyomoSimulator(pyomo_model)
    simulator.apply_discretization('dae.collocation',nfe=100,ncp=3,scheme='LAGRANGE-RADAU')
    
    # Provide good initial guess
    simulator.initialize_from_trajectory('C',results_casadi.C)

    results_pyomo = simulator.run_sim('ipopt',tee=True)
    plt.plot(results_pyomo.C)
    plt.xlabel("time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile")
    plt.show()
