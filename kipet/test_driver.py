from model.PyomoModelBuilderHelper import *
from sim.PyomoSimulator import *

from model.CasadiModelBuilderHelper import *
from sim.CasadiSimulator import *

#from BaseAbstractModel import BaseModel
import matplotlib.pyplot as plt


if __name__ == "__main__":

    helper2 = CasadiModelBuilderHelper()    
    helper2.add_mixture_component('A',1)
    helper2.add_mixture_component('B',0)
    helper2.add_kinetic_parameter('k')

    casadi_model = helper2.create_casadi_model(0.0,200.0)
    
    casadi_model.diff_exprs = dict()
    casadi_model.diff_exprs['A'] = -casadi_model.kinetic_parameter['k']*casadi_model.C['A']
    casadi_model.diff_exprs['B'] = casadi_model.kinetic_parameter['k']*casadi_model.C['A']


    # fixes parameters
    fixed_params = dict()
    fixed_params[casadi_model.kinetic_parameter] = 0.01
    
    for key,val in casadi_model.diff_exprs.iteritems():
        for key1,val1 in fixed_params.iteritems():
            casadi_model.diff_exprs[key] = ca.substitute(val,key1,val1)

    print casadi_model.diff_exprs

    sim = CasadiSimulator(casadi_model)
    sim.apply_discretization('integrator',nfe=700)
    sim_results = sim.run_sim("cvodes")
    
    ##########################################################

    helper = PyomoModelBuilderHelper()
    
    helper.add_mixture_component('A',1)
    helper.add_mixture_component('B',0)
    helper.add_kinetic_parameter('k')
    
    fix_dict = {'k':0.01}
    model = helper.create_pyomo_concrete_model(0.0,200.0,fixed_dict=fix_dict)
    
    def rule_mass_A(m,t):
        if t == m.start_time:
            return Constraint.Skip
        else:
            return m.dCdt[t,'A']== -m.kinetic_parameter['k']*m.C[t,'A']
    model.mass_balance_A = Constraint(model.time,rule=rule_mass_A)
    
    def rule_mass_B(m,t):
        if t == m.start_time:
            return Constraint.Skip
        else:
            return m.dCdt[t,'B']== m.kinetic_parameter['k']*m.C[t,'A']
    model.mass_balance_B = Constraint(model.time,rule=rule_mass_B)

    model.pprint()
    
    ##########################################################
    
    
    simulator = PyomoSimulator(model)
    simulator.apply_discretization('dae.collocation',nfe=100,ncp=3,scheme='LAGRANGE-RADAU')
    
    
    simulator.initialize_from_trajectory('C',sim_results.panel['concentration'])

    results = simulator.run_sim('ipopt',tee=True)
    plt.plot(results.panel['concentration'])
    
