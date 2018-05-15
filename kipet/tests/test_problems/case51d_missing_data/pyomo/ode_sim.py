#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 2 (From Sawall et.al.)
# First example from WF paper simulation of ODE system using pyomo discretization and IPOPT
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2*Z_b	                Z_c(0) = 0

from kipet.model.TemplateBuilder import *
from kipet.sim.PyomoSimulator import *
from kipet.utils.data_tools import read_absorption_data_from_txt
import matplotlib.pyplot as plt
import sys

#from casadi_ode_sim import results_casadi

import pickle

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False


    fixed_traj = read_absorption_data_from_txt('extra_states.txt')

    # create template model 
    builder = TemplateBuilder()

    # components
    components = dict()
    components['SA'] = 1.0714                  # Salicitilc acid
    components['AA'] = 9.3828               # Acetic anhydride
    components['ASA'] = 0.0177                 # Acetylsalicylic acid
    components['HA'] = 0.0177                  # Acetic acid
    components['ASAA'] = 0.0                # Acetylsalicylic anhydride
    components['H2O'] = 0.0                 # water

    builder.add_mixture_component(components)

    # add parameters
    params = dict()
    params['k1'] = 0.0360309
    params['k2'] = 0.1596062
    params['k3'] = 6.8032345
    params['k4'] = 1.8028763
    params['kc'] = 0.7566864
    params['kd'] = 7.1108682
    params['Csa'] = 2.06269996

    builder.add_parameter(params)

    # add additional state variables
    extra_states = dict()
    extra_states['Masa'] = 0.0
    extra_states['Msa'] = 9.537
    extra_states['V'] = 0.0202
    extra_states['T'] = 313
    extra_states['f'] = 0.0

    builder.add_complementary_state_variable(extra_states)

    model = builder.create_pyomo_model(0.0,220.5257)

    gammas = dict()
    gammas['SA']= [-1.0,0.0,0.0,0.0]
    gammas['AA']= [-1.0,-1.0,0.0,-1.0]
    gammas['ASA']= [1.0,-1.0,1.0,0.0]
    gammas['HA']= [1.0,1.0,1.0,2.0,0.0]
    gammas['ASAA']= [0.0,1.0,-1.0,0.0]
    gammas['H2O']= [0.0,0.0,-1.0,-1.0]

    partial_vol = dict()
    partial_vol['SA']=0.0952552311614
    partial_vol['AA']=0.101672206869
    partial_vol['ASA']=0.132335206093
    partial_vol['HA']=0.060320218688
    partial_vol['ASAA']=0.186550717015
    partial_vol['H2O']=0.0243603912169

    def vel_rxns(m,t):
        r = list()
        r.append(m.P['k1']*m.Z[t,'SA']*m.Z[t,'AA'])
        r.append(m.P['k2']*m.Z[t,'ASA']*m.Z[t,'AA'])
        r.append(m.P['k3']*m.Z[t,'ASAA']*m.Z[t,'H2O'])
        r.append(m.P['k4']*m.Z[t,'AA']*m.Z[t,'H2O'])
        return r

    # variables for the disolution and cristalization rates

    model.rc = Var(model.time,
                   bounds=(0.0,None),
                   initialize=1.0)


    def rule_rc(m,t):
        C_sat = 0.000403961838576*(m.X[t,'T']-273.15)**2 - 0.002335673472454*(m.X[t,'T']-273.15)+0.428791235875747
        C_asa = m.Z[t,'ASA']
        rhs = 0.5*m.P['kc']**0.7462686567*(C_asa-C_sat+((C_asa-C_sat)**2+1e-6)**0.5)
        return m.rc[t] == 0.3950206559*m.P['kc']*(C_asa-C_sat+((C_asa-C_sat)**2+1e-6)**0.5)**1.34
        #return m.rc[t]**0.7462686567 == rhs
    model.rc_constraint = Constraint(model.time,rule=rule_rc)


    model.rd = Var(model.time,
                   bounds=(0.0,None),
                   initialize=1.0)


    def rule_rd(m,t):

        C_sat = m.P['Csa']
        C_sa = m.Z[t,'SA']
        m_sa = m.X[t,'Msa']
        #step = 0.5*(1+m_sa/(m_sa**2+1e-2**2)**0.5)
        step = 1.0/(1.0+exp(-m_sa/1e-3))
        return m.rd[t] == 0.0 #m.P['kd']*(C_sat-C_sa)**2.0*step

    model.rd_constraint = Constraint(model.time,rule=rule_rd)



    def mass_balances(m,t):
        r = vel_rxns(m,t)
        exprs = dict()
        V = m.X[t,'V']
        exprs['SA'] = -V*(r[0]+m.rd[t]) - m.dXdt[t,'V']*m.Z[t,'SA']
        exprs['AA'] = -V*(r[0]+r[1]+r[3]) - m.dXdt[t,'V']*m.Z[t,'AA']
        exprs['ASA'] = V*(r[0]-r[1]+r[2]-m.rc[t]) - m.dXdt[t,'V']*m.Z[t,'ASA']
        exprs['HA'] = V*(r[0]+r[1]+r[2]+2*r[3]) - m.dXdt[t,'V']*m.Z[t,'HA']
        exprs['ASAA'] = V*(r[1]-r[2]) - m.dXdt[t,'V']*m.Z[t,'ASAA']
        exprs['H2O'] = V*(-r[2]-r[3]) - m.dXdt[t,'V']*m.Z[t,'H2O']
        return exprs

    def rule_odes(m,t,k):
        exprs = mass_balances(m,t)
        if t == m.start_time.value:
            return Constraint.Skip
        else:
            return m.dZdt[t,k]*m.X[t,'V'] == exprs[k]

    model.odes = Constraint(model.time,
                            model.mixture_components,
                            rule=rule_odes)

    # deal with additional states
    def rule_volume(m,t):
        r = vel_rxns(m,t)
        vol_sum = 0.0
        for c in m.mixture_components:
            vol_sum += partial_vol[c]*sum(gammas[c][j]*r_val for j,r_val in enumerate(r))

        return m.dXdt[t,'V'] == m.X[t,'V']*vol_sum

    model.volume = Constraint(model.time,
                              rule=rule_volume)

    def rule_temperature(m,t):
        return m.dXdt[t,'T'] == 0.0

    model.temperature = Constraint(model.time,
                                   rule=rule_temperature)


    def rule_Masa(m,t):
        PM = 180.157
        return m.dXdt[t,'Masa'] == PM*m.X[t,'V']*m.rc[t]

    model.Masa = Constraint(model.time,
                            rule=rule_Masa)

    def rule_Msa(m,t):
        PM = 138.121
        return m.dXdt[t,'Msa'] == -PM*m.X[t,'V']*m.rd[t]

    model.Msa = Constraint(model.time,
                           rule=rule_Msa)

    simulator = PyomoSimulator(model)
    # defines the discrete points wanted in the concentration profile
    simulator.apply_discretization('dae.collocation',nfe=100,ncp=3,scheme='LAGRANGE-RADAU')

    #with open('init2.pkl', 'rb') as f:
    #    results_casadi = pickle.load(f)

    #simulator.initialize_from_trajectory('Z',results_casadi.Z)
    #simulator.initialize_from_trajectory('X',fixed_traj)

    # fixes the flow
    #for t in model.time:
    #    model.X[t,'f'].fixed = True

    options = {'halt_on_ampl_error' :'yes'}
    results_pyomo = simulator.run_sim('ipopt',
                                      tee=True,
                                      solver_opts=options)
    if with_plots:
        # display concentration results

        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        plt.figure()

        results_pyomo.X['V'].plot.line()
        plt.xlabel("time (s)")
        plt.ylabel("volumne (L)")
        plt.title("Volume Profile")

        plt.figure()
        results_pyomo.X['T'].plot.line()
        plt.xlabel("time (s)")
        plt.ylabel("Temperature (K)")
        plt.ylim([300,350])
        plt.title("Temperature Profile")


        plt.figure()

        results_pyomo.X['Masa'].plot.line()
        plt.xlabel("time (s)")
        plt.ylabel("m_dot (g)")
        plt.title("Masa Profile")


        plt.figure()

        results_pyomo.X['Msa'].plot.line()
        plt.xlabel("time (s)")
        plt.ylabel("m_dot (g)")
        plt.title("Msa Profile")

        plt.show()
