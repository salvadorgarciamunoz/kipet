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
from kipet.sim.CasadiSimulator import *
from kipet.utils.data_tools import *
import matplotlib.pyplot as plt
import numpy as np
import sys

import pickle

if __name__ == "__main__":
    

    fixed_traj = read_absorption_data_from_txt('extra_states.txt')
    C = read_absorption_data_from_txt('concentrations.txt')

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
    extra_states['V'] = 0.0202
    extra_states['T'] = 273
    extra_states['f'] = 0.0
    extra_states['Masa'] = 0.0
    extra_states['Msa'] = 9.537
    
    builder.add_complementary_state_variable(extra_states)

    gammas = dict()
    gammas['SA']= [-1.0,0.0,0.0,0.0,0.0,1.0]
    gammas['AA']= [-1.0,-1.0,0.0,-1.0,0.0,0.0]
    gammas['ASA']= [1.0,-1.0,1.0,0.0,-1.0,0.0]
    gammas['HA']= [1.0,1.0,1.0,2.0,0.0,0.0,0.0]
    gammas['ASAA']= [0.0,1.0,-1.0,0.0,0.0,0.0]
    gammas['H2O']= [0.0,0.0,-1.0,-1.0,0.0,0.0]


    epsilon = dict()
    epsilon['SA']= 0.0
    epsilon['AA']= 0.0
    epsilon['ASA']= 0.0
    epsilon['HA']= 0.0
    epsilon['ASAA']= 0.0
    epsilon['H2O']= 1.0
    
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

        # cristalization rate
        C_sat = 0.000403961838576*(m.X[t,'T']-273.15)**2 - 0.002335673472454*(m.X[t,'T']-273.15)+0.428791235875747
        C_asa = m.Z[t,'ASA']
        rc = 0.3950206559*m.P['kc']*(C_asa-C_sat+((C_asa-C_sat)**2+1e-6)**0.5)**1.34
        r.append(rc)
        # disolution rate
        C_sat = m.P['Csa']
        C_sa = m.Z[t,'SA']
        m_sa = m.X[t,'Msa']
        #step = 0.5*(1+m_sa/(m_sa**2+1e-2**2)**0.5)
        step = 1.0/(1.0+ca.exp(-m_sa/1e-6))
        rd = m.P['kd']*(C_sat-C_sa)**1.90*step
        r.append(rd)
        
        return r

    def rule_odes(m,t):
        r = vel_rxns(m,t)
        exprs = dict()

        V = m.X[t,'V']
        f = m.X[t,'f']
        vol_sum = 0.0
        for c in m.mixture_components:
            vol_sum += partial_vol[c]*(sum(gammas[c][j]*r_val for j,r_val in enumerate(r))+ epsilon[c]*f/V)
        
        exprs['V'] = V*vol_sum 
        exprs['SA'] = -r[0]+r[5] - exprs['V']/V*m.Z[t,'SA']
        exprs['AA'] = -r[0]-r[1]-r[3] - exprs['V']/V*m.Z[t,'AA']
        exprs['ASA'] = r[0]-r[1]+r[2]-r[4] - exprs['V']/V*m.Z[t,'ASA']
        exprs['HA'] = r[0]+r[1]+r[2]+2*r[3] - exprs['V']/V*m.Z[t,'HA']
        exprs['ASAA'] = r[1]-r[2] - exprs['V']/V*m.Z[t,'ASAA']
        exprs['H2O'] = -r[2]-r[3] - exprs['V']/V*m.Z[t,'H2O']
        exprs['Masa'] = 180.157*V*r[4]
        exprs['Msa'] = -138.121*V*r[5]
        return exprs

    builder.set_odes_rule(rule_odes)
    casadi_model = builder.create_casadi_model(0.0,210.5257)    

    #print casadi_model.odes
    
    sim = CasadiSimulator(casadi_model)
    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('integrator',nfe=400)
    # simulate
    sim.fix_from_trajectory('X','T',fixed_traj)
    sim.fix_from_trajectory('X','f',fixed_traj*1e6)
    results_casadi = sim.run_sim("cvodes")
    
    # display concentration results

    #with open('init.pkl', 'wb') as f:
        #pickle.dump(results_casadi, )
    
    print results_casadi.X['Msa']
    
    results_casadi.Z.plot.line(legend=True)
    plt.xlabel("time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.title("Concentration Profile")

    plt.figure()
    
    results_casadi.X['V'].plot.line()
    plt.xlabel("time (s)")
    plt.ylabel("volumne (L)")
    plt.title("Volume Profile")
    
    plt.figure()
    results_casadi.X['T'].plot.line()
    plt.xlabel("time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature Profile")

    print results_casadi.X['T']

    plt.figure()
    results_casadi.X['f'].plot.line()
    plt.xlabel("time (s)")
    plt.ylabel("flow (K)")
    plt.title("Inlet flow Profile")
    
    plt.figure()
    
    results_casadi.X['Masa'].plot.line()
    plt.xlabel("time (s)")
    plt.ylabel("m_dot (g)")
    plt.title("Masa Profile")

    plt.figure()
    
    results_casadi.X['Msa'].plot.line()
    plt.xlabel("time (s)")
    plt.ylabel("m_dot (g)")
    plt.title("Msa Profile")
    
    plt.show()
    

