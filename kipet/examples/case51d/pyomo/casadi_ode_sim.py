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
import matplotlib.pyplot as plt
import numpy as np
import sys
    
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
builder.add_complementary_state_variable(extra_states)

gammas = dict()
gammas['SA']= [-1.0,0.0,0.0,0.0]
gammas['AA']= [-1.0,-1.0,0.0,-1.0]
gammas['ASA']= [1.0,-1.0,1.0,0.0]
gammas['HA']= [1.0,1.0,1.0,2.0]
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

def rule_odes(m,t):
    r = vel_rxns(m,t)
    exprs = dict()
    
    vol_sum = 0.0
    for c in m.mixture_components:
        vol_sum += partial_vol[c]*sum(gammas[c][j]*r_val for j,r_val in enumerate(r)) 
        
    exprs['V'] = m.X[t,'V']*vol_sum
    exprs['SA'] = -r[0] - exprs['V']/m.X[t,'V']*m.Z[t,'SA']
    exprs['AA'] = -r[0]-r[1]-r[3] - exprs['V']/m.X[t,'V']*m.Z[t,'AA']
    exprs['ASA'] = r[0]-r[1]+r[2] - exprs['V']/m.X[t,'V']*m.Z[t,'ASA']
    exprs['HA'] = r[0]+r[1]+r[2]+2*r[3] - exprs['V']/m.X[t,'V']*m.Z[t,'HA']
    exprs['ASAA'] = r[1]-r[2] - exprs['V']/m.X[t,'V']*m.Z[t,'ASAA']
    exprs['H2O'] = -r[2]-r[3] - exprs['V']/m.X[t,'V']*m.Z[t,'H2O']
    return exprs

builder.set_odes_rule(rule_odes)
#casadi_model = builder.create_casadi_model(0.0,220.5257)    
casadi_model = builder.create_casadi_model(0.0,3.0)    

sim = CasadiSimulator(casadi_model)
# defines the discrete points wanted in the concentration profile
sim.apply_discretization('integrator',nfe=5)
# simulate
results_casadi = sim.run_sim("cvodes")

print results_casadi.Z['SA']

sim2 = CasadiSimulator(casadi_model)
# defines the discrete points wanted in the concentration profile
sim2.apply_discretization('integrator',nfe=3)
sim2.fix_from_trajectory('Z','SA',results_casadi.Z)
print sim2._fixed_variables
print sim2._fixed_trajectories

# simulate
results_casadi2 = sim2.run_sim("cvodes")


