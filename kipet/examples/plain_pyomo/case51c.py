# Basic simulation of ODE with spectral data using pyomo discretization 
#
#		\frac{dZ_a}{dt} = -k_1*Z_a*Z_b	                                Z_a(0) = 1
#		\frac{dZ_b}{dt} = -k_1*Z_a*Z_b                   		Z_b(0) = 0.8
#               \frac{dZ_c}{dt} = k_1*Z_a*Z_b-2*k_2*Z_c^2	                Z_c(0) = 0
#               \frac{dZ_d}{dt} = k_2*Z_c^2             	                Z_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 

from pyomo.dae import *
from pyomo.environ import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import inspect
import sys
import os

# load spectral data
dataDirectory = os.path.abspath(
    os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
    inspect.currentframe() ) ) ), '..','data_sets'))
filename =  os.path.join(dataDirectory,'Dij_case51c.csv')
spectral_data = pd.read_csv(filename,index_col=0)

model = ConcreteModel()

# define sets
model.components = Set(initialize=['A','B','C','D'])

model.meas_times = Set(initialize = spectral_data.index,
                       ordered=True)

model.meas_lambdas = Set(initialize = spectral_data.columns,
                         ordered=True)

start_time = 0.0
end_time = 10.0
model.time = ContinuousSet(initialize=model.meas_times,
                           bounds = (start_time,end_time))

# variables
model.k1 = Var(bounds = (0.0,5.0),
              initialize=1)

model.k2 = Var(bounds = (0.0,5.0),
              initialize=1)

model.Z = Var(model.time,
              model.components,
              initialize=1)

model.dZdt = DerivativeVar(model.Z,
                           wrt=model.time)


model.C = Var(model.meas_times,
              model.components,
              bounds=(0.0,None),
              initialize=1.0)

model.S = Var(model.meas_lambdas,
              model.components,
              bounds=(0.0,None),
              initialize=1.0)

# parametes
s_data_dict = dict()
for t in model.meas_times:
    for l in model.meas_lambdas:
        s_data_dict[t,l] = float(spectral_data[l][t])

model.D = Param(model.meas_times,
                model.meas_lambdas,
                initialize = s_data_dict)


# constraints
def mass_balances(m,t):
    exprs = dict()
    exprs['A'] = -m.k1* m.Z[t,'A']*m.Z[t,'B']
    exprs['B'] = -m.k1* m.Z[t,'A']*m.Z[t,'B']
    exprs['C'] = m.k1*m.Z[t,'A']*m.Z[t,'B']-2*m.k2*m.Z[t,'C']**2
    exprs['D'] = m.k2*m.Z[t,'C']**2
    return exprs

def rule_odes(m,t,k):
    exprs = mass_balances(m,t)
    if t == start_time:
        return Constraint.Skip
    else:
        return m.dZdt[t,k] == exprs[k]
                    
model.odes = Constraint(model.time,
                        model.components,
                        rule=rule_odes)
    
init_conditions = {'A':1.0,'B':0.8,'C':0.0,'D':0.0}
def rule_init_conditions(m,k):
    st = start_time
    return m.Z[st,k] == init_conditions[k]
    
model.init_conditions_c = Constraint(model.components,rule=rule_init_conditions)

variances = {'device':1.66507e-6,
              'A':4.34682e-6,
              'B':2.62689e-6,
              'C':5.31808e-6,
              'D':2.72712e-6}


discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(model,wrt=model.time,nfe=30,ncp=1,scheme='LAGRANGE-RADAU')
opt = SolverFactory('ipopt')

# simulate first

model.k1.value = 8.0
model.k1.fixed = True
model.k2.value = 4.0
model.k2.fixed = True

solver_results = opt.solve(model,tee=True)

model.k1.fixed = False
model.k2.fixed = False

def rule_objective(m):
    expr = 0
    for t in m.meas_times:
        for l in m.meas_lambdas:
            D_bar = sum(m.C[t,k]*m.S[l,k] for k in m.components)
            expr+= (m.D[t,l] - D_bar)**2/variances['device']

    for t in m.meas_times:
        expr += sum((m.C[t,k]-m.Z[t,k])**2/variances[k] for k in m.components)
    return expr

model.objective = Objective(rule=rule_objective)


solver_results = opt.solve(model,tee=True)

# ploting solution profiles
nt = len(model.meas_times)
nw = len(model.meas_lambdas)
nc = len(model.components)

v_values = np.zeros((nt,nc))
for i,t in enumerate(model.meas_times):
    for j,k in enumerate(model.components):
        v_values[i,j] = model.C[t,k].value
        
C = pd.DataFrame(data=v_values,
                 columns = model.components,
                 index=model.meas_times)

v_values = np.zeros((nw,nc))
for i,l in enumerate(model.meas_lambdas):
    for j,k in enumerate(model.components):
        v_values[i,j] = model.S[l,k].value
        
S = pd.DataFrame(data=v_values,
                 columns = model.components,
                 index=model.meas_lambdas)

S.plot.line()
C.plot.line()

plt.show()
