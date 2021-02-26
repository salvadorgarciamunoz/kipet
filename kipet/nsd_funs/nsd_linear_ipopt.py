#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:02:58 2020

@author: kevin
"""
import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import *

from kipet.nsd_funs.NSD_TrustRegion_Ipopt import NSD


""" Model 1 """

m = ConcreteModel()

x_data = dict(zip(list(range(1, 6)), list(range(1, 6))))
y_data = dict(zip(list(range(1, 6)), [2, 4, 6, 8, 10]))

m.indx = Set(initialize=list(range(1, 6)))

m.X = Param(m.indx, initialize=x_data)
m.Y = Param(m.indx, initialize=y_data)

m.parameter_names = Set(initialize=[1, 2])

m.P = Var(m.parameter_names, bounds=(-5, 5), initialize=1)

m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

@m.Objective(sense=minimize)
def objective(m):
    obj = 0
    for i in m.indx:
        obj += (m.P[1]*m.X[i] + m.P[2] - m.Y[i])**2
        
    return obj

""" Model 2 """

m2 = ConcreteModel()

x_data2 = dict(zip(list(range(1, 6)), list(range(1, 6))))
y_data2 = dict(zip(list(range(1, 6)), [1.50, 4.45, 6.47, 8.60, 10.96]))

m2.indx = Set(initialize=list(range(1, 6)))

m2.X = Param(m2.indx, initialize=x_data2)
m2.Y = Param(m2.indx, initialize=y_data2)

m2.parameter_names = Set(initialize=[1, 2])

m2.P = Var(m2.parameter_names, bounds=(-5, 5), initialize=1)

m2.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

@m2.Objective(sense=minimize)
def objective(m2):
    obj = 0
    for i in m2.indx:
        obj += (m2.P[1]*m2.X[i] + m2.P[2] - m2.Y[i])**2
        
    return obj

""" Model 3 """

m3 = ConcreteModel()

x_data3 = dict(zip(list(range(1, 6, 2)), list(range(1, 6, 2))))
y_data3 = dict(zip(list(range(1, 6, 2)), [4.67, 10.47, 15.23]))

m3.indx = Set(initialize=list(range(1, 6, 2)))

m3.X = Param(m3.indx, initialize=x_data3)
m3.Y = Param(m3.indx, initialize=y_data3)

m3.parameter_names = Set(initialize=[1, 2])

m3.P = Var(m3.parameter_names, bounds=(-5, 5), initialize=1)

m3.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

@m3.Objective(sense=minimize)
def objective(m3):
    obj = 0
    for i in m3.indx:
        obj += (m3.P[1]*m3.X[i] + m3.P[2] - m3.Y[i])**2
        
    return obj

""" NSD Methods """

# Generate the ReactionModels (at some point KipetModel)
models = [m, m2, m3]

# Create the NSD object using the ReactionModels list
nsd = NSD(models, kwargs = dict(kipet=False))

# Set the initial values
nsd.set_initial_value({1 : 2,
                       2 : 1.2e-4}
                     )

# Runs the TR Method
#results = nsd.trust_region(scaled=False)
#nsd.plot_paths()

# Runs the Ipopt Method
results = nsd.ipopt_method(scaled=False)

for i, (k, v) in enumerate(m.P.items()):
    v.set_value(nsd.parameters_opt[k])

""" Plot the resulting line """

x = list(x_data.values())
y = list(y_data.values())
y2 = list(y_data2.values())
x3 = list(x_data3.values())
y3 = list(y_data3.values())

xm = np.linspace(1, 5, 9)
ym = xm*m.P[1].value + m.P[2].value

plt.plot(x, y, 'o')
plt.plot(x, y2, 'ro')
plt.plot(x3, y3, 'go')
plt.plot(xm, ym)