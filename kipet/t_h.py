# -*- coding: utf-8 -*-
from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.diffvar import DerivativeVar

m = ConcreteModel()
m.t = ContinuousSet(bounds=(0, 10))
m.x = Var(m.t, initialize=1.22)
m.y = Var(m.t, initialize=1.)
m.dx = DerivativeVar(m.x)
m.dy = DerivativeVar(m.y)

m.c = Constraint(m.t, rule=lambda m, i: m.dx[i] + m.y[i] == 1.2)
m.d = Constraint(m.t, rule=lambda m, i: m.dy[i] + m.x[i] == 2.0)
e = m.c[0].expr._args[0]
print(e)

for i in m.component_objects(Var):
    print(i.name)

for i in m.component_data_objects(DerivativeVar):
    print(i.name, end='\t')
    h = hex(id(i))
    print(h, end='\t')
    print(i.type(), end='\t')
    print(type(i), end='\t')
    result = isinstance(i, DerivativeVar)
    print(result)

print('\n\n\n')
for i in e._args:
    i = i.parent_component()
    print(i, end='\t')
    h = hex(id(i))
    print(h, end='\t')
    print(i.type(), end='\t')
    print(type(i), end='\t')
    result = isinstance(i, DerivativeVar)
    print(result)

# for j in dir(i):
#     print(j)

d = TransformationFactory('dae.collocation')
d.apply_to(m, nfe=2, ncp=2, scheme='LAGRANGE-RADAU')

e = m.c[0].expr._args[0]
print(e)

for i in e._args:
    print(i.type())

for i in m.component_data_objects(DerivativeVar):
    print(i.name, end='\t')
    h = hex(id(i))
    print(h, end='\t')
    print(i.type(), end='\t')
    print(type(i), end='\t')
    result = isinstance(i, DerivativeVar)
    print(result)


for i in m.component_objects(Var):
    print(i.name)

con_names = []
dvs_names = []
for i in m.component_objects(Constraint):
    name = i.name
    namel = name.split('_', 1)
    if len(namel) > 1:
        if namel[1] == "disc_eq":
            print("variable found")
            con_names.append(i.name)
            dvs_names.append(namel[0])

m.p = Param(m.t, mutable=True, default=1.0)

for i in dvs_names:
    con = getattr(m, i + '_disc_eq')
    dv = getattr(m, i)
    e_dict = {}
    for k in con.keys():
        print(k)
        if isinstance(k, tuple):
            pass
        else:
            k = (k,)
        e = con[k].expr._args[0]
        e_dict[k] = e * m.p[k[0]] == dv[k] * (1 - m.p[k[0]])  #: As long as you don't clone

    print(e_dict)

    m.add_component(i + "_deq_aug", Constraint(con.index_set(), rule=lambda m, j:e_dict[(j,)] if (j,)[0] > 0 else Constraint.Skip))
    n_con = getattr(m, i + "_deq_aug")
    n_con.pprint()
    print(n_con.rule)

print("hallobitteschön")


