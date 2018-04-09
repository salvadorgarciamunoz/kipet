# -*- coding: utf-8 -*-

from __future__ import print_function
from pyomo.environ import *
from pyomo.dae import *

__author__ = 'David M Thierry'  #: April 2018


class fe_initialize(object):
    def __init__(self, tgt_mod, src_mod, ncp):
        # type: (fe_initialize, ConcreteModel, ConcreteModel) -> None
        self.mod = src_mod.clone()  #: deepcopy

        for i in self.mod.component_objects(ContinuousSet):
            if i:
                pass
            else:
                raise Exception('no continuous_set')

        cs = getattr(self.mod, i.name)
        cs._bounds = (0, 1)
        cs.clear()
        cs.construct()

        for i in self.mod.component_objects([Var, Constraint]):
            i.clear()
            i.construct()

        d = TransformationFactory('dae.collocation')
        d.apply_to(self.mod, nfe=1, ncp=ncp, scheme='LAGRANGE-RADAU')

        self.dvs_names = []
        for i in self.mod.component_objects(Constraint):
            name = i.name
            namel = name.split('_', 1)
            if len(namel) > 1:
                if namel[1] == "disc_eq":
                    self.dvs_names.append(namel[0])

        self.mod.h_i = Param(cs, mutable=True, default=1.0)

        for i in self.dvs_names:
            con = getattr(self.mod, i + '_disc_eq')
            dv = getattr(self.mod, i)
            e_dict = {}
            for k in con.keys():
                print(k)
                if isinstance(k, tuple):
                    pass
                else:
                    k = (k,)
                e = con[k].expr._args[0]
                e_dict[k] = e * self.mod.h_i[k[0]] == dv[k] * (1 - self.mod.h_i[k[0]])  #: As long as you don't clone
            self.mod.add_component(i + "_deq_aug", Constraint(con.index_set(), rule=lambda m, j: \
                e_dict[(j,)] if (j,)[0] > 0 else Constraint.Skip))
            n_con = getattr(self.mod, i + "_deq_aug")
            con.deactivate()

    def get_time_steps(self):
        pass

    def march_forward(self):
        pass

    def cycle_ics(self):
        pass

    def patch(self):
        pass

