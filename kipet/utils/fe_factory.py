# -*- coding: utf-8 -*-

from __future__ import print_function
from pyomo.environ import *
from pyomo.dae import *

__author__ = 'David M Thierry'  #: April 2018


class fe_initialize(object):
    def __init__(self, tgt_mod, src_mod):
        # type: (ConcreteModel, ConcreteModel) -> None
        self.tgt = tgt_mod
        if not self.tgt.time.get_discretization_info():
            raise RuntimeError('apply discretization first before fe_factory')
        self.mod = src_mod.clone()  #: deepcopy
        fe_l = self.tgt.get_finite_elements()
        cs_tgt = None
        cs_found = None
        for i in self.mod.component_objects(ContinuousSet):
            if i:
                cs_tgt = i
            else:
                raise Exception('no continuous_set')

        self.time_set_tgt = cs_tgt.name
        self.ncp = cs_tgt.get_discretization_info()['ncp']
        self.fe_list = [fe_l[i] - fe_l[i+1] for i in range(0, len(cs_tgt.get_finite_elements() - 1))]
        self.nfe = len(self.fe_list)

        for i in self.mod.component_objects(ContinuousSet):
            if i:
                cs_found = i
            else:
                raise Exception('no continuous_set')

        self.time_set_mod = cs_found.name
        cs = getattr(self.mod, self.time_set_mod)
        cs._bounds = (0, 1)
        cs.clear()
        cs.construct()

        for i in self.mod.component_objects([Var, Constraint]):
            i.clear()
            i.construct()

        d = TransformationFactory('dae.collocation')
        d.apply_to(self.mod, nfe=1, ncp=self.ncp, scheme='LAGRANGE-RADAU')

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
            con.deactivate()

    def get_time_steps(self):
        pass

    def march_forward(self):
        pass

    def cycle_ics(self):
        ts = getattr(self.mod, self.time_set_mod)
        t_last = t_ij(ts, 1, self.ncp)
        for i in self.dvs_names:
            dv = getattr(self.mod, i)
            set = dv._implicit_subsets
            remaining_set = set()
            for s in set:
                if s.name == ts.name:
                    continue
                else:
                    remaining_set *= s






            #: get the rest of the set
            #: iterate over that



    def patch(self):
        pass



def t_ij(time_set, i, j):
    # type: (ContinuousSet, int, int) -> float
    """Return the corresponding time(continuous set) based on the i-th finite element and j-th collocation point
    From the NMPC_MHE framework by @dthierry.

    Args:
        time_set (ContinuousSet): Parent Continuous set
        i (int): finite element
        j (int): collocation point

    Returns:
        float: Corresponding index of the ContinuousSet
    """
    h = time_set.get_finite_elements()[1] - time_set.get_finite_elements()[0]  #: This would work even for 1 fe
    tau = time_set.get_discretization_info()['tau_points']
    fe = time_set.get_finite_elements()[i]
    time = fe + tau[j] * h
    return round(time, 6)
