# -*- coding: utf-8 -*-

from __future__ import print_function
from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory, ProblemFormat
from pyomo.core.kernel.numvalue import value as value
from os import getcwd, remove
import sys

__author__ = 'David M Thierry'  # type: str #: April 2018


class fe_initialize(object):
    def __init__(self, tgt_mod, src_mod, init_con=None, param_name=None, param_values=None):
        # type: (ConcreteModel, ConcreteModel) -> None
        """

        :type fixed_params: object
        """

        self.tgt = tgt_mod
        self.mod = src_mod.clone()  #: deepcopy
        zeit = None
        for i in self.mod.component_objects(ContinuousSet):
            if i:
                zeit = i
            else:
                raise Exception('no continuous_set')
        self.time_set = zeit.name

        tgt_cts = getattr(self.tgt, self.time_set)
        self.ncp = tgt_cts.get_discretization_info()['ncp']

        fe_l = tgt_cts.get_finite_elements()
        self.fe_list = [fe_l[i + 1] - fe_l[i] for i in range(0, len(fe_l) - 1)]
        self.nfe = len(self.fe_list)

        zeit = getattr(self.mod, self.time_set)
        zeit._bounds = (0, 1)
        zeit.clear()
        zeit.construct()

        for i in self.mod.component_objects([Var, Constraint, DerivativeVar]):
            i.clear()
            i.construct()

        d = TransformationFactory('dae.collocation')
        d.apply_to(self.mod, nfe=1, ncp=self.ncp, scheme='LAGRANGE-RADAU')

        self.dvs_names = []
        self.dvar_names = []
        for i in self.mod.component_objects(Constraint):
            name = i.name
            namel = name.split('_', 1)
            if len(namel) > 1:
                if namel[1] == "disc_eq":
                    realname = getattr(self.mod, namel[0])
                    self.dvar_names.append(namel[0])
                    self.dvs_names.append(realname.get_state_var().name)
        self.mod.h_i = Param(zeit, mutable=True, default=1.0)

        for i in self.dvar_names:
            con = getattr(self.mod, i + '_disc_eq')
            dv = getattr(self.mod, i)
            e_dict = {}
            fun_tup = True
            for k in con.keys():
                if isinstance(k, tuple):
                    pass
                else:
                    k = (k,)
                    fun_tup = False
                e = con[k].expr._args[0]
                e_dict[k] = e * self.mod.h_i[k[0]] == dv[k] * (1 - self.mod.h_i[k[0]])  #: As long as you don't clone
            if fun_tup:
                self.mod.add_component(i + "_deq_aug",
                                       Constraint(con.index_set(),
                                                  rule=lambda m, *j: e_dict[j] if j[0] > 0.0 else Constraint.Skip))
            else:
                self.mod.add_component(i + "_deq_aug",
                                       Constraint(con.index_set(),
                                                  rule=lambda m, j: e_dict[j] if j > 0.0 else Constraint.Skip))
            self.mod.del_component(con)

        self.remaining_set = {}
        for i in self.dvs_names:
            dv = getattr(self.mod, i)
            set_i = dv._implicit_subsets
            remaining_set = set_i[1]
            for s in set_i[2:]:
                if s.name == zeit.name:
                    continue
                else:
                    remaining_set *= s
            if isinstance(remaining_set, list):
                self.remaining_set[i] = remaining_set
            else:
                self.remaining_set[i] = []
                self.remaining_set[i].append(remaining_set)
        self.remaining_set_alg = {}
        for av in self.mod.component_objects(Var):
            if av.name in self.dvs_names:
                continue
            set_i = av._implicit_subsets
            if not set_i:
                continue
            remaining_set = set_i[1]
            for s in set_i[2:]:
                if s.name == zeit.name:
                    continue
                else:
                    remaining_set *= s
            if isinstance(remaining_set, list):
                self.remaining_set_alg[av.name] = remaining_set
            else:
                self.remaining_set_alg[av.name] = []
                self.remaining_set_alg[av.name].append(remaining_set)

        if init_con:
            ic = getattr(self.mod, init_con)
            self.mod.del_component(ic)


        if isinstance(param_name, list):
            if param_values:
                if isinstance(param_values, dict):
                    for pname in param_name:
                        p = getattr(self.mod, pname)
                        for key in p.keys():
                            try:
                                val = param_values[pname, key]
                                print(val)
                                p[key].set_value(val)
                            except KeyError:
                                raise Exception("Missing a key of the param_values\n"
                                                "Please provide all the required keys.\n"
                                                "missing: {}".format(key))
                            p[key].fix()
                else:
                    Exception("Arg param_values should be provided in a dictionary")
            else:
                Exception("Arg param_values should be provided in a dictionary")
        elif isinstance(param_name, str):
            if param_values:
                if isinstance(param_values, dict):
                    p = getattr(self.mod, param_name)
                    for key in p.keys():
                        try:
                            val = param_values[param_name, key]
                            print(val)
                            p[key].set_value(val)
                        except KeyError:
                            raise Exception("Missing a key of the param_values\n"
                                      "Please provide all the required keys.\n"
                                            "missing: {}".format(key))
                        p[key].fix()
        elif not param_name:
            pass
        else:
            raise Exception("wrong type for param_name")



        self.ip = SolverFactory('ipopt')

        for i in self.dvs_names:
            dv = getattr(self.mod, i)

            for rs in self.remaining_set[i]:
                for k in rs:
                    k = k if isinstance(k, tuple) else (k,)
                    dv[(0,) + k].fix()
        (n, m) = reconcile_nvars_mequations(self.mod)
        if n != m:
            raise Exception("whps, n={}, m={}".format(n, m))

    def load_initial_conditions(self, init_cond=None):
        if not isinstance(init_cond, dict):
            raise Exception("init_cond must be a dictionary")
        for i in self.dvs_names:
            dv = getattr(self.mod, i)
            for s in self.remaining_set[i]:
                for k in s:
                    val = init_cond[i, k]
                    k = k if isinstance(k, tuple) else (k,)
                    dv[(0,) + k].set_value(val)
                    if not dv[(0,) + k].fixed:
                        dv[(0,) + k].fix()


    def march_forward(self, fe):
        """

        :rtype: object
        """
        print("fe {}".format(fe))
        self.adjust_h(fe)
        sol = self.ip.solve(self.mod, tee=True)
        # for i in self.mod.component_objects([Var, Constraint]):
        #     i.pprint()
        # sys.exit()
        self.patch(fe)
        self.cycle_ics()

    def cycle_ics(self):
        ts = getattr(self.mod, self.time_set)
        t_last = t_ij(ts, 0, self.ncp)
        for i in self.dvs_names:
            dv = getattr(self.mod, i)
            for s in self.remaining_set[i]:
                for k in s:
                    k = k if isinstance(k, tuple) else (k,)
                    dv[(0,) + k].set_value(value(dv[(t_last,) + k]))
                    if not dv[(0,) + k].fixed:
                        dv[(0,) + k].fix()

    def patch(self, fe):
        ts = getattr(self.mod, self.time_set)
        ttgt = getattr(self.tgt, self.time_set)
        for v in self.mod.component_objects(Var, active=True):
            if not v._implicit_subsets:
                continue
            if ts not in v._implicit_subsets:
                continue
            v_tgt = getattr(self.tgt, v.name)
            if v.name in self.dvs_names:
                drs = self.remaining_set[v.name]
            else:
                drs = self.remaining_set_alg[v.name]
            for j in range(0, self.ncp):
                t_tgt = t_ij(ttgt, fe, j)
                t_src = t_ij(ts, 0, j)
                for k in drs:
                    for key in k:
                        key = key if isinstance(key, tuple) else (key,)
                        try:
                            val = value(v[(t_src,) + key])
                        except ValueError:
                            pass
                        v_tgt[(t_tgt,) + key].set_value(val)

    def adjust_h(self, fe):
        hi = getattr(self.mod, "h_i")
        zeit = getattr(self.mod, self.time_set)
        for t in zeit:
            hi[t].value = self.fe_list[fe]

    def run(self):
        for i in range(0, len(self.fe_list)):
            self.march_forward(i)


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
    if i < time_set.get_discretization_info()['nfe']:
        h = time_set.get_finite_elements()[i + 1] - time_set.get_finite_elements()[i]  #: This would work even for 1 fe
    else:
        h = time_set.get_finite_elements()[i] - time_set.get_finite_elements()[i - 1]  #: This would work even for 1 fe
    tau = time_set.get_discretization_info()['tau_points']
    fe = time_set.get_finite_elements()[i]
    time = fe + tau[j] * h
    return round(time, 6)


def write_nl(d_mod, filename=None):
    # type: (ConcreteModel, str) -> str
    """
    Write the nl file
    Args:
        d_mod (ConcreteModel): the model of interest.

    Returns:
        cwd (str): The current working directory.
    """
    if not filename:
        filename = d_mod.name + '.nl'
    d_mod.write(filename, format=ProblemFormat.nl,
                io_options={"symbolic_solver_labels":True})
    cwd = getcwd()
    print("nl file {}".format(cwd + "/" + filename))
    return cwd


def reconcile_nvars_mequations(d_mod):
    # type: (ConcreteModel) -> tuple
    """
    Compute the actual number of variables and equations in a model by reading the relevant line at the nl file.
    Args:
        d_mod (ConcreteModel):  The model of interest

    Returns:
        tuple: The number of variables and the number of constraints.

    """
    fullpth = getcwd()
    fullpth += "/_reconcilied.nl"
    write_nl(d_mod, filename=fullpth)
    with open(fullpth, 'r') as nl:
        lines = nl.readlines()
        line = lines[1]
        newl = line.split()
        nvar = int(newl[0])
        meqn = int(newl[1])
        nl.close()
    # remove(fullpth)
    return (nvar, meqn)
