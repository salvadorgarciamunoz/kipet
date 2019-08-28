# -*- coding: utf-8 -*-

from __future__ import print_function
from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory, ProblemFormat, TerminationCondition
from pyomo import *
#from pyomo.core.kernel.numvalue import value as value
from os import getcwd, remove
import sys
import six
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import NumericConstant


__author__ = 'David M Thierry'  #: April 2018


class fe_initialize(object):
    def __init__(self, tgt_mod, src_mod, init_con=None, param_name=None, param_values=None, inputs=None, inputs_sub=None,jump_times=None,jump_states=None):
        # type: (ConcreteModel, ConcreteModel, str, list, dict, dict, dict) -> None
        """fe_factory: fe_initialize class.

                This class implements the finite per finite element initialization for a pyomo model initialization.
                A march-forward simulation will be run and the resulting data will be patched to the tgt_model.
                The current strategy is as follows:
                Create a copy of the undiscretized model.
                Change the corresponding time set bounds to (0,1).
                Discretize and create the new model with the parameter h_i.
                Deactivate initial conditions.
                Check for params and inputs.

                Note that an input needs to be a variable(fixed) indexed over time. Otherwise it would be a parameter.

                The `the paran name` might be a list of strings or a single string
                 corresponding to the parameters of the model.
                The `param_values` dictionary needs to be declared with the following sintax:
                `param_dict["P", "k0"] = 49.7796`
                Where the first key corresponds to one of the parameter names, and the second to the corresponding
                index (if any).
                A similar structure is expected for the initial conditions and inputs.

                The `inputs` and `input_sub` parameters are in place depending of whether there is a single index input
                or a multiple index input.

                Note that if the user does not provide correct information to fe_factory; an exception will be thrown
                because of the n_var and m_eqn check for simulation.

                Once the constructor is called, one can initialize the model with the following sintax:
                `self.load_initial_conditions(init_cond=ics_dict)`

                Finally, to run the initialization and automatic data patching to tgt model use:
                `self.run()`

                If a given finite element problem fails, we do will try once again with relaxed options. It is
                recommended to go back and check the model for better understanding of the issue.

                Finally, an explicit function of time on the right hand side is prohibited. Please put this information
                into an input (fixed variable) instead.

                Args:
                    tgt_mod (ConcreteModel): The originall fully discretized model that we want to patch the information to.
                    src_mod (ConcreteModel): The undiscretized reference model.
                    init_con (str): The initial constraint name (corresponds to a Constraint object).
                    param_name (list): The param name list. (Each element must correspond to a pyomo Var)
                    param_values (dict): The corresponding values: `param_dict["param_name", "param_index"] = 49.7796`
                    inputs (dict): The input dictionary. Use this dictonary for single index (time) inputs
                    inputs_sub (dict): The multi-index dictionary. Use this dictionary for multi-index inputs.
                """




        self.ip = SolverFactory('ipopt')
        self.ip.options['halt_on_ampl_error'] = 'yes'
        self.ip.options['print_user_options'] = 'yes'
        self.tgt = tgt_mod
        # src_mod.display(filename="src.txt")
        self.mod = src_mod.clone()  #: Deepcopy of the reference model
        # self.mod.display(filename="selfmod0.txt")
        zeit = None
        for i in self.mod.component_objects(ContinuousSet):
            zeit = i
            break
        if zeit is None:
            raise Exception('no continuous_set')

        self.time_set = zeit.name

        tgt_cts = getattr(self.tgt, self.time_set)
        self.ncp = tgt_cts.get_discretization_info()['ncp']

        fe_l = tgt_cts.get_finite_elements()
        self.fe_list = [fe_l[i + 1] - fe_l[i] for i in range(0, len(fe_l) - 1)]
        self.nfe = len(self.fe_list)  #: Create a list with the step-size
        #: Re-construct the model with [0,1] time domain
        zeit = getattr(self.mod, self.time_set)
        #print()

        zeit._bounds = (0, 1)
        zeit.clear()
        zeit.construct()
        for i in self.mod.component_objects(Var):
            i.clear()
            i.reconstruct()
            #i.pprint()
        for i in self.mod.component_objects(Var):
            i.clear()
            i.reconstruct()
        for i in self.mod.component_objects(Constraint):
            i.clear()
            i.construct()

        # self.mod.display(filename="selfmoddisc0.txt")
        #: Discretize
        d = TransformationFactory('dae.collocation')
        d.apply_to(self.mod, nfe=1, ncp=self.ncp, scheme='LAGRANGE-RADAU')

        #: Find out the differential variables
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
        self.mod.h_i = Param(zeit, mutable=True, default=1.0)  #: Length of finite element
        #: Modify the collocation equations to introduce h_i (the length of finite element)
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
                e = con[k].expr.args[0]
                e_dict[k] = e * self.mod.h_i[k[0]] + dv[k] * (1 - self.mod.h_i[k[0]]) == 0.0  #: As long as you don't clone
            if fun_tup:
                self.mod.add_component(i + "_deq_aug",
                                       Constraint(con.index_set(),
                                                  rule=lambda m, *j: e_dict[j] if j[0] > 0.0 else Constraint.Skip))
            else:
                self.mod.add_component(i + "_deq_aug",
                                       Constraint(con.index_set(),
                                                  rule=lambda m, j: e_dict[j] if j > 0.0 else Constraint.Skip))
            self.mod.del_component(con)

        #: Sets for iteration
        #: Differential variables
        self.remaining_set = {}
        for i in self.dvs_names:
            dv = getattr(self.mod, i)
            if dv.index_set().name == zeit.name:  #: Just time set
                #print(i, 'here')
                self.remaining_set[i] = None
                continue
            set_i = dv._implicit_subsets  #: More than just time set
            remaining_set = set_i[1]
            for s in set_i[2:]:
                remaining_set *= s
            if isinstance(remaining_set, list):
                self.remaining_set[i] = remaining_set
            else:
                self.remaining_set[i] = []
                self.remaining_set[i].append(remaining_set)
        #: Algebraic variables
        self.weird_vars = [] #:Not indexed by time
        self.remaining_set_alg = {}
        for av in self.mod.component_objects(Var):
            if av.name in self.dvs_names:
                continue
            if av.index_set().name == zeit.name:  #: Just time set
                self.remaining_set_alg[av.name] = None
                continue
            set_i = av._implicit_subsets
            if set_i is None or not zeit in set_i:
                self.weird_vars.append(av.name)  #: Not indexed by time!
                continue  #: if this happens we might be in trouble
            remaining_set = set_i[1]  #: Index by time and others
            for s in set_i[2:]:
                if s.name == zeit.name:
                    self.remaining_set_alg[av.name] = None
                    continue
                else:
                    remaining_set *= s
            if isinstance(remaining_set, list):
                self.remaining_set_alg[av.name] = remaining_set
            else:
                self.remaining_set_alg[av.name] = []
                self.remaining_set_alg[av.name].append(remaining_set)

        if init_con is not None:  #: Delete the initial conditions (we use .fix() instead)
            ic = getattr(self.mod, init_con)
            self.mod.del_component(ic)

        if isinstance(param_name, list):  #: Time independent parameters
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

        #: Fix initial conditions
        for i in self.dvs_names:
            dv = getattr(self.mod, i)
            if self.remaining_set[i] is None:
                dv[0].fix()
            for rs in self.remaining_set[i]:
                for k in rs:
                    k = k if isinstance(k, tuple) else (k,)
                    dv[(0,) + k].fix()

        self.inputs = None
        self.input_remaining_set = {}

        #: Check if inputs are declared
        if self.inputs is not None:
            if not isinstance(inputs, dict) or isinstance(inputs, str):
                raise Exception("Must be a dict or str")
            if isinstance(inputs, str):
                self.inputs = [self.inputs]
            for i in self.inputs:
                p = getattr(self.mod, i)
                p.fix()
                if p.index_set().name == zeit.name:  #: Only time-set
                    self.input_remaining_set[i] = None
                    continue
                set_i = p._implicit_subsets
                if not zeit in set_i:
                    raise RuntimeError("{} is not by index by time, this can't be an input".format(i))
                remaining_set = set_i[1]
                for s in set_i[2:]:
                    if s.name == zeit.name:  #: would this ever happen?
                        continue
                    else:
                        remaining_set *= s
                if isinstance(remaining_set, list):
                    self.input_remaining_set[i] = remaining_set
                else:
                    self.input_remaining_set[i] = []
                    self.input_remaining_set[i].append(remaining_set)
        self.inputs_sub = None
        # inputs_sub['some_var'] = ['index0', 'index1', ('index2a', 'index2b')]
        self.inputs_sub = inputs_sub
        if not self.inputs_sub is None:
            if not isinstance(self.inputs_sub, dict):
                raise TypeError("inputs_sub must be a dictionary")
            for key in self.inputs_sub.keys():
                if not isinstance(self.inputs_sub[key], list):
                    raise TypeError("input_sub[{}] must be a list".format(key))
                p = getattr(self.mod, key)
                if p._implicit_subsets is None:
                    raise RuntimeError("This variable is does not have multiple indices"
                                       "Pass {} as part of the inputs keyarg instead.".format(key))
                elif p.index_set().name == zeit.name:
                    raise RuntimeError("This variable is indexed over time"
                                       "Pass {} as part of the inputs keyarg instead.".format(key))
                else:
                    if not zeit in p._implicit_subsets:
                        raise RuntimeError("{} is not indexed over time; it can not be an input".format(key))
                for k in self.inputs_sub[key]:
                    if isinstance(k, str) or isinstance(k, int) or isinstance(k, tuple):
                        k = (k,) if not isinstance(k, tuple) else k
                    else:
                        raise RuntimeError("{} is not a valid index".format(k))
                    for t in zeit:
                        p[(t,) + k].fix()

        #: Check nvars and mequations
        (n, m) = reconcile_nvars_mequations(self.mod)
        # self.mod.display(filename="selfmod1.txt")
        if n != m:
            raise Exception("Inconsistent problem; n={}, m={}".format(n, m))
        self.jump = False



    def load_initial_conditions(self, init_cond=None):
        if not isinstance(init_cond, dict):
            raise Exception("init_cond must be a dictionary")

        for i in self.dvs_names:
            dv = getattr(self.mod, i)
            ts = getattr(self.mod, self.time_set)
            for t in ts:
                for s in self.remaining_set[i]:
                    if s is None:
                        val = init_cond[i]  #: if you do not have an extra index, just put the value there
                        dv[t].set_value(val)
                        if t == 0:
                            if not dv[0].fixed:
                                dv[0].fix()
                        continue
                    for k in s:
                        val = init_cond[i, k]
                        k = k if isinstance(k, tuple) else (k,)
                        dv[(t,) + k].set_value(val)
                        if t == 0:
                            if not dv[(0,) + k].fixed:
                                dv[(0,) + k].fix()

    def march_forward(self, fe, resto_strategy="bound_relax"):
        # type: (int) -> None
        """Moves forward with the simulation.

        This method performs the actions required for setting up the `fe-th` problem.

        Adjust inputs.
        Solve current problem.
        Patches tgt_model.
        Cycles initial conditions

        Args:
            fe (int): The correspoding finite element.
        """
        print("fe {}".format(fe))
        self.adjust_h(fe)
        if self.inputs or self.inputs_sub:
            self.load_input(fe)
        # self.mod.X.display()

        # for i in self.mod.X.itervalues():
        #     idx = i.index()
        #     if idx[1] in ['Msa']:
        #         i.setlb(-0.01)
        #     else:
        #         i.setlb(0)
        #for i in self.mod.Z.itervalues():
            #i.setlb(0)
        self.ip.options["print_level"] = 1  #: change this on demand
        # self.ip.options["start_with_resto"] = 'no'
        self.ip.options['bound_push'] = 1e-02
        sol = self.ip.solve(self.mod, tee=True, symbolic_solver_labels=True)

        if sol.solver.termination_condition != TerminationCondition.optimal:
            self.ip.options["OF_start_with_resto"] = 'yes'
            # self.ip.options["linear_solver"] = "ma57"
            # for i in self.mod.component_objects(Var):
            #     i.pprint()
            sol = self.ip.solve(self.mod, tee=True, symbolic_solver_labels=True)
            if sol.solver.termination_condition != TerminationCondition.optimal:
                self.ip.options["OF_start_with_resto"] = 'no'
                self.ip.options["bound_push"] = 1E-02
                self.ip.options["OF_bound_relax_factor"] = 1E-05
                # self.ip.options[""]
                # for i in self.mod.component_data_objects(Var):
                #     i.setlb(None)
                # for i in self.mod.Z.itervalues():
                #     i.setlb(None)
                # for i in self.mod.X.itervalues():
                #     idx = i.index()
                #     if idx[1] in ['Msa']:
                #         i.setlb(-0.05)
                #     else:
                #         i.setlb(None)
                sol = self.ip.solve(self.mod, tee=True, symbolic_solver_labels=True)
                self.ip.options["OF_bound_relax_factor"] = 1E-08
                if sol.solver.termination_condition != TerminationCondition.optimal:
                    raise Exception("The current iteration was unsuccessful. Iteration :{}".format(fe))

        else:
            print("fe {} - status: optimal".format(fe))
        self.patch(fe)
        self.cycle_ics(fe)

    #Inclusion of discrete jumps: (CS)
    def load_discrete_jump(self, var_dic, jump_times, feed_times):
        """Method is used to define and load the places where discrete jumps are located, e.g.
        dosing points or external inputs.
        Args:
            var_dic (dict): dictionary containing which variables are inputted and by how much
            jump_times (dict): dict containing the times that each variable is inputted
            feed_times (list): list of additional time points needed for inputs

        Returns:
            None
        """
        self.jump = True
        self.disc_jump_v_dict = var_dic
        self.jump_times_dict = jump_times #now dictionary
        self.feed_times_set = feed_times
        count = 0
        for i in six.iterkeys(self.jump_times_dict):
            for j in six.iteritems(self.jump_times_dict[i]):
                count += 1
        if len(self.feed_times_set) > count:
            raise Exception("Error: Check feed time points in set feed_times and in jump_times again.\n"
                            "There are more time points in feed_times than jump_times provided.")



    def cycle_ics(self, curr_fe):
        """Cycles the initial conditions of the initializing model.
        Take the values of states (initializing model) at t=last and patch them into t=0.
        Check:
        https://github.com/dthierry/cappresse/blob/pyomodae-david/nmpc_mhe/aux/utils.py
        fe_cp function!
        """

        #For checking whether jump happens in right element:
        #print('*'* 20)
        #print("Current Finite element [cycle_ics]{}".format(curr_fe)) #comment out these lines?
        #print('*' * 20)

        ts = getattr(self.mod, self.time_set)
        t_last = t_ij(ts, 0, self.ncp)

        #Inclusion of discrete jumps: (CS)
        ttgt = getattr(self.tgt, self.time_set)

        for i in self.dvs_names:
            dv = getattr(self.mod, i)
            for s in self.remaining_set[i]:
                if s is None:
                    val = value(dv[t_last])
                    dv[0].set_value(val)
                    if not dv[0].fixed:
                        dv[0].fix()
                    continue
                for k in s:
                    k = k if isinstance(k, tuple) else (k,)
                    val = value(dv[(t_last,) + k])
                    dv[(0,) + k].set_value(val)
                    if not dv[(0,) + k].fixed:
                        dv[(0,) + k].fix()

    def patch(self, fe):
        # type: (int) -> None
        """ Take the current state of variables of the initializing model at fe and load it into the tgt_model
        Note that this will skip fixed variables as a safeguard.

        Args:
            fe (int): The current finite element to be patched (tgt_model).
        """
        ###########################
        ts = getattr(self.mod, self.time_set)
        ttgt = getattr(self.tgt, self.time_set)
        for v in self.mod.component_objects(Var, active=True):
            v_tgt = getattr(self.tgt, v.name)
            if v.name in self.weird_vars:  #: This has got to work.
                for k in v.keys():
                    if v[k].stale or v[k].is_fixed():
                        continue
                    try:
                        val = value(v[k])
                    except ValueError:
                        pass
                    v_tgt[k].set_value(val)
                continue
            #: From this point on all variables are indexed over time.
            if v.name in self.dvs_names:
                drs = self.remaining_set[v.name]
            else:
                drs = self.remaining_set_alg[v.name]
            for j in range(0, self.ncp + 1):
                t_tgt = t_ij(ttgt, fe, j)
                t_src = t_ij(ts, 0, j)

                if drs is None:
                    if v[t_src].stale or v[t_src].is_fixed():
                        continue
                    try:
                        val = value(v[t_src])
                    except ValueError:
                        print("Error at {}, {}".format(v.name, t_src))
                    v_tgt[t_tgt].set_value(val)
                    continue
                for k in drs:
                    for key in k:
                        key = key if isinstance(key, tuple) else (key,)
                        if v[(t_src,) + key].stale or v[(t_src,) + key].is_fixed():
                            continue
                        try:
                            val = value(v[(t_src,) + key])
                        except ValueError:
                            print("Error at {}, {}".format(v.name, (t_src,) + key))
                        v_tgt[(t_tgt,) + key].set_value(val)
        ##############################
        #Inclusion of discrete jumps: (CS)
        if self.jump:
            vs = ReplacementVisitor()  #: trick to replace variables
            kn=0
            for ki in self.jump_times_dict.keys():
                if not isinstance(ki, str):
                    print("ki is not str")
                vtjumpkeydict = self.jump_times_dict[ki]
                for l in vtjumpkeydict.keys():
                    self.jump_time = vtjumpkeydict[l]
                    self.jump_fe, self.jump_cp = fe_cp(ttgt,self.jump_time)
                    if self.jump_time not in self.feed_times_set:
                        raise Exception("Error: Check feed time points in set feed_times and in jump_times again.\n"
                                        "They do not match.\n"
                                        "Jump_time is not included in feed_times.")
                    elif fe == self.jump_fe+1:
                                #################################
                        for v in self.disc_jump_v_dict.keys():
                            if not isinstance(v, str):
                                print("v is not str")
                                sys.exit()
                            vkeydict = self.disc_jump_v_dict[v]
                            # print(len(self.feed_times_set))
                            # print(len(self.jump_times_dict.keys()))
                            for k in vkeydict.keys():
                                if k==l:##############!!!!!#Match in between two components of dictionaries
                                    var = getattr(self.tgt, v)
                                    con_name = 'd' + v + 'dt_disc_eq'
                                    con = getattr(self.tgt, con_name)
                                    self.tgt.add_component(v + "_dummy_eq_" + str(kn), ConstraintList())
                                    conlist = getattr(self.tgt, v + "_dummy_eq_" + str(kn))
                                    varname = v + "_dummy_" + str(kn)
                                    self.tgt.add_component(varname, Var([0]))  #: this is now indexed [0]
                                    vdummy = getattr(self.tgt, varname)
                                    vs.change_replacement(vdummy[0])   #: who is replacing.
                                    jump_delta = vkeydict[k]
                                    self.tgt.add_component(v + '_jumpdelta' + str(kn), Param(initialize=jump_delta))
                                    jump_param  = getattr(self.tgt, v + '_jumpdelta' + str(kn))
                                    if not isinstance(k, tuple):
                                        k = (k,)
                                    exprjump = vdummy[0] - var[(self.jump_time,)+k] == jump_param  #: this changed
                                    self.tgt.add_component("jumpdelta_expr"+str(kn), Constraint(expr=exprjump))
                                    for kcp in range(1,self.ncp+1):
                                        curr_time = t_ij(ttgt,self.jump_fe+1,kcp)
                                        if not isinstance(k, tuple):
                                            knew = (k,)
                                        else:
                                            knew = k
                                        idx = (curr_time,) +  knew
                                        con[idx].deactivate()
                                        e=con[idx].expr
                                        suspect_var = e.args[0].args[1].args[0].args[0].args[1]  #: seems that this is the correct variable
                                        vs.change_suspect(id(suspect_var))  #: who to replace
                                        e_new = vs.dfs_postorder_stack(e)  #: replace
                                        con[idx].set_value(e_new)
                                        conlist.add(con[idx].expr)
                    kn=kn+1
            #########################################
    def adjust_h(self, fe):
        # type: (int) -> None
        """Adjust the h_i parameter of the initializing model.

        The initializing model goes from t=(0,1) so it needs to be scaled by the current time-step size.

        Args:
            fe (int): The current value of h_i
        """

        hi = getattr(self.mod, "h_i")
        zeit = getattr(self.mod, self.time_set)
        for t in zeit:
            hi[t].value = self.fe_list[fe]

    def run(self, resto_strategy="bound_relax"):
        """Runs the sequence of problems fe=0,nfe

        """
        print("*"*5, end='\t')
        print("Fe Factory: fe_initialize by DT \@2018", end='\t')
        print("*" * 5)
        print("*" * 5 + '\tSolving for {} elements\t'.format(len(self.fe_list)) + "*" * 5 )
        for i in range(0, len(self.fe_list)):
            self.march_forward(i, resto_strategy=resto_strategy)

    def load_input(self, fe):
        # type: (int) -> None
        """ Loads the current value of input from tgt_model into the initializing model at the current fe.

        Args:
            fe (int):  The current finite element to be loaded.
        """
        if not self.inputs is None:
            ts = getattr(self.mod, self.time_set)
            ttgt = getattr(self.tgt, self.time_set)
            for i in self.inputs:
                p_data = getattr(self.tgt, i)
                p_sim = getattr(self.mod, i)
                if self.input_remaining_set[i] is None:
                    for j in range(0, self.ncp + 1):
                        t = t_ij(ttgt, fe, j)
                        tsim = t_ij(ts, 0, j)
                        val = value(p_data[t])
                        p_sim[tsim].set_value(val)
                    continue
                for k in self.input_remaining_set[i]:
                    for key in k:
                        for j in range(0, self.ncp + 1):
                            t = t_ij(ttgt, fe, j)
                            tsim = t_ij(ts, 0, j)
                            val = value(p_data[(t,) + key])
                            p_sim[(tsim,) + key].set_value(val)
        if not self.inputs_sub is None:
            ts = getattr(self.mod, self.time_set)
            ttgt = getattr(self.tgt, self.time_set)
            for key in self.inputs_sub.keys():
                p_data = getattr(self.tgt, key)
                p_sim = getattr(self.mod, key)
                for k in self.inputs_sub[key]:
                    k = (k,) if not isinstance(k, tuple) else k
                    for j in range(0, self.ncp + 1):
                        t = t_ij(ttgt, fe, j)
                        tsim = t_ij(ts, 0, j)
                        val = value(p_data[(t,) + k])
                        p_sim[(tsim,) + k].set_value(val)

    def create_bounds(self, bound_dict):
        ts = getattr(self.mod, self.time_set)
        for v in bound_dict.keys():
            var = getattr(self.mod, v)
            varbnd = bound_dict[v]
            if not isinstance(varbnd, dict):
                raise RuntimeError("The entry for {} is not a dictionary".format(v))
            for t in ts:
                for k in varbnd.keys():
                    bnd = varbnd[k]
                    if not isinstance(k, tuple):
                        k = (k,)
                    var[(t,) + k].setlb(bnd[0])  #: Lower bound
                    var[(t,) + k].setub(bnd[1])  #: Upper bound

    def clear_bounds(self):
        for v in self.mod.component_data_objects(Var):
            v.setlb(None)
            v.setub(None)



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
                io_options={"symbolic_solver_labels": True})
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
    fullpth += "/_reconciled.nl"
    write_nl(d_mod, filename=fullpth)
    with open(fullpth, 'r') as nl:
        lines = nl.readlines()
        line = lines[1]
        newl = line.split()
        nvar = int(newl[0])
        meqn = int(newl[1])
        nl.close()
    #try:
    #    remove(fullpth)
    #except OSError:
    #    pass
    return (nvar, meqn)


def disp_vars(model, file):
    """Helper function for debugging

    Args:
        model (ConcreteModel): Model of interest.
        file (str): Destination text file.
    """
    with open(file, 'w') as f:
        for c in model.component_objects(Var):
            c.pprint(ostream=f)
        f.close()

#Function needed for inclusion of discrete jump to get fe and cp:
def fe_cp(time_set, feedtime):
    # type: (ContinuousSet, float) -> tuple
    # """Return the corresponding fe and cp for a given time
    # Args:
    #    time_set:
    #    t:
    # """
    fe_l = time_set.get_lower_element_boundary(feedtime)
   # print("fe_l", fe_l)
    fe = None
    j = 0
    for i in time_set.get_finite_elements():
        if fe_l == i:
            fe = j
            break
        j += 1
    h = time_set.get_finite_elements()[1] - time_set.get_finite_elements()[0]
    tauh = [i * h for i in time_set.get_discretization_info()['tau_points']]
    j = 0  #: Watch out for LEGENDRE
    cp = None
    for i in tauh:
        if round(i + fe_l, 6) == feedtime:
            cp = j
            break
        j += 1
    return fe, cp


#: This class can replace variables from an expression
class ReplacementVisitor(EXPR.ExpressionReplacementVisitor):
    def __init__(self):
        super(ReplacementVisitor, self).__init__()
        self._replacement = None
        self._suspect = None

    def change_suspect(self, suspect_):
        self._suspect = suspect_

    def change_replacement(self, replacement_):
        self._replacement = replacement_

    def visiting_potential_leaf(self, node):
        #
        # Clone leaf nodes in the expression tree
        #
        if node.__class__ in native_numeric_types:
            return True, node

        if node.__class__ is NumericConstant:
            return True, node


        if node.is_variable_type():
            if id(node) == self._suspect:
                d = self._replacement
                return True, d
            else:
                return True, node

        return False, None

