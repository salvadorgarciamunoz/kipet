"""
This module contains the code related to jumps - removes redundancy

Changed to not being a mixin to clean up inheritance
"""
from pyomo.core import Var, Param, Constraint, ConstraintList
from pyomo.dae import ContinuousSet

from kipet.model_tools.visitor_classes import ReplacementVisitor

    
def set_up_jumps(model, kwargs):
    """Takes the kwargs from the Estimator classes and initializes the dosing points (or any other type
    of sudden change in the states)

    :param ConcreteModel model: Pyomo model (from Variance or Parameter Estimator)
    :param dict var_dict: Dict of states
    :param dict jump_times: Dict of jump times
    :param array-like feed_times: List of feed times (could be a set)

    :return: None
    
    """
    var_dict = kwargs.pop("jump_states", None)
    jump_times = kwargs.pop("jump_times", None)
    feed_times = kwargs.pop("feed_times", None)
    
    if not isinstance(var_dict, dict):
        print("disc_jump_v_dict is of type {}".format(type(var_dict)))
        raise Exception  # wrong type
    if not isinstance(jump_times, dict):
        print("disc_jump_times is of type {}".format(type(jump_times)))
        raise Exception  # wrong type
    count = 0
    
    for i in jump_times.keys():
        for j in jump_times[i].items():
            count += 1
    if len(feed_times) > count:
        raise Exception("Error: Check feed time points in set feed_times and in jump_times again.\n"
                    "There are more time points in feed_times than jump_times provided.")

    time_set = None
    for i in model.component_objects(ContinuousSet):
        time_set = i
        break
    if time_set is None:
        raise Exception('No continuous_set')

    time_set = time_set.name

    ttgt = getattr(model, time_set)
    ncp = ttgt.get_discretization_info()['ncp']
    fe_l = ttgt.get_finite_elements()
    fe_list = [fe_l[i + 1] - fe_l[i] for i in range(0, len(fe_l) - 1)]

    for fe in range(0, len(fe_list)):
        
        vs = ReplacementVisitor()
        kn = 0
        for ki in jump_times.keys():
            if not isinstance(ki, str):
                print("ki is not str")
            vtjumpkeydict = jump_times[ki]
            for l in vtjumpkeydict.keys():
                jump_time = vtjumpkeydict[l]
                jump_fe, jump_cp = fe_cp(ttgt, jump_time)
                if jump_time not in feed_times:
                    raise Exception("Error: Check feed time points in set feed_times and in jump_times again.\n"
                                    "They do not match.\n"
                                    "Jump_time is not included in feed_times.")
                if fe == jump_fe + 1:
                    for v in var_dict.keys():
                        if not isinstance(v, str):
                            print("v is not str")
                        vkeydict = var_dict[v]
                        for k in vkeydict.keys():
                            if k == l:  # Match in between two components of dictionaries
                                var = getattr(model, v)
                                #dvar = getattr(model, "d" + v + "dt")
                                con_name = 'd' + v + 'dt_disc_eq'
                                con = getattr(model, con_name)

                                model.add_component(v + "_dummy_eq_" + str(kn), ConstraintList())
                                conlist = getattr(model, v + "_dummy_eq_" + str(kn))
                                varname = v + "_dummy_" + str(kn)
                                model.add_component(varname, Var([0]))  #: this is now indexed [0]
                                vdummy = getattr(model, varname)
                                vs.change_replacement(vdummy[0])   #: who is replacing.
                                jump_delta = vkeydict[k]
                                model.add_component(v + '_jumpdelta' + str(kn),
                                                         Param(initialize=jump_delta))
                                jump_param = getattr(model, v + '_jumpdelta' + str(kn))
                                if not isinstance(k, tuple):
                                    k = (k,)
                                exprjump = vdummy[0] - var[(jump_time,) + k] == jump_param  #: this cha
                                # exprjump = vdummy - var[(self.jump_time,) + k] == jump_param
                                model.add_component("jumpdelta_expr" + str(kn), Constraint(expr=exprjump))
                                for kcp in range(1, ncp + 1):
                                    curr_time = t_ij(ttgt, jump_fe + 1, kcp)
                                    if not isinstance(k, tuple):
                                        knew = (k,)
                                    else:
                                        knew = k
                                    idx = (curr_time,) + knew
                                    con[idx].deactivate()
                                    e = con[idx].expr
                                    suspect_var = e.args[0].args[1].args[0].args[0].args[1]  #: seems that
                                    vs.change_suspect(id(suspect_var))  #: who to replace
                                    e_new = vs.dfs_postorder_stack(e)  #: replace
                                    con[idx].set_value(e_new)
                                    conlist.add(con[idx].expr)
                kn = kn + 1

def t_ij(time_set, i, j):
    """Return the corresponding time(continuous set) based on the i-th finite element and j-th collocation point
    From the NMPC_MHE framework by @dthierry.

    :param ContinuousSet time_set: Parent Continuous set
    :param int i: finite element
    :param int j: collocation point

    :return: Corresponding index of the ContinuousSet
    :rtype: float

    """
    if i < time_set.get_discretization_info()['nfe']:
        h = time_set.get_finite_elements()[i + 1] - time_set.get_finite_elements()[i]  #: This would work even for 1 fe
    else:
        h = time_set.get_finite_elements()[i] - time_set.get_finite_elements()[i - 1]  #: This would work even for 1 fe
    tau = time_set.get_discretization_info()['tau_points']
    fe = time_set.get_finite_elements()[i]
    time = fe + tau[j] * h
    return round(time, 6)


def fe_cp(time_set, feedtime):
    """Return the corresponding fe and cp for a given time

    :param ContinuousSet time_set: Parent ContinuousSet
    :param float feedtime: Time of the step change
    
    :return fe: The finite element
    :return cp: The collocation point
    :rtype tuple:
             
    """
    fe_l = time_set.get_lower_element_boundary(feedtime)
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
