"""
This module contains the code related to jumps - removes redundancy
"""
from pyomo.core import *
from pyomo.dae import *
from pyomo.environ import *

from kipet.library.mixins.VisitorMixins import ReplacementVisitor
    
def set_up_jumps(est_object, kwargs):

    var_dic = kwargs.pop("jump_states", None)
    jump_times = kwargs.pop("jump_times", None)
    feed_times = kwargs.pop("feed_times", None)

    est_object.disc_jump_v_dict = var_dic
    est_object.jump_times_dict = jump_times  # now dictionary
    est_object.feed_times_set = feed_times
    if not isinstance(est_object.disc_jump_v_dict, dict):
        print("disc_jump_v_dict is of type {}".format(type(est_object.disc_jump_v_dict)))
        raise Exception  # wrong type
    if not isinstance(est_object.jump_times_dict, dict):
        print("disc_jump_times is of type {}".format(type(est_object.jump_times_dict)))
        raise Exception  # wrong type
    count = 0
    for i in est_object.jump_times_dict.keys():
        for j in est_object.jump_times_dict[i].items():
            count += 1
    if len(est_object.feed_times_set) > count:
        raise Exception("Error: Check feed time points in set feed_times and in jump_times again.\n"
                    "There are more time points in feed_times than jump_times provided.")
    est_object.load_discrete_jump()

def load_discrete_jump(est_object):
    est_object.jump = True

    zeit = None
    for i in est_object.model.component_objects(ContinuousSet):
        zeit = i
        break
    if zeit is None:
        raise Exception('No continuous_set')

    est_object.time_set = zeit.name

    tgt_cts = getattr(est_object.model, est_object.time_set)  ## please correct me (not necessary!)
    est_object.ncp = tgt_cts.get_discretization_info()['ncp']
    fe_l = tgt_cts.get_finite_elements()
    fe_list = [fe_l[i + 1] - fe_l[i] for i in range(0, len(fe_l) - 1)]

    for i in range(0, len(fe_list)):  # test whether integer elements
        est_object.jump_constraints(i)
  

# I want to change this spaghetti
def jump_constraints(est_object, fe):
    # type: (int) -> None
    """ Take the current state of variables of the initializing model at fe and load it into the tgt_model
    Note that this will skip fixed variables as a safeguard.

    Args:
        fe (int): The current finite element to be patched (tgt_model).
    """
    ###########################
    if not isinstance(fe, int):
        raise Exception  # wrong type
    ttgt = getattr(est_object.model, est_object.time_set)
    ##############################
    # Inclusion of discrete jumps: (CS)
    if est_object.jump:
        vs = ReplacementVisitor()  #: trick to replace variables
        kn = 0
        for ki in est_object.jump_times_dict.keys():
            if not isinstance(ki, str):
                print("ki is not str")
            vtjumpkeydict = est_object.jump_times_dict[ki]
            for l in vtjumpkeydict.keys():
                est_object.jump_time = vtjumpkeydict[l]
                # print('jumptime:',est_object.jump_time)
                est_object.jump_fe, est_object.jump_cp = fe_cp(ttgt, est_object.jump_time)
                if est_object.jump_time not in est_object.feed_times_set:
                    raise Exception("Error: Check feed time points in set feed_times and in jump_times again.\n"
                                    "They do not match.\n"
                                    "Jump_time is not included in feed_times.")
                # print('jump_el, el:',est_object.jump_fe, fe)
                if fe == est_object.jump_fe + 1:
                    #################################
                    for v in est_object.disc_jump_v_dict.keys():
                        if not isinstance(v, str):
                            print("v is not str")
                        vkeydict = est_object.disc_jump_v_dict[v]
                        for k in vkeydict.keys():
                            if k == l:  # Match in between two components of dictionaries
                                var = getattr(est_object.model, v)
                                dvar = getattr(est_object.model, "d" + v + "dt")
                                con_name = 'd' + v + 'dt_disc_eq'
                                con = getattr(est_object.model, con_name)

                                est_object.model.add_component(v + "_dummy_eq_" + str(kn), ConstraintList())
                                conlist = getattr(est_object.model, v + "_dummy_eq_" + str(kn))
                                varname = v + "_dummy_" + str(kn)
                                est_object.model.add_component(varname, Var([0]))  #: this is now indexed [0]
                                vdummy = getattr(est_object.model, varname)
                                vs.change_replacement(vdummy[0])   #: who is replacing.
                                # est_object.model.add_component(varname, Var())
                                # vdummy = getattr(est_object.model, varname)
                                jump_delta = vkeydict[k]
                                est_object.model.add_component(v + '_jumpdelta' + str(kn),
                                                         Param(initialize=jump_delta))
                                jump_param = getattr(est_object.model, v + '_jumpdelta' + str(kn))
                                if not isinstance(k, tuple):
                                    k = (k,)
                                exprjump = vdummy[0] - var[(est_object.jump_time,) + k] == jump_param  #: this cha
                                # exprjump = vdummy - var[(est_object.jump_time,) + k] == jump_param
                                est_object.model.add_component("jumpdelta_expr" + str(kn), Constraint(expr=exprjump))
                                for kcp in range(1, est_object.ncp + 1):
                                    curr_time = t_ij(ttgt, est_object.jump_fe + 1, kcp)
                                    if not isinstance(k, tuple):
                                        knew = (k,)
                                    else:
                                        knew = k
                                    idx = (curr_time,) + knew
                                    con[idx].deactivate()
                                    e = con[idx].expr
                                    suspect_var = e.args[0].args[1].args[0].args[0].args[1]  #: seems that
                                    # e = con[idx].expr.clone()
                                    # e.args[0].args[1] = vdummy
                                    # con[idx].set_value(e)
                                    vs.change_suspect(id(suspect_var))  #: who to replace
                                    e_new = vs.dfs_postorder_stack(e)  #: replace
                                    con[idx].set_value(e_new)
                                    conlist.add(con[idx].expr)
                kn = kn + 1

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


def fe_cp(time_set, feedtime):
    """Return the corresponding fe and cp for a given time
     Args:
        time_set:
        t:
    
    type: (ContinuousSet, float) -> tuple
             
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