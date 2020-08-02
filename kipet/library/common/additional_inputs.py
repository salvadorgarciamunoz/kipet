"""
Additional Input Options for Variance and Parameter Estimation Classes
Place into Optimizer?
"""

def add_inputs(est_object, kwds):
    
    est_object.fixedtraj = kwds.pop("fixedtraj", False)
    est_object.fixedy = kwds.pop("fixedy", False)
    est_object.inputs_sub = kwds.pop("inputs_sub", None)
    est_object.yfix = kwds.pop("yfix", None)
    est_object.yfixtraj = kwds.pop("yfixtraj", None)
    trajectories = kwds.pop("trajectories", None)
    
    if est_object.inputs_sub != None:
        for k in est_object.inputs_sub.keys():
            if not isinstance(est_object.inputs_sub[k], list):
                print("wrong type for inputs_sub {}".format(type(est_object.inputs_sub[k])))
                # raise Exception
            for i in est_object.inputs_sub[k]:
                # print(est_object.inputs_sub[k])
                # print(i)
                if est_object.fixedtraj == True or est_object.fixedy == True:
                    if est_object.fixedtraj == True:
                        for j in est_object.yfixtraj.keys():
                            for l in est_object.yfixtraj[j]:
                                if i == l:
                                    # print('herel:fixedy', l)
                                    if not isinstance(est_object.yfixtraj[j], list):
                                        print("wrong type for yfixtraj {}".format(type(est_object.yfixtraj[j])))
                                    reft = trajectories[(k, i)]
                                    est_object.fix_from_trajectory(k, i, reft)
                    if est_object.fixedy == True:
                        for j in est_object.yfix.keys():
                            for l in est_object.yfix[j]:
                                if i == l:
                                    # print('herel:fixedy',l)
                                    if not isinstance(est_object.yfix[j], list):
                                        print("wrong type for yfix {}".format(type(est_object.yfix[j])))
                                    for key in est_object.model.alltime.value:
                                        vark = getattr(est_object.model, k)
                                        vark[key, i].set_value(key)
                                        vark[key, i].fix()  # since these are inputs we need to fix this
                else:
                    print("A trajectory or fixed input is missing for {}\n".format((k, i)))
    """/end inputs section"""
    
    return None

