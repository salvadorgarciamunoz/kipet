import numpy as np

def g_handling_status_messages(est_object):
    if est_object.G_contribution == 'unwanted_G':
        print("\nType of unwanted contributions not set, so assumed that it is time-variant.\n")
        est_object.G_contribution = 'time_variant_G'
    elif est_object.G_contribution == 'time_variant_G':
        print("\nTime-variant unwanted contribution is involved.\n")
    elif est_object.G_contribution == 'time_invariant_G_decompose':
        print("\nTime-invariant unwanted contribution is involved and G can be decomposed.\n")
    elif est_object.G_contribution == 'time_invariant_G_no_decompose':
        print("\nTime-invariant unwanted contribution is involved but G cannot be decomposed.\n")
    return None
 
def decompose_G_test(est_object, St, Z_in):
    """Check whether or not G can be decomposed"""
    
    if St == dict() and Z_in == dict():
        raise RuntimeError('Because time-invariant unwanted contribution is chosen, please provide information of St or Z_in to build omega matrix.')
    
    omega_list = [St[i] for i in St.keys()]
    omega_list += [Z_in[i] for i in Z_in.keys()]
    omega_sub = np.array(omega_list)
    rank = np.linalg.matrix_rank(omega_sub)
    cols = omega_sub.shape[1]
    rko = cols - rank
    
    if rko > 0:
        est_object.time_invariant_G_decompose = True
        est_object.G_contribution = 'time_invariant_G_decompose'
    else:
        est_object.time_invariant_G_no_decompose = True
        est_object.G_contribution = 'time_invariant_G_no_decompose'
    return None