"""
Module to hold the method developed by Chen et al. 2016
"""
from pyomo.environ import Objective, SolverFactory


def solve_C(var_est_object, solver, **kwds):
    """Solves formulation 23 in Chen. et al 2016 for C

    :param VarianceEstimator var_est_object: The variance estimation object
    :param str solver: The solver being used (currently not used)
    :param dict kwds: The dict of user settings passed on from the ReactionModel

    :return: None

    """
    solver_opts = kwds.pop('solver_opts', dict())
    tee = kwds.pop('tee', False)
    update_nl = kwds.pop('update_nl', False)
    profile_time = kwds.pop('profile_time', False)

    # D_bar obj
    obj = 0.0
    for t in var_est_object.model.times_spectral:
        for l in var_est_object._meas_lambdas:
            D_bar = sum(var_est_object.model.S[l, k].value*var_est_object.C_model.C[t, k] for k in var_est_object.comps['absorbing'])
            obj += (var_est_object.model.D[t, l]-D_bar)**2
     
    var_est_object.C_model.objective = Objective(expr=obj)
             
    if profile_time:
        print('-----------------Solve_C--------------------')

    opt = SolverFactory(solver)

    for key, val in solver_opts.items():
        opt.options[key]=val
         
    solver_results = opt.solve(var_est_object.C_model,
                               logfile=var_est_object._tmp4,
                               tee=tee,
                               #keepfiles=True,
                               report_timing=profile_time)

    var_est_object.C_model.del_component('objective')
     
    for t in var_est_object.model.allmeas_times:
        for c in var_est_object.comps['absorbing']:
            var_est_object.model.C[t, c].value = var_est_object.C_model.C[t, c].value

    return None


def solve_S(var_est_object, solver, **kwds):
    """Solves formulation 23 in Chen. et al 2016 for S

    :param VarianceEstimator var_est_object: The variance estimation object
    :param str solver: The solver being used (currently not used)
    :param dict kwds: The dict of user settings passed on from the ReactionModel

    :return: None

    """
    solver_opts = kwds.pop('solver_opts', dict())
    tee = kwds.pop('tee', False)
    update_nl = kwds.pop('update_nl', False)
    profile_time = kwds.pop('profile_time', False)
        
    for l in var_est_object._meas_lambdas:
        for c in var_est_object.comps['unknown_absorbance']:
            var_est_object.S_model.S[l, c].value = var_est_object.model.S[l, c].value
            if hasattr(var_est_object.model, 'known_absorbance'):
                if c in var_est_object.model.known_absorbance:
                    if var_est_object.model.S[l, c].value != var_est_object.model.known_absorbance_data[c][l]:
                        var_est_object.model.S[l, c].set_value(var_est_object.model.known_absorbance_data[c][l])
                        var_est_object.S_model.S[l, c].fix()
    
    #D_bar obj
    obj = 0.0
    for t in var_est_object.model.times_spectral:
        for l in var_est_object._meas_lambdas:
            D_bar = sum(var_est_object.S_model.S[l, k] * var_est_object.model.Z[t, k].value for k in var_est_object.comps['unknown_absorbance'])
            obj += (D_bar - var_est_object.model.D[t, l]) ** 2
                
    var_est_object.S_model.objective = Objective(expr=obj)

    if profile_time:
        print('-----------------Solve_S--------------------')

    opt = SolverFactory(solver)

    for key, val in solver_opts.items():
        opt.options[key]=val
    solver_results = opt.solve(var_est_object.S_model,
                               logfile=var_est_object._tmp3,
                               tee=tee,
                               #keepfiles=True,
                               #show_section_timing=True,
                               report_timing=profile_time)

    var_est_object.S_model.del_component('objective')
    
    for l in var_est_object._meas_lambdas:
        for c in var_est_object.comps['unknown_absorbance']:
            var_est_object.model.S[l, c].value = var_est_object.S_model.S[l, c].value
            if hasattr(var_est_object.model, 'known_absorbance'):
                if c in var_est_object.model.known_absorbance:
                    if var_est_object.model.S[l, c].value != var_est_object.model.known_absorbance_data[c][l]:
                        var_est_object.model.S[l, c].set_value(var_est_object.model.known_absorbance_data[c][l])
                        var_est_object.S_model.S[l, c].fix()
               

def solve_Z(var_est_object, solver, **kwds):
    """Solves formulation 20 in Chen. et al 2016

    :param VarianceEstimator var_est_object: The variance estimation object
    :param str solver: The solver being used (currently not used)
    :param dict kwds: The dict of user settings passed on from the ReactionModel

    :return: None

    :Keyword Args:

        - solver_opts (dict, optional): options passed to the nonlinear solver
        - tee (bool,optional): flag to tell the optimizer whether to stream output
        to the terminal or not
        - profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model.
        Default False
        - subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem
        (Weifeng paper). Default all wavelengths.

    """
    solver_opts = kwds.pop('solver_opts', dict())
    tee = kwds.pop('tee', False)
    profile_time = kwds.pop('profile_time', False)
    
    for t in var_est_object.model.allmeas_times:
        for k in var_est_object.comps['unknown_absorbance']:
            if hasattr(var_est_object.model, 'non_absorbing'):
                if k not in var_est_object.model.non_absorbing:
                    pass
                else:
                    var_est_object.model.C[t, k].fixed = True

    obj = 0.0
    
    # Conc obj no sigma - will this work with missing data?
    for k in var_est_object.comps['unknown_absorbance']:
        obj += sum((var_est_object.model.C[t, k]-var_est_object.model.Z[t, k])**2 for t in var_est_object.model.times_spectral)
        

    var_est_object.model.z_objective = Objective(expr=obj)
    if profile_time:
        print('-----------------Solve_Z--------------------')
        
    opt = SolverFactory(solver)

    for key, val in solver_opts.items():
        opt.options[key]=val

    solver_results = opt.solve(var_est_object.model,
                               logfile=var_est_object._tmp2,
                               tee=tee,
                               #show_section_timing=True,
                               report_timing=profile_time)

    var_est_object.model.del_component('z_objective')
