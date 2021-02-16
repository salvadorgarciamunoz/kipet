"""
Holds the alternate method used in VarianceEstimator
"""
from pyomo.core import (
    log,
    value,
    )
from pyomo.environ import (
    Objective,
    SolverFactory,
    Param,
    )

from kipet.core_methods.ResultsObject import ResultsObject

def run_alternate_method(var_est_object, solver, run_opt_kwargs):
    """Calls the alternative method - Short et al 2020"""
    
    solver_opts = run_opt_kwargs.pop('solver_opts', dict())
    init_sigmas = run_opt_kwargs.pop('initial_sigmas', float())
    tee = run_opt_kwargs.pop('tee', False)
    tol = run_opt_kwargs.pop('tolerance', 5.0e-5)
    A = run_opt_kwargs.pop('subset_lambdas', None)
    secant_point2 = run_opt_kwargs.pop('secant_point', None)
    individual_species = run_opt_kwargs.pop('individual_species', False)
    
    solver = 'ipopt'
    
    nu_squared = var_est_object.solve_max_device_variance(
                                           solver,
                                           tee=tee,
                                           subset_lambdas=A, 
                                           solver_opts=solver_opts)
        
    if secant_point2 is not None:
        second_point = secant_point2
    else:
        second_point = init_sigmas*10
    itersigma = dict()
    
    itersigma[1] = init_sigmas
    itersigma[0] = second_point
    
    iterdelta = dict()
    
    count = 1
    tol = tol
    funcval = 1000
    tee = False
    
    iterdelta[0] = _solve_delta_given_sigma(var_est_object,
                                            solver,
                                            tee=tee, 
                                            subset_lambdas=A, 
                                            solver_opts=solver_opts, 
                                            init_sigmas=itersigma[0])

    while abs(funcval) >= tol:
        print("Overall sigma value at iteration", count,": ", itersigma[count])
        
        new_delta = _solve_delta_given_sigma(var_est_object,
                                             solver, 
                                             tee=tee, 
                                             subset_lambdas=A, 
                                             solver_opts=solver_opts, 
                                             init_sigmas=itersigma[count])
        print("New delta_sq val: ", new_delta)
        
        iterdelta[count] = new_delta
        
        def func1(nu_squared, new_delta, init_sigmas):
            sigmult = 0
            nwp = len(var_est_object._meas_lambdas)
            for l in var_est_object._meas_lambdas:
                for k in var_est_object.component_set:
                   sigmult += value(var_est_object.model.S[l, k])
            funcval = nu_squared - new_delta - init_sigmas*(sigmult/nwp)
            return funcval, sigmult
       
        funcval, sigmult = func1(nu_squared, new_delta, itersigma[count])

        if abs(funcval) <= tol:
            break
        else:
            denom_secant = func1(nu_squared, new_delta, itersigma[count])[0] - func1(nu_squared, iterdelta[count-1], itersigma[count - 1])[0]
            itersigma[count + 1] = itersigma[count] - func1(nu_squared, new_delta, itersigma[count])[0]*((itersigma[count]-itersigma[count-1])/denom_secant)
            
            if itersigma[count + 1] < 0:
                itersigma[count + 1] = -1*itersigma[count + 1]
            count += 1
            
    if individual_species:
        print("Solving for individual species' variance based on the obtained delta")
        max_likelihood_val, sigma_vals, stop_it, results = \
            _solve_sigma_given_delta(var_est_object,
                                          solver, 
                                          subset_lambdas= A, 
                                          solver_opts=solver_opts, 
                                          tee=tee, 
                                          delta=new_delta)
    else:
        print("The overall model variance is: ", itersigma[count])
        #sigma_vals = {k: abs(itersigma[count]) for k in component_set}
        
        sigma_vals = {}
        for k in var_est_object.component_set:
            sigma_vals[k] = abs(itersigma[count])
        
    print(f'sigma_vals: {sigma_vals}')
    results = ResultsObject()
    
    vars_to_load = ['Z', 'dZdt', 'X', 'dXdt', 'C', 'Cs', 'S', 'Y']
    if not hasattr(var_est_object, '_abs_components'):
        vars_to_load.remove('Cs')
    results.load_from_pyomo_model(var_est_object.model, to_load=vars_to_load)
     
    results.P = {name: var_est_object.model.P[name].value for name in var_est_object.model.parameter_names}
    results.sigma_sq = sigma_vals
    results.sigma_sq['device'] = new_delta

    return results

def run_direct_sigmas_method(var_est_object, solver, run_opt_kwargs):
    """Calls the direct sigma method"""
        
    solver_opts = run_opt_kwargs.pop('solver_opts', dict())
    tee = run_opt_kwargs.pop('tee', False)
    A = run_opt_kwargs.pop('subset_lambdas', None)
    fixed_device_var = run_opt_kwargs.pop('fixed_device_variance', None)
    device_range = run_opt_kwargs.pop('device_range', None)
    num_points = run_opt_kwargs.pop('num_points', None)
    
    print("Solving for sigmas assuming known device variances")
    direct_or_it = "it"
    
    if device_range:
        if not isinstance(device_range, tuple):
            print("device_range is of type {}".format(type(device_range)))
            print("It should be a tuple")
            raise Exception
        elif device_range[0] > device_range[1]:
            print("device_range needs to be arranged in order from lowest to highest")
            raise Exception
        else:
            print("Device range means that we will solve iteratively for different delta values in that range")
    if device_range and not num_points:
        print("Need to specify the number of points that we wish to evaluate in the device range")
    if not num_points:
        pass
    elif not isinstance(num_points, int):  
        print("num_points needs to be an integer!")
        raise Exception
    if not device_range:
        device_range = list() 
    if not device_range and not num_points:
        direct_or_it = "direct"
        print("assessing for the value of delta provided")
        if not fixed_device_var:
            print("If iterative method not selected then need to provide fixed device variance (delta**2)")
            raise Exception
        else:
            if not isinstance(fixed_device_var, float):
                raise Exception("fixed device variance needs to be of type float")
    
    if direct_or_it in ["it"]:
        dist = abs((device_range[1] - device_range[0])/num_points)
        
        max_likelihood_vals = []
        delta_vals = []
        iteration_counter = []
        delta = device_range[0]
        
        count = 0
        
        print('*** Starting Variance Iterations ***')
       
        while delta < device_range[1]:
            
            print(f"Iteration: {count}\tdelta_sq: {delta}")
            max_likelihood_val, sigma_vals, stop_it, results = \
                _solve_sigma_given_delta(var_est_object,solver, 
                                              subset_lambdas=A, 
                                              solver_opts=solver_opts, 
                                              tee=tee,
                                              delta=delta)
            
            if max_likelihood_val >= 5000:
                max_likelihood_vals.append(5000)
                delta_vals.append(log(delta))
                iteration_counter.append(count)
                max_likelihood_vals.append(max_likelihood_val)
                delta_vals.append(log(delta))
            else:
                iteration_counter.append(count)
                
            delta = delta + dist
            count += 1 

    else:
        max_likelihood_val, sigma_vals, stop_it = \
            _solve_sigma_given_delta(var_est_object,
                                          solver, 
                                          subset_lambdas=A, 
                                          solver_opts=solver_opts, 
                                          tee=tee,
                                          delta=fixed_device_var)
   
    results = ResultsObject()
    
    vars_to_load = ['Z', 'dZdt', 'X', 'dXdt', 'C', 'Cs', 'S', 'Y']
    if not hasattr(var_est_object, '_abs_components'):
        vars_to_load.remove('Cs')
    results.load_from_pyomo_model(var_est_object.model, to_load=vars_to_load)
    
    results.P = {name: var_est_object.model.P[name].value for name in var_est_object.model.parameter_names}
    results.sigma_sq = sigma_vals
    results.sigma_sq['device'] = delta
        
    return results

def _solve_delta_given_sigma(var_est_object, solver, **kwds):
    """Solves the maximum likelihood formulation with fixed sigmas in order to
    obtain delta. This formulation is highly unstable as the problem is ill-posed
    with the optimal solution being that C-Z = 0. Should not use.

       This method is not intended to be used by users directly

    Args:
        init_sigmas (dict): variances 
    
        tee (bool,optional): flag to tell the optimizer whether to stream output
        to the terminal or not
    
        profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
        Default False
    
        subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
        (Weifeng paper). Default all wavelengths.

    Returns:

        None

    """   
    solver_opts = kwds.pop('solver_opts', dict())
    tee = kwds.pop('tee', False)
    set_A = kwds.pop('subset_lambdas', list())
    profile_time = kwds.pop('profile_time', False)
    sigmas = kwds.pop('init_sigmas', dict() or float() )

    sigmas_sq = dict()
    if not set_A:
        set_A = var_est_object._meas_lambdas
    
    if isinstance(sigmas, float):
        for k in var_est_object.component_set:
            sigmas_sq[k] = sigmas
    
    elif isinstance(sigmas, dict):
        keys = sigmas.keys()
        for k in var_est_object.component_set:
            if k not in keys:
                sigmas_sq[k] = sigmas
       
    print("Solving delta from given sigmas\n")
    print(sigmas_sq)
   
    var_est_object._warn_if_D_negative()  
   
    obj = 0.0
    ntp = len(var_est_object._meas_times)
    nwp = len(var_est_object._meas_lambdas) 
    inlog = 0
    nc = len(var_est_object.component_set)
    for t in var_est_object._meas_times:
        for l in set_A:
            D_bar = sum(var_est_object.model.C[t, k] * var_est_object.model.S[l, k] for k in var_est_object.component_set)
            #D_bar = sum(var_est_object.model.C[t, k] * var_est_object.model.S[l, k] for k in var_est_object._sublist_components)
            inlog += (var_est_object.model.D[t, l] - D_bar)**2
    # Orig had sublist in both parts - is this an error?

    # Concentration - correct
    # Move this to objectives module
    for t in var_est_object._meas_times:
        for k in var_est_object._sublist_components:
            #obj += conc_objective(var_est_object.model, sigma=sigmas_sq)
            obj += 0.5*((var_est_object.model.C[t, k] - var_est_object.model.Z[t, k])**2)/(sigmas_sq[k])
            
    obj += (nwp)*log((inlog)+1e-12)     
    var_est_object.model.init_objective = Objective(expr=obj)
    
    opt = SolverFactory(solver)

    for key, val in solver_opts.items():
        opt.options[key]=val
    solver_results = opt.solve(var_est_object.model,
                               tee=tee,
                               report_timing=profile_time)
    residuals = (value(var_est_object.model.init_objective))
    
    print("residuals: ", residuals)

    for k, v in var_est_object.model.P.items():
        print(k, v.value)
        
    etaTeta = 0
    for t in var_est_object._meas_times:
        for l in set_A:
            D_bar = sum(value(var_est_object.model.C[t, k]) * value(var_est_object.model.S[l, k]) for k in var_est_object.component_set)
            etaTeta += (value(var_est_object.model.D[t, l]) - D_bar)**2
    
    deltasq = etaTeta/(ntp*nwp)  
    var_est_object.model.del_component('init_objective')
   
    return deltasq

def _solve_sigma_given_delta(var_est_object, solver, **kwds):
    """Solves the maximum likelihood formulation to determine the model variance from a
    given device variance.

       This method is not intended to be used by users directly

    Args:
        delta (float): the device variance squared 
    
        tee (bool,optional): flag to tell the optimizer whether to stream output
        to the terminal or not
    
        profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
        Default False
    
        subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
        (Weifeng paper). Default all wavelengths.
        
        solver_opts (dict, optional): dictionary containing solver options for IPOPT

    Returns:

        residuals (float): the objective function value from the optimization
        
        variancesdict (dict): dictionary containing the model variance values
        
        stop_it (bool): boolean indicator showing whether no solution was found 
                            (True) or if there is a solution (False)
                            
        solver_results: Variance estimation model results, similar to VarianceEstimator 
                                    results from previous method

    """   
    solver_opts = kwds.pop('solver_opts', dict())
    tee = kwds.pop('tee', False)
    set_A = kwds.pop('subset_lambdas', list())
    profile_time = kwds.pop('profile_time', False)
    delta = kwds.pop('delta', dict())
    species_list = kwds.pop('subset_components', None)

    model = var_est_object.model.clone()

    if not set_A:
        set_A = var_est_object._meas_lambdas
        
    if not var_est_object._sublist_components:
        list_components = []
        if species_list is None:
            list_components = [k for k in var_est_object._mixture_components]
            
        else:
            for k in species_list:
                if k in var_est_object._mixture_components:
                    list_components.append(k)
                else:
                    warnings.warn("Ignored {} since is not a mixture component of the model".format(k))

        var_est_object._sublist_components = list_components
    
    var_est_object._warn_if_D_negative()  
    ntp = len(var_est_object._meas_times)
    obj = 0.0
   
    for t in var_est_object._meas_times:
        for l in set_A:
            D_bar = sum(var_est_object.model.C[t, k] * var_est_object.model.S[l, k] for k in var_est_object.component_set)
            obj += 0.5/delta*(var_est_object.model.D[t, l] - D_bar)**2

    inlog = {k: 0 for k in var_est_object.component_set}
    var_est_object.model.eps = Param(initialize = 1e-8)  
    
    variancesdict = {k: 0 for k in var_est_object.component_set}
            
    for t in var_est_object._meas_times:
        for k in var_est_object.component_set:
            inlog[k] += ((var_est_object.model.C[t, k] - var_est_object.model.Z[t, k])**2)
    
    for k in var_est_object.component_set:
        obj += 0.5*ntp*log(inlog[k]/ntp + var_est_object.model.eps)
    
    var_est_object.model.init_objective = Objective(expr=obj)

    opt = SolverFactory(solver)

    for key, val in solver_opts.items():
        opt.options[key]=val
    try:
        solver_results = opt.solve(var_est_object.model,
                               tee=tee,
                               report_timing=profile_time)

        residuals = (value(var_est_object.model.init_objective))
        for t in var_est_object._allmeas_times:
            for k in var_est_object.component_set:
                variancesdict[k] += 1 / ntp*((value(var_est_object.model.C[t, k]) - value(var_est_object.model.Z[t, k]))**2)
        
        print("Variances")
        for k in var_est_object.component_set:
            print(k, variancesdict[k])
        
        print("Parameter estimates")
        for k, v in var_est_object.model.P.items():
            print(k, v.value)
           
        stop_it = False
        
    except:
        print("FAILED AT THIS ITERATION")
        variancesdict = None
        residuals = 0
        stop_it = True
        solver_results = None
        for k, v in var_est_object.model.P.items():
            print(k, v.value)
        var_est_object.model = model
        for k, v in var_est_object.model.P.items():
            print(k, v.value)
    var_est_object.model.del_component('init_objective')
    var_est_object.model.del_component('eps')

    return residuals, variancesdict, stop_it, solver_results