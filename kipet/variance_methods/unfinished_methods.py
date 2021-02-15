"""
This method is not currently used
"""

def _solve_iterative_init(self, solver, **kwds):
    """This method first fixes params and makes C = Z to solve for S. Requires fixed delta and sigmas.
    following this, the parameters are freed to users bounds and full problem is solved with fixed variances.
    This function is only meant to be used as part of a full variance estimation strategy. Currently not
    implemented in any of the strategies, however it can be useful in future implementations to initialize.
    
    Not meant to be directly used by users

    Args:
        sigma_sq (dict): variances 
    
        tee (bool,optional): flag to tell the optimizer whether to stream output
        to the terminal or not
    
        profile_time (bool,optional): flag to tell pyomo to time the construction and solution of the model. 
        Default False
    
        subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
        (Weifeng paper). Default all wavelengths.
        
        solver_options (dict, optional): Solver options for IPOPT

    Returns:

        None

    """   
    solver_opts = kwds.pop('solver_options', dict())
    tee = kwds.pop('tee', True)
    set_A = kwds.pop('subset_lambdas', list())
    profile_time = kwds.pop('profile_time', False)
    sigmas_sq = kwds.pop('variances', dict())

    if not set_A:
        set_A = self._meas_lambdas
    
    keys = sigmas_sq.keys()
    for k in self.component_set:
        print(k)
        if k not in keys:
            sigmas_sq[k] = 0.0
    
    self._warn_if_D_negative()
    list_components = [k for k in self._mixture_components]
             
    print("Solving Initialization Problem with fixed parameters\n")
    original_bounds = dict()
    for v, k in self.model.P.items():
        low = value(self.model.P[v].lb)
        high = value(self.model.P[v].ub)
        original_bounds[v] = (low, high)
        ub = value(self.model.P[v])
        lb = ub
        self.model.P[v].setlb(lb)
        self.model.P[v].setub(ub)
   
    obj = 0.0
   
    for t in self._meas_times:
        for l in set_A:
            D_bar = sum(self.model.Z[t, k] * self.model.S[l, k] for k in self.component_set)
            obj += (self.model.D[t, l] - D_bar) ** 2
    
    self.model.init_objective = Objective(expr=obj)
    opt = SolverFactory(solver)
    for key, val in solver_opts.items():
        opt.options[key]=val
    solver_results = opt.solve(self.model,
                               tee=tee,
                               report_timing=profile_time)
    
    for k, v in self.model.P.items():
        print(k, v.value)
    
    for t in self._allmeas_times:
        for k in self._mixture_components:
            if k in sigmas_sq and sigmas_sq[k] > 0.0:
                self.model.C[t, k].value = np.random.normal(self.model.Z[t, k].value, sigmas_sq[k])
            else:
                self.model.C[t, k].value = self.model.Z[t, k].value
            
    self.model.del_component('init_objective')
    
    for v, k in self.model.P.items():
        ub = original_bounds[v][1]
        lb = original_bounds[v][0]
        self.model.P[v].setlb(lb)
        self.model.P[v].setub(ub)
        
    m = self.model

    m.D_bar = Var(m.meas_times,
                  m.meas_lambdas)
    
    def rule_D_bar(m, t, l):
        return m.D_bar[t, l] == sum(getattr(m, self.component_var)[t, k] * m.S[l, k] for k in self.component_set)
    
    m.D_bar_constraint = Constraint(m.meas_times,
                                    m.meas_lambdas,
                                    rule=rule_D_bar)

    # estimation
    def rule_objective(m):
        expr = 0
        for t in m.meas_times:
            for l in m.meas_lambdas:

                expr += (m.D[t, l] - m.D_bar[t, l]) ** 2 / (sigmas_sq['device'])

        #expr = spectra_objective(m)

        # This part is not even used????
        # second_term = 0.0
        # for t in m.meas_times:
        #     second_term += sum((m.C[t, k] - m.Z[t, k]) ** 2 / sigmas_sq[k] for k in list_components)
        return expr

    m.objective = Objective(rule=rule_objective) 
    
    opt = SolverFactory(solver)

    for key, val in solver_opts.items():
        opt.options[key]=val
    
    print("Solving problem with unfixed parameters")
    solver_results = opt.solve(self.model,
                               tee=tee,
                               report_timing=profile_time)
    
    for k, v in self.model.P.items():
        print(k, v.value)      

    m.del_component('objective')
    self.model = m