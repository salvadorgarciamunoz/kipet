from pyomo.environ import *
from pyomo.dae import *
from kipet.opt.Optimizer import *
from pyomo.core.base.expr import Expr_if

import copy

class ParameterEstimator(Optimizer):
    """Optimizer for parameter estimation.

    Attributes:
        model (model): Pyomo model.

    """
    def __init__(self,model):
        super(ParameterEstimator, self).__init__(model)
        
    def run_sim(self,solver,**kdws):
        raise NotImplementedError("ParameterEstimator object does not have run_sim method. Call run_opt")       
        
    def _solve_extended_model(self,
                              sigma_sq,
                              optimizer,
                              tee=False,
                              with_d_vars=False):

        """Solves estimation based on spectral data.

           This method is not intended to be used by users directly
        Args:
            sigma_sq (dict): variances 
            
            optimizer (SolverFactory): Pyomo Solver factory object
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not

            with_d_vars (bool,optional): flag to the optimizer whether to add 
            variables and constraints for D_bar(i,j)
        
        Returns:
            None
        """
        
        if not self._spectra_given:
            raise NotImplementedError("Extended model requires spectral data model.D[ti,lj]")

        keys = sigma_sq.keys()
        for k in self._mixture_components:
            if k not in keys:
                sigma_sq[k] = 1.0

        if not sigma_sq.has_key('device'):
            sigma_sq['device'] = 1.0
            
        m = self.model

        if with_d_vars:
            m.D_bar = Var(m.meas_times,
                          m.meas_lambdas)

            def rule_D_bar(m,t,l):
                return m.D_bar[t,l] == sum(m.C[t,k]*m.S[l,k] for k in m.mixture_components)
            m.D_bar_constraint = Constraint(m.meas_times,
                                            m.meas_lambdas,
                                            rule=rule_D_bar)

        # estimation
        def rule_objective(m):
            expr = 0
            for t in m.meas_times:
                for l in m.meas_lambdas:
                    if with_d_vars:
                        expr+= (m.D[t,l] - m.D_bar[t,l])**2/(sigma_sq['device'])
                    else:
                        D_bar = sum(m.C[t,k]*m.S[l,k] for k in m.mixture_components)
                        expr+= (m.D[t,l] - D_bar)**2/(sigma_sq['device'])

            for t in m.meas_times:
                expr += sum((m.C[t,k]-m.Z[t,k])**2/(sigma_sq[k]) for k in m.mixture_components)
            return expr

        m.objective = Objective(rule=rule_objective)
        
        solver_results = optimizer.solve(m,tee=tee)

        if with_d_vars:
            m.del_component('D_bar')
            m.del_component('D_bar_constraint')
        m.del_component('objective')

    def _compute_B_matrix(self,**kwds):
        """Builds B matrix for calculation of covariances

           This method is not intended to be used by users directly

        Args:
            variances (dict,optional): variances 
            
        Returns:
            None
        """
        variances = kwds.pop('variances',dict())
        
        # add check for model already solved

        # fixes variances that are not passed 
        keys = variances.keys()
        for k in self._mixture_components:
            if k not in keys:
                variances[k] = 1.0

        if not variances.has_key('device'):
            variances['device'] = 1.0

        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_components
        # this changes depending on the order of the suffixes passed to sipopt
        self.B_matrix = np.zeros((nc*(nw*nt),nw*nt))
        for i,t in enumerate(self.model.meas_times):
            for j,l in enumerate(self.model.meas_lambdas):
                for k,c in enumerate(self.model.mixture_components):
                    r_idx1 = k*nt+i
                    r_idx2 = k*nw+j+nc*nt
                    c_idx = i+j*nt
                    self.B_matrix[r_idx1,c_idx] = -2*self.model.S[l,c]/variances['device']
                    self.B_matrix[r_idx2,c_idx] = -2*self.model.C[t,c]/variances['device']

    def _compute_Vd_matrix(self,**kwds):
        """Builds d covariance matrix

           This method is not intended to be used by users directly

        Args:
            variances (dict,optional): variances 
            
        Returns:
            None
        """
        variances = kwds.pop('variances',dict())
        
        # add check for model already solved

        # fixes variances that are not passed 
        keys = variances.keys()
        for k in self._mixture_components:
            if k not in keys:
                variances[k] = 0.0

        if not variances.has_key('device'):
            variances['device'] = 0.0

        row =  []
        col =  []
        data = []
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_components
        for i,t in enumerate(self.model.meas_times):
            for j,l in enumerate(self.model.meas_lambdas):
                for q,tt in enumerate(self.model.meas_times):
                    for p,ll in enumerate(self.model.meas_lambdas):
                        if i==q and j!=p:
                            val = sum(variances[c]*self.model.S[l,c]*self.model.S[ll,c] for c in self.model.mixture_components)
                            row.append(i*nw+j)
                            col.append(q*nw+p)
                            data.append(val)
                        if i==q and j==p:
                            val = sum(variances[c]*self.model.S[l,c]**2 for c in self.model.mixture_components)+variances['device']
                            row.append(i*nw+j)
                            col.append(q*nw+p)
                            data.append(val)

        nd = nt*nw
        self.Vd_matrix = scipy.sparse.coo_matrix((data, (row, col)),
                                                 shape=(nd,nd))
                                

    def run_opt(self,solver,**kwds):

        """ Solves parameter estimation problem.
        
        Args:
            solver (str): name of the nonlinear solver to used
          
            solver_opts (dict, optional): options passed to the nonlinear solver
        
            variances (dict, optional): map of component name to noise variance. The
            map also contains the device noise variance
            
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
            
            with_d_vars (bool,optional): flag to the optimizer whether to add 
            variables and constraints for D_bar(i,j)

        Returns:
            Results object with loaded results

        """

        solver_opts = kwds.pop('solver_opts', dict())
        variances = kwds.pop('variances',dict())
        tee = kwds.pop('tee',False)
        with_d_vars = kwds.pop('with_d_vars',False)
        
        if not self.model.time.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')
        
        # Look at the output in results
        opt = SolverFactory(solver)

        for key, val in solver_opts.iteritems():
            opt.options[key]=val                

        active_objectives = [o for o in self.model.component_map(Objective,active=True)]
        if active_objectives:
            print("WARNING: The model has an active objective. Running optimization with models objective.\n To solve optimization with default objective (Weifengs) deactivate all objectives in the model.")
            solver_results = opt.solve(self.model,tee=tee)
        else:
            self._solve_extended_model(variances,opt,tee=tee,with_d_vars=with_d_vars)
            
        results = ResultsObject()

        results.load_from_pyomo_model(self.model,
                                      to_load=['Z','dZdt','X','dXdt','C','S'])
            
        self.compute_D_given_SC(results)
        
        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name].value

        results.P = param_vals

        return results

            
            
