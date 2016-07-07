from pyomo.environ import *
from pyomo.dae import *
#from kipet.ResultsObject import *
from kipet.sim.PyomoSimulator import *
from pyomo.core.base.expr import Expr_if

import copy

class ParameterEstimator(PyomoSimulator):
    def __init__(self,model):
        super(ParameterEstimator, self).__init__(model)
        
    def run_sim(self,solver,**kdws):
        raise NotImplementedError("ParameterEstimator object does not have run_sim method. Call run_opt")       
        
    def _solve_extended_model(self,
                              sigma_sq,
                              optimizer,
                              tee=False):
        if not self._spectra_given:
            raise NotImplementedError("Extended model requires spectral data model.D[ti,lj]")
        
        keys = sigma_sq.keys()
        for k in self._mixture_components:
            if k not in keys:
                sigma_sq[k] = 1.0

        if not sigma_sq.has_key('device'):
            sigma_sq['device'] = 1.0
            
        m = ConcreteModel()
        m.add_component('dae',self.model)

        # try
        m.D_bar = Var(m.dae.meas_times,
                      m.dae.meas_lambdas)

        def rule_D_bar(m,t,l):
            return m.D_bar[t,l] == sum(m.dae.C[t,k]*m.dae.S[l,k] for k in m.dae.mixture_components)
        m.D_bar_constraint = Constraint(m.dae.meas_times,
                                        m.dae.meas_lambdas,
                                        rule=rule_D_bar)

        # estimation
        def rule_objective(m):
            expr = 0
            for t in m.dae.meas_times:
                for l in m.dae.meas_lambdas:
                    #D_bar = sum(m.dae.C[t,k]*m.dae.S[l,k] for k in m.dae.mixture_components)
                    #expr+= (m.D[t,l] - D_bar)**2/(sigma_sq['device'])
                    expr+= (m.dae.D[t,l] - m.D_bar[t,l])**2/(sigma_sq['device'])

            for t in m.dae.meas_times:
                expr += sum((m.dae.C[t,k]-m.dae.Z[t,k])**2/(sigma_sq[k]) for k in m.dae.mixture_components)
            return expr
        m.objective = Objective(rule=rule_objective)
        
        solver_results = optimizer.solve(m,tee=tee)
        m.del_component('dae')
    
    def run_opt(self,solver,**kwds):

        solver_opts = kwds.pop('solver_opts', dict())
        variances = kwds.pop('variances',dict())
        tee = kwds.pop('tee',False)
    
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
            self._solve_extended_model(variances,opt,tee=tee)
            
        results = ResultsObject()

        results.load_from_pyomo_model(self.model,
                                      to_load=['Z','dZdt','X','dXdt','C','S'])
            
        # computes D from the estimated values
        d_results = []
        for i,t in enumerate(self._meas_times):
            for j,l in enumerate(self._meas_lambdas):
                suma = 0.0
                for w,k in enumerate(self._mixture_components):
                    C =  results.C[k][t]  
                    S = results.S[k][l]
                    suma+= C*S
                d_results.append(suma)
                    
        d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
        results.D = pd.DataFrame(data=d_array,
                                 columns=self._meas_lambdas,
                                 index=self._meas_times)
            
        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name].value

        results.P = param_vals

        return results

            
            
