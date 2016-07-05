from pyomo.environ import *
from pyomo.dae import *
#from kipet.ResultsObject import *
from kipet.sim.PyomoSimulator import *
from pyomo.core.base.expr import Expr_if

import copy

class ParameterEstimator(PyomoSimulator):
    def __init__(self,model):
        super(ParameterEstimator, self).__init__(model)
        # add variance variables
        self.model.device_variance = Var(within=NonNegativeReals, initialize=1)
        self.model.device_variance.fixed = True
        self.model.sigma_sq = Var(self.model.mixture_components,
                               within=NonNegativeReals,
                               initialize=1)
        for k in self.model.mixture_components:
            self.model.sigma_sq.fixed = True

        

        # try
        self.model.D_bar = Var(self.model.meas_times,
                               self.model.meas_lambdas)

        def rule_D_bar(m,t,l):
            return m.D_bar[t,l] == sum(m.C[t,k]*m.S[l,k] for k in m.mixture_components)
        self.model.D_bar_constraint = Constraint(self.model.meas_times,
                                                 self.model.meas_lambdas,
                                                 rule=rule_D_bar)

        # estimation
        def rule_objective(m):
            expr = 0
            for t in m.meas_times:
                for l in m.meas_lambdas:
                    D_bar = sum(m.C[t,k]*m.S[l,k] for k in m.mixture_components)
                    #expr+= (m.D[t,l] - D_bar)**2/(m.device_variance)
                    expr+= (m.D[t,l] - m.D_bar[t,l])**2/(m.device_variance)

            for t in m.meas_times:
                expr += sum((m.C[t,k]-m.Z[t,k])**2/(m.sigma_sq[k]) for k in m.mixture_components)
            return expr
        self.model.direct_estimation = Objective(rule=rule_objective)
        
    def run_sim(self,solver,tee=False,solver_opts={}):
        raise NotImplementedError("ParameterEstimator object does not have run_sim method. Call run_opt")

    def _solve_extended_model(self,
                              sigma_sq,
                              optimizer,
                              tee=False):

        keys = sigma_sq.keys()
        for k in self._mixture_components:
            if k not in keys:
                print("WARNING: Variance of component {} not found. Default 1.0".format(k))
                sigma_sq[k] = 1.0

        if not sigma_sq.has_key('device'):
            print("WARNING: Variance of device not found. Default 1.0")
            sigma_sq['device'] = 1.0
            
        m = ConcreteModel()
        m.add_component('dae',self.model)

        # try
        m.D_bar = Var(m.dae.meas_times,
                      m.dae.meas_lambdas)

        def rule_D_bar(m,t,l):
            return m.D_bar[t,l] == sum(m.C[t,k]*m.S[l,k] for k in m.dae.mixture_components)
        m.D_bar_constraint = Constraint(m.dae.meas_times,
                                        m.dae.meas_lambdas,
                                        rule=rule_D_bar)

        # estimation
        def rule_objective(m):
            expr = 0
            for t in m.dae.meas_times:
                for l in m.dae.meas_lambdas:
                    D_bar = sum(m.C[t,k]*m.S[l,k] for k in m.dae.mixture_components)
                    #expr+= (m.D[t,l] - D_bar)**2/(sigma_sq['device'])
                    expr+= (m.D[t,l] - m.D_bar[t,l])**2/(sigma_sq['device'])

            for t in m.dae.meas_times:
                expr += sum((m.C[t,k]-m.Z[t,k])**2/(sigma_sq[k]) for k in m.dae.mixture_components)
            return expr
        m.objective = Objective(rule=rule_objective)
        
        solver_results = optimizer.solve(m,tee=tee)
        m.del_component('dae')
    
    def run_opt(self,solver,tee=False,solver_opts={},variances={}):
        
        Z_var = self.model.Z 
        dZ_var = self.model.dZdt

        X_var = self.model.X
        dX_var = self.model.dXdt
        if self._discretized is False:
            raise RuntimeError('apply discretization first before runing simulation')
        
        # fixes the sigmas
        for k,v in variances.iteritems():
            if k=='device':
                self.model.device_variance.value = v
                self.model.device_variance.fixed = True
            else:
                self.model.sigma_sq[k] = v
                self.model.sigma_sq[k].fixed = True

        #check if all sigmas are fixed
        all_sigmas_fixed = True
        if self.model.device_variance.fixed == False:
            all_sigmas_fixed = False
        for k in self._mixture_components:
            if self.model.sigma_sq[k].fixed == False:
                all_sigmas_fixed = False
                break
            
        #self.model.direct_estimation.pprint()
        if not all_sigmas_fixed:
            warnings.warn('Not all variances are specified. Unspecified variances fixed to 1.0.\n')
        
        # Look at the output in results
        opt = SolverFactory(solver)

        for key, val in solver_opts.iteritems():
            opt.options[key]=val                
                
        solver_results = opt.solve(self.model,tee=tee)
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

        results.sigma_sq = dict()
        for name in self._mixture_components:
            results.sigma_sq[name] = self.model.sigma_sq[name].value
            results.device_variance = self.model.device_variance.value
        
        return results

            
            
