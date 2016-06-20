from pyomo.environ import *
from pyomo.dae import *
#from kipet.ResultsObject import *
from kipet.sim.PyomoSimulator import *
from pyomo.core.base.expr import Expr_if

import copy

class Optimizer(PyomoSimulator):
    def __init__(self,model):
        super(Optimizer, self).__init__(model)
        # m = copy.deepcopy(model)
        # add variance variables
        self.model.device_variance = Var(within=NonNegativeReals, initialize=1)
        self.model.sigma_sq = Var(self.model.mixture_components,
                               within=NonNegativeReals,
                               initialize=1)

        # estimation
        def rule_objective(m):
            expr = 0
            for t in m.measurement_times:
                for l in m.measurement_lambdas:
                    current = m.D[t,l] - sum(m.C[t,k]*m.S[l,k] for k in m.mixture_components)
                    expr+= current**2/(m.device_variance)

            for t in m.measurement_times:
                expr += sum((m.C[t,k]-m.Z[t,k])**2/(m.sigma_sq[k]) for k in m.mixture_components)
            return expr
        self.model.direct_estimation = Objective(rule=rule_objective)
        
    def run_sim(self,solver,tee=False,solver_opts={}):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def run_opt(self,solver,tee=False,solver_opts={},std_deviations={}):
        
        Z_var = self.model.Z 
        dZ_var = self.model.dZdt
        if self._discretized is False:
            raise RuntimeError('apply discretization first before runing simulation')
        
        # fixes the sigmas
        for k,v in std_deviations.iteritems():
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
        if all_sigmas_fixed:

            # Look at the output in results
            opt = SolverFactory(solver)

            print solver_opts
            for key, val in solver_opts.iteritems():
                opt.options[key]=val
                print "option:",key, val
                
            solver_results = opt.solve(self.model,tee=tee)
            results = ResultsObject()
                
            c_results = []
            for t in self._times:
                for k in self._mixture_components:
                    c_results.append(Z_var[t,k].value)

            c_array = np.array(c_results).reshape((self._n_times,self._n_components))
        
            results.Z = pd.DataFrame(data=c_array,
                                     columns=self._mixture_components,
                                     index=self._times)

            dc_results = []
            for t in self._times:
                for k in self._mixture_components:
                    dc_results.append(dZ_var[t,k].value)

            dc_array = np.array(dc_results).reshape((self._n_times,self._n_components))
        
            results.dZdt = pd.DataFrame(data=dc_array,
                                        columns=self._mixture_components,
                                        index=self._times)
                
            c_noise_results = []
            for t in self._meas_times:
                for k in self._mixture_components:
                    c_noise_results.append(self.model.C[t,k].value)

            s_results = []
            for l in self._meas_lambdas:
                for k in self._mixture_components:
                    s_results.append(self.model.S[l,k].value)

            c_noise_array = np.array(c_noise_results).reshape((self._n_meas_times,self._n_components))
            s_array = np.array(s_results).reshape((self._n_meas_lambdas,self._n_components))

            results.C = pd.DataFrame(data=c_noise_array,
                                           columns=self._mixture_components,
                                           index=self._meas_times)
                
            results.S = pd.DataFrame(data=s_array,
                                     columns=self._mixture_components,
                                     index=self._meas_lambdas)
            param_vals = dict()
            for name in self.model.parameter_names:
                param_vals[name] = self.model.P[name].value

            results.P = param_vals

            results.sigma_sq = dict()
            for name in self._mixture_components:
                results.sigma_sq[name] = self.model.sigma_sq[name].value
                results.device_variance = self.model.device_variance.value
        
            return results
        else:
            print "TODO"
            results = ResultsObject()
            
            
