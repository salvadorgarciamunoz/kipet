from pyomo.environ import *
from pyomo.dae import *
#from kipet.ResultsObject import *
from kipet.sim.PyomoSimulator import *


class Optimizer(PyomoSimulator):
    def __init__(self,model):
        super(Optimizer, self).__init__(model)
        # add variance variables
        self.model.device_std_dev = Var(within=NonNegativeReals, initialize=1)
        self.model.sigma = Var(self.model.mixture_components,
                               within=NonNegativeReals,
                               initialize=1)
        # estimation
        def rule_objective(m):
            expr = 0
            for t in m.measurement_times:
                for l in m.measurement_lambdas:
                    current = m.spectral_data[t,l] - sum(m.C_noise[t,k]*m.S[l,k] for k in m.mixture_components)
                    expr+= current**2/m.device_std_dev**2
            for t in m.measurement_times:
                expr += sum((m.C_noise[t,k]-m.C[t,k])**2/m.sigma[k]**2 for k in m.mixture_components)
            return expr
        self.model.direct_estimation = Objective(rule=rule_objective)
        
    def run_sim(self,solver,tee=False,solver_opts={}):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def run_opt(self,solver,tee=False,solver_opts={},std_deviations={}):

        if self._discretized is False:
            raise RuntimeError('apply discretization first before runing simulation')
        
        # fixes the sigmas
        for k,v in std_deviations.iteritems():
            if k=='device':
                self.model.device_std_dev.value = v
                self.model.device_std_dev.fixed = True
            else:
                self.model.sigma[k] = v
                self.model.sigma[k].fixed = True

        #check if all sigmas are fixed
        all_sigmas_fixed = True
        if self.model.device_std_dev.fixed == False:
            all_sigmas_fixed = False
        for k in self._mixture_components:
            if self.model.sigma[k].fixed == False:
                all_sigmas_fixed = False
                break
            
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
                    c_results.append(self.model.C[t,k].value)

            c_array = np.array(c_results).reshape((self._n_times,self._n_components))
        
            results.C = pd.DataFrame(data=c_array,
                                     columns=self._mixture_components,
                                     index=self._times)

            dc_results = []
            for t in self._times:
                for k in self._mixture_components:
                    dc_results.append(self.model.dCdt[t,k].value)

            dc_array = np.array(dc_results).reshape((self._n_times,self._n_components))
        
            results.dCdt = pd.DataFrame(data=dc_array,
                                        columns=self._mixture_components,
                                        index=self._times)
                
            c_noise_results = []
            for t in self._meas_times:
                for k in self._mixture_components:
                    c_noise_results.append(self.model.C_noise[t,k].value)

            s_results = []
            for l in self._meas_lambdas:
                for k in self._mixture_components:
                    s_results.append(self.model.S[l,k].value)

            c_noise_array = np.array(c_noise_results).reshape((self._n_meas_times,self._n_components))
            s_array = np.array(s_results).reshape((self._n_meas_lambdas,self._n_components))

            results.C_noise = pd.DataFrame(data=c_noise_array,
                                           columns=self._mixture_components,
                                           index=self._meas_times)
                
            results.S = pd.DataFrame(data=s_array,
                                     columns=self._mixture_components,
                                     index=self._meas_lambdas)
            param_vals = dict()
            for name in self.model.parameter_names:
                param_vals[name] = self.model.P[name].value

            results.P = param_vals

            results.sigma = dict()
            for name in self._mixture_components:
                results.sigma[name] = self.model.sigma[name].value
                results.device_std_dev = self.model.device_std_dev.value
        
            return results
        else:
            print "TODO"
            results = ResultsObject()
            
            
