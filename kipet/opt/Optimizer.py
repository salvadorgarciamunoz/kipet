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
                expr += sum((m.C_noise[t,k]-m.C[t,k])**2/m.sigma[k]**2 for k in m.mixture_components)
            return expr
        self.model.direct_estimation = Objective(rule=rule_objective)
        
    def run_sim(self,solver,tee=False,solver_opts={}):
        raise NotImplementedError("Simulator abstract method. Call child class")

    def run_opt(self,solver,tee=False,solver_opts={},std_deviations={}):

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
            results = super(Optimizer,self).run_sim(solver,tee,solver_opts)
        else:
            print "TODO"
            results = ResultsObject()

        
        results.sigma = dict()
        for name in self._mixture_components:
            results.sigma[name] = self.model.sigma[name].value
        results.device_std_dev = self.model.device_std_dev.value
        
        return results
            
            
