import casadi as ca
from casadi.tools import *
from ResultsObject import *
from Simulator import *

class CasadiSimulator(Simulator):
    def __init__(self,model):
        super(CasadiSimulator, self).__init__(model)
        self.nfe = None
        
    def apply_discretization(self,transformation,**kwargs):
        if kwargs.has_key('nfe'):
            self.nfe = kwargs['nfe']
        else:
            raise RuntimeError('Specify discretization points nfe=int8')
        
    def initialize_from_trajectory(self,trajectory_dictionary):
        pass

    def run_sim(self,solver,tee=False):
        
        if self.nfe is None:
            raise RuntimeError('apply discretization first before runing simulation')
        
        states_l = []
        ode_l = []
        init_conditions_l = []
        map_back = dict()
        mixture_components = []
        for i,k in enumerate(self.model.mixture_components):
            states_l.append(self.model.C[k])
            ode_l.append(self.model.diff_exprs[k])
            init_conditions_l.append(self.model.init_conditions[k])
            mixture_components.append(k)

        states = ca.vertcat(*states_l)
        ode = ca.vertcat(*ode_l)
        x_0 = ca.vertcat(*init_conditions_l)
    
        system = {'x':states, 'ode':ode}
        step = (self.model.end_time - self.model.start_time)/self.nfe
        opts = {'tf':step}

        I = integrator("I", "cvodes", system, opts)

        results = ResultsObject()
        times = [self.model.start_time]

        c_results =  [j for j in init_conditions_l]
        xk = x_0
        for i in xrange(1,self.nfe+1):
            xk = I(x0=xk)['xf']
            for j in xrange(xk.numel()):
                c_results.append(xk[j])
            times.append(i*step)
        
        n_times = len(times)
        n_components = len(mixture_components)
        c_array = np.array(c_results).reshape((n_times,n_components))
        results.C = pd.DataFrame(data=c_array,columns=mixture_components,index=times)
                    
        return results
        
        
        
