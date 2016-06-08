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
        mixture_component_names = []
        for i,k in enumerate(self.model.mixture_component_names):
            states_l.append(self.model.C[k])
            ode_l.append(self.model.diff_exprs[k])
            init_conditions_l.append(self.model.init_conditions[k])
            mixture_component_names.append(k)

        states = ca.vertcat(*states_l)
        ode = ca.vertcat(*ode_l)
        x_0 = ca.vertcat(*init_conditions_l)
    
        system = {'x':states, 'ode':ode}
        step = (self.model.end_time - self.model.start_time)/self.nfe
        print 'step',step
        opts = {'tf':step}

        I = integrator("I", "cvodes", system, opts)

        results = ResultsObject()
        results.time = [self.model.start_time]
        tmp_containers = {}
        tmp_containers['component_concentration'] = [j for j in init_conditions_l]
        
        print tmp_containers
        xk = x_0
        for i in xrange(1,self.nfe+1):
            xk = I(x0=xk)['xf']
            for j in xrange(xk.numel()):
                tmp_containers['component_concentration'].append(xk[j])
            results.time.append(i*step)
        tmp_dict = dict()
        tmp_dict['concentration'] = tmp_containers['component_concentration']
        
        n_times = len(results.time)
        n_components = len(mixture_component_names)
        for key,val in tmp_dict.iteritems():
            tmp_dict[key] = np.array(val).reshape((n_times,n_components))
        
        results.panel =  pd.Panel(tmp_dict, major_axis=results.time, minor_axis=mixture_component_names)
            
        return results
        
        
        
