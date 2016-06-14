import casadi as ca
from casadi.tools import *
from ResultsObject import *
from Simulator import *
import copy


class CasadiSimulator(Simulator):
    def __init__(self,model):
        super(CasadiSimulator, self).__init__(model)
        self.nfe = None
        self._times = set([t for t in model.measurement_times])
        self._n_times = len(self._times)
        self._spectra_given = hasattr(self.model, 'spectral_data')
        
    def apply_discretization(self,transformation,**kwargs):
        
        if kwargs.has_key('nfe'):
            self.nfe = kwargs['nfe']
            self.model.start_time
            step = (self.model.end_time - self.model.start_time)/self.nfe
            for i in xrange(0,self.nfe+1):
                self._times.add(i*step)
                
            self._n_times = len(self._times)
            self._discretized = True
        else:
            raise RuntimeError('Specify discretization points nfe=int8')
        
        
    def initialize_from_trajectory(self,trajectory_dictionary):
        pass

    def run_sim(self,solver,tee=False,solver_opts={}):
        
        if self._discretized is False:
            raise RuntimeError('apply discretization first before runing simulation')
        states_l = []
        ode_l = []
        init_conditions_l = []
        map_back = dict()
        for i,k in enumerate(self._mixture_components):
            states_l.append(self.model.C[k])
            ode_l.append(self.model.diff_exprs[k])
            init_conditions_l.append(self.model.init_conditions[k])
            

        states = ca.vertcat(*states_l)
        ode = ca.vertcat(*ode_l)
        x_0 = ca.vertcat(*init_conditions_l)
    
        system = {'x':states, 'ode':ode}
        
        step = (self.model.end_time - self.model.start_time)/self.nfe

        results = ResultsObject()

        c_results =  []
        xk = x_0
        times = sorted(self._times)
        for i,t in enumerate(times):
            if t == self.model.start_time:
                for j in init_conditions_l:
                    c_results.append(j)
            else:
                step = t - times[i-1]
                opts = {'tf':step}
                I = integrator("I",solver, system, opts)
                xk = I(x0=xk)['xf']
                for j in xrange(xk.numel()):
                    c_results.append(xk[j])
        
        c_array = np.array(c_results).reshape((self._n_times,self._n_components))
        results.C = pd.DataFrame(data=c_array,columns=self._mixture_components,index=times)
        

        if self._spectra_given:
            # solves over determined system
            c_noise_array, s_array = self._solve_CS_from_D(results.C)

            d_results = []
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    d_results.append(self.model.spectral_data[t,l])
                    
            d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
        else:
            
            c_noise_results = []
            for t in self._meas_times:
                for k in self._mixture_components:
                    c_noise_results.append(results.C[k][t])

            s_results = []
            for l in self._meas_lambdas:
                for k in self._mixture_components:
                    s_results.append(self.model.S[l,k])

            d_results = []
            if s_results and c_noise_results:
                for i,t in enumerate(self._meas_times):
                    for j,l in enumerate(self._meas_lambdas):
                        suma = 0.0
                        for w,k in enumerate(self._mixture_components):
                            C = c_noise_results[i*self._n_components+w]
                            S = s_results[j*self._n_components+w]
                            suma+= C*S
                        d_results.append(suma)

            c_noise_array = np.array(c_noise_results).reshape((self._n_meas_times,self._n_components))
            s_array = np.array(s_results).reshape((self._n_meas_lambdas,self._n_components))
            d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
            
        # stores everything in restuls object
        results.C_noise = pd.DataFrame(data=c_noise_array,
                                       columns=self._mixture_components,
                                       index=self._meas_times)
        results.S = pd.DataFrame(data=s_array,
                                 columns=self._mixture_components,
                                 index=self._meas_lambdas)
        results.D = pd.DataFrame(data=d_array,
                                 columns=self._meas_lambdas,
                                 index=self._meas_times)
        
        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name]

        results.P = param_vals
        
        return results
        
        
        
