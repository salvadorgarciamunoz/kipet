from pyomo.environ import *
from pyomo.dae import *
from ResultsObject import *
from Simulator import *


class PyomoSimulator(Simulator):
    def __init__(self,model):
        super(PyomoSimulator, self).__init__(model)
        self._times = sorted(self.model.time)
        self._n_times = len(self._times)
        self._spectra_given = hasattr(self.model, 'spectral_data')
        
        
    def apply_discretization(self,transformation,**kwargs):
        discretizer = TransformationFactory(transformation)
        discretizer.apply_to(self.model,wrt=self.model.time,**kwargs)
        self._times = sorted(self.model.time)
        self._n_times = len(self._times)
        self._discretized = True
        
    def initialize_from_trajectory(self,variable_name,trajectories):
        if self._discretized is False:
            raise RuntimeError('apply discretization first before runing simulation')
        
        if variable_name == 'C':
            var = self.model.C
            inner_set = self.model.time
        elif variable_name == 'C_noise':
            var = self.model.C_noise
            inner_set = self._meas_times
        elif variable_name == 'S':
            var = self.model.S
            inner_set = self._meas_lambdas
        else:
            raise RuntimeError('Initialization of variable {} is not supported'.format(variable_name))

        mixture_components = trajectories.columns
        
        for component in mixture_components:
            if component not in self._mixture_components:
                raise RuntimeError('Mixture component {} is not in model mixture components'.format(component))

        trajectory_times = np.array(trajectories.index)
        n_ttimes = len(trajectory_times)
        first_time = trajectory_times[0]
        last_time = trajectory_times[-1]
        for component in mixture_components:
            for t in inner_set:
                if t>=first_time and t<=last_time:
                    idx = find_nearest(trajectory_times,t)
                    t0 = trajectory_times[idx]
                    if t==t0:
                        var[t,component].value = trajectories[component][t0]
                    else:
                        if t0==last_time:
                            var[t,component].value = trajectories[component][t0]
                        else:
                            idx1 = idx+1
                            t1 = trajectory_times[idx1]
                            x_tuple = (t0,t1)
                            y_tuple = (trajectories[component][t0],trajectories[component][t1])
                            y = interpolate_linearly(t,x_tuple,y_tuple)
                            var[t,component].value = y
            
    def run_sim(self,solver,tee=False,solver_opts={}):

        if self._discretized is False:
            raise RuntimeError('apply discretization first before runing simulation')

        # Look at the output in results
        opt = SolverFactory(solver)

        for key, val in solver_opts.iteritems():
            opt.options[key]=val

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

        if self._spectra_given: 
            if self._n_meas_times and self._n_meas_times<self._n_components:
                raise RuntimeError('Not enough measurements num_meas>= num_components')

            # solves over determined system
            c_noise_array, s_array = self._solve_CS_from_D(results.C)

            d_results = []
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    d_results.append(self.model.spectral_data[t,l])
            d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
            
            results.C_noise = pd.DataFrame(data=c_noise_array,
                                           columns=self._mixture_components,
                                           index=self._meas_times)
            
            results.S = pd.DataFrame(data=s_array,
                                     columns=self._mixture_components,
                                     index=self._meas_lambdas)

            results.D = pd.DataFrame(data=d_array,
                                     columns=self._meas_lambdas,
                                     index=self._meas_times)

            for t in self.model.measurement_times:
                for k in self._mixture_components:
                    self.model.C_noise[t,k].value = results.C_noise[k][t]

            for l in self.model.measurement_lambdas:
                for k in self._mixture_components:
                    self.model.S[l,k].value =  results.S[k][l]
            
        else:
            c_noise_results = []
            for t in self._meas_times:
                for k in self._mixture_components:
                    c_noise_results.append(self.model.C[t,k].value)
                    
            s_results = []
            for l in self._meas_lambdas:
                for k in self._mixture_components:
                    s_results.append(self.model.S[l,k].value)

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
            
            results.C_noise = pd.DataFrame(data=c_noise_array,
                                           columns=self._mixture_components,
                                           index=self._meas_times)
            results.S = pd.DataFrame(data=s_array,
                                     columns=self._mixture_components,
                                     index=self._meas_lambdas)
            
                        
            d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
            results.D = pd.DataFrame(data=d_array,
                                     columns=self._meas_lambdas,
                                     index=self._meas_times)
            
        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name].value

        results.P = param_vals
        return results
        


