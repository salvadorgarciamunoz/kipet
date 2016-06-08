from pyomo.environ import *
from pyomo.dae import *
from ResultsObject import *
from Simulator import *

class PyomoSimulator(Simulator):
    def __init__(self,model):
        super(PyomoSimulator, self).__init__(model)

    def apply_discretization(self,transformation,**kwargs):
        discretizer = TransformationFactory(transformation)
        discretizer.apply_to(self.model,wrt=self.model.time,**kwargs)
    
    def initialize_from_trajectory(self,variable_name,trajectories):
        if variable_name == 'C':
            mixture_components = trajectories.columns

            for component in mixture_components:
                if component not in self.model.mixture_components:
                    raise RuntimeError('Mixture component {} is not in model mixture components'.format(component))

            trajectory_times = np.array(trajectories.index)
            n_ttimes = len(trajectory_times)
            first_time = trajectory_times[0]
            last_time = trajectory_times[-1]
            for component in mixture_components:
                for t in self.model.time:
                    if t>=first_time and t<=last_time:
                        idx = find_nearest(trajectory_times,t)
                        t0 = trajectory_times[idx]
                        if t==t0:
                            type(trajectories[component][t])
                            self.model.C[t,component].value = trajectories[component][t0]
                        else:
                            idx1 = idx+1
                            t1 = trajectory_times[idx1]
                            x_tuple = (t0,t1)
                            y_tuple = (trajectories[component][t0],trajectories[component][t1])
                            y = interpolate_linearly(t,x_tuple,y_tuple)
                            #print component,t,y,x_tuple,y_tuple
                            self.model.C[t,component].value = y
                
    def run_sim(self,solver,tee=False):
        # Look at the output in results
        opt = SolverFactory(solver)
        solver_results = opt.solve(self.model,tee=tee)
        results = ResultsObject()
        
        times = sorted(self.model.time)
        mixture_components = [name for name in self.model.mixture_components]
        measured_times = [t for t in self.model.measurement_times]
        measured_lambdas = [l for l in self.model.measurement_lambdas] 
        n_times = len(times)
        n_measured_times = len(measured_times)
        n_measured_lambdas = len(measured_lambdas)
        n_components = len(mixture_components)
        
        c_results = []
        for t in times:
            for k in mixture_components:
                c_results.append(self.model.C[t,k].value)

        c_noise_results = []
        for t in measured_times:
            for k in mixture_components:
                c_noise_results.append(self.model.C_noise[t,k].value)

        s_results = []
        for l in measured_lambdas:
            for k in mixture_components:
                s_results.append(self.model.S[t,k].value)

        c_array = np.array(c_results).reshape((n_times,n_components))
        results.C = pd.DataFrame(data=c_array,columns=mixture_components,index=times)
        
        c_noise_array = np.array(c_noise_results).reshape((n_measured_times,n_components))
        results.C_noise = pd.DataFrame(data=c_noise_array,columns=mixture_components,index=measured_times)

        s_array = np.array(s_results).reshape((n_measured_lambdas,n_components))
        results.S = pd.DataFrame(data=s_array,columns=mixture_components,index=measured_lambdas)

        return results
        
