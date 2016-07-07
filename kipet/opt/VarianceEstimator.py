from pyomo.environ import *
from pyomo.dae import *
from kipet.sim.PyomoSimulator import *
import copy
import os

class VarianceEstimator(PyomoSimulator):
    def __init__(self,model):
        super(VarianceEstimator, self).__init__(model)

        self.S_model = ConcreteModel()
        self.S_model.S = Var(self._meas_lambdas,
                             self._mixture_components,
                             bounds=(0.0,None),
                             initialize=1.0)

        self.C_model = ConcreteModel()
        self.C_model.C = Var(self._meas_times,
                           self._mixture_components,
                           bounds=(0.0,None),
                           initialize=1.0)

        # initializes the s and c models
        for k,v in self.model.S.iteritems():
            self.S_model.S[k].value = v.value
        for k,v in self.model.C.iteritems():
            self.C_model.C[k].value = v.value

        # To pass scaling to the submodels
        self.C_model.scaling_factor = Suffix(direction=Suffix.EXPORT)
        self.S_model.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # add suffixes for warm start
        
        # Ipopt bound multipliers (obtained from solution)
        self.model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # Ipopt bound multipliers (sent to solver)
        self.model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        # Obtain dual solutions from first solve and send to warm start
        self.model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)


        self.C_model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.C_model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # Ipopt bound multipliers (sent to solver)
        self.C_model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.C_model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        # Obtain dual solutions from first solve and send to warm start
        self.C_model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        self.S_model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.S_model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # Ipopt bound multipliers (sent to solver)
        self.S_model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.S_model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        # Obtain dual solutions from first solve and send to warm start
        self.S_model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        
        
        self._tmp1 = "tmp_init"
        self._tmp2 = "tmp_solve_Z"
        self._tmp3 = "tmp_solve_S"
        self._tmp4 = "tmp_solve_C"

        f = open(self._tmp2,"w")
        f.close()
        f = open(self._tmp3,"w")
        f.close()
        f = open(self._tmp4,"w")
        f.close()

        self._profile_time = True
        
    def __del__(self):
        if os.path.exists(self._tmp2):
            os.remove(self._tmp2)
        if os.path.exists(self._tmp3):
            os.remove(self._tmp3)
        if os.path.exists(self._tmp4):
            os.remove(self._tmp4)
        
    def run_sim(self,solver,**kwds):
        raise NotImplementedError("VarianceEstimator object does not have run_sim method. Call run_opt")

    def initialize_from_trajectory(self,variable_name,trajectories):
        super(VarianceEstimator, self).initialize_from_trajectory(variable_name,trajectories)
        if variable_name=='S':
            for k,v in self.model.S.iteritems():
                self.S_model.S[k].value = v.value
        if variable_name=='C':
            for k,v in self.model.C.iteritems():
                self.C_model.C[k].value = v.value        

    def scale_variables_from_trajectory(self,variable_name,trajectories):
        super(VarianceEstimator, self).scale_variables_from_trajectory(variable_name,trajectories)
        if variable_name=='S':
            for k,v in self.model.S.iteritems():
                value = self.model.scaling_factor.get(self.model.S[k])
                self.S_model.scaling_factor.set_value(v,value) 
        if variable_name=='C':
            for k,v in self.model.C.iteritems():
                value = self.model.scaling_factor.get(self.model.C[k])
                self.C_model.scaling_factor.set_value(v,value) 
        

    def _solve_initalization(self,
                             optimizer,
                             subset_lambdas=None):

        if subset_lambdas:
            set_A = set(subset_lambdas)
        else:
            set_A = self._meas_lambdas

        # build model
        dae = self.model
        m = ConcreteModel()
        m.A = Set(initialize=set_A)
        m.add_component('dae', dae)
        
        # build objective
        obj = 0.0
        for t in self._meas_times:
            for l in m.A:
                D_bar = sum(m.dae.Z[t,k]*m.dae.S[l,k] for k in m.dae.mixture_components)
                obj+= (m.dae.D[t,l] - D_bar)**2
        m.objective = Objective(expr=obj)
        print("Initialization Problem")
        solver_results = optimizer.solve(m, tee=True,report_timing=self._profile_time)

        for t in self._meas_times:
            for k in self._mixture_components:
                m.dae.C[t,k].value = m.dae.Z[t,k].value
                self.C_model.C[t,k].value =  m.dae.C[t,k].value
        m.del_component('dae')

    def _solve_Z(self,
                 optimizer,
                 C_trajectory=None):
        
        dae = self.model
        m = ConcreteModel()
        m.add_component('dae', dae)
        
        obj = 0.0
        reciprocal_ntp = 1.0/len(m.dae.time)
        if C_trajectory is not None:
            for k in m.dae.mixture_components:
                x = sum((C_trajectory[k][t]-m.dae.Z[t,k])**2 for t in self._meas_times)
                #x*= reciprocal_ntp
                obj+= x
        else:
            # asume this value was computed beforehand
            for t in self._meas_times:
                for k in self._mixture_components:
                    m.dae.C[t,k].fixed = True
            
            for k in m.dae.mixture_components:
                x = sum((m.dae.C[t,k]-m.dae.Z[t,k])**2 for t in self._meas_times)
                #x*= reciprocal_ntp
                obj+= x

        m.objective = Objective(expr=obj)
        if self._profile_time:
            print('-----------------Solve_Z--------------------')
        
        solver_results = optimizer.solve(m,logfile=self._tmp2,report_timing=self._profile_time)

        if C_trajectory is not None:
            # unfixes all concentrations
            for t in self._meas_times:
                for k in self._mixture_components:
                    m.dae.C[t,k].fixed = False
        
        m.del_component('dae')

    def _solve_S(self,
                 optimizer,
                 Z_trajectory=None):
        
        obj = 0.0
        if Z_trajectory is not None:
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    D_bar = sum(self.S_model.S[l,k]*Z_trajectory[k][t] for k in self._mixture_components)
                    obj+=(self.model.D[t,l]-D_bar)**2
        else:
            # asumes base model has been solved already for Z
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    D_bar = sum(self.S_model.S[l,k]*self.model.Z[t,k].value for k in self._mixture_components)
                    obj+=(self.model.D[t,l]-D_bar)**2
                    
        self.S_model.objective = Objective(expr=obj)

        if self._profile_time:
            print('-----------------Solve_S--------------------')
            
        solver_results = optimizer.solve(self.S_model,
                                         logfile=self._tmp3,
                                         report_timing=self._profile_time)
        
        self.S_model.del_component('objective')
        
        #updates values in main model
        for k,v in self.S_model.S.iteritems():
            self.model.S[k].value = v.value
        
    def _solve_C(self,
                 optimizer,
                 S_trajectory=None):

        
        obj = 0.0
        if S_trajectory is not None:
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    D_bar = sum(S_trajectory[k][l]*self.C_model.C[t,k] for k in self._mixture_components)
                    obj+=(self.model.D[t,l]-D_bar)**2
        else:
            # asumes that s model has been solved first
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    D_bar = sum(self.S_model.S[l,k].value*self.C_model.C[t,k] for k in self._mixture_components)
                    obj+=(self.model.D[t,l]-D_bar)**2
                    
        self.C_model.objective = Objective(expr=obj)

        if self._profile_time:
            print('-----------------Solve_C--------------------')
            
        solver_results = optimizer.solve(self.C_model,
                                         logfile=self._tmp4,
                                         report_timing=self._profile_time)
        
        self.C_model.del_component('objective')

        for t in self._meas_times:
            for k in self._mixture_components:
                self.model.C[t,k].value = self.C_model.C[t,k].value

    def _solve_variances(self,S_trajectory,Z_trajectory):
        nl = self._n_meas_lambdas
        nt = self._n_meas_times
        nc = self._n_components
        A = np.ones((nl,nc+1))
        b = np.zeros((nl,1))

        reciprocal_nt = 1.0/nt
        for i,l in enumerate(self._meas_lambdas):
            for j,t in enumerate(self._meas_times):
                D_bar = 0.0
                for w,k in enumerate(self._mixture_components):
                    A[i,w] = S_trajectory[k][l]**2
                    D_bar += S_trajectory[k][l]*Z_trajectory[k][t]
                b[i] += (self.model.D[t,l]-D_bar)**2
            b[i]*=reciprocal_nt

        results = np.linalg.lstsq(A, b)
        return results            

    def _log_iterations(self,filename,iteration):
        f = open(filename, "a")

        f.write("\n#######################Iteration {}#######################\n".format(iteration))
        tf = open(self._tmp2,'r')
        f.write("\n#######################Solve Z#######################\n")
        f.write(tf.read())
        tf.close()
        tf = open(self._tmp3,'r')
        f.write("\n#######################Solve S#######################\n")
        f.write(tf.read())
        tf.close()
        tf = open(self._tmp4,'r')
        f.write("\n#######################Solve C#######################\n")
        f.write(tf.read())
        tf.close()
        
        f.close()
    
    def run_opt(self,solver,**kwds):

        solver_opts = kwds.pop('solver_opts', dict())
        variances = kwds.pop('variances',dict())
        tee = kwds.pop('tee',False)
        norm_order = kwds.pop('norm',np.inf)
        max_iter = kwds.pop('max_iter',400)
        tol = kwds.pop('tolerance',1e-5)
        A = kwds.pop('subset_lambdas',None)
        
        if not self.model.time.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')

        # disable objectives of dae
        objectives_map = self.model.component_map(ctype=Objective,active=True)
        for obj in objectives_map.itervalues():
            obj.deactivate()

         
        opt = SolverFactory(solver)
        for key, val in solver_opts.iteritems():
            opt.options[key]=val
        
        # solves formulation 18
        self._solve_initalization(opt,subset_lambdas=A)
        Z_i = np.array([v.value for v in self.model.Z.itervalues()])

        print("Solving Variance estimation")
        # perfoms the fisrt iteration 
        self._solve_Z(opt)
        self._solve_S(opt)
        self._solve_C(opt)

        if tee:
            self._log_iterations("iterations.log",0)
        
        Z_i1 = np.array([v.value for v in self.model.Z.itervalues()])

        diff = Z_i1-Z_i
        norm_diff = np.linalg.norm(diff,norm_order)

        print("{: >11} {: >16}".format('Iter','|Zi-Zi+1|'))
        #print("{: >10} {: >20}".format(0,norm_diff))
        count=1
        norm_diff=1
        while norm_diff>tol and count<max_iter:
            
            if count>2:
                self.model.ipopt_zL_in.update(self.model.ipopt_zL_out)
                self.model.ipopt_zU_in.update(self.model.ipopt_zU_out)
                self.C_model.ipopt_zL_in.update(self.C_model.ipopt_zL_out)
                self.C_model.ipopt_zU_in.update(self.C_model.ipopt_zU_out)
                self.S_model.ipopt_zL_in.update(self.S_model.ipopt_zL_out)
                self.S_model.ipopt_zU_in.update(self.S_model.ipopt_zU_out)
                opt.options['warm_start_init_point'] = 'yes'
                opt.options['warm_start_bound_push'] = 1e-6
                opt.options['warm_start_mult_bound_push'] = 1e-6
                opt.options['mu_init'] = 1e-6
            Z_i = Z_i1
            self._solve_Z(opt)
            self._solve_S(opt)
            self._solve_C(opt)
            Z_i1 = np.array([v.value for v in self.model.Z.itervalues()])
            norm_diff = np.linalg.norm(Z_i1-Z_i,norm_order)

            print("{: >10} {: >20}".format(count,norm_diff))
            if tee:
                self._log_iterations("iterations.log",count)
            count+=1

        results =  ResultsObject()
        results.load_from_pyomo_model(self.model,
                                      to_load=['Z','dZdt','X','dXdt','C','S'])

        res_lsq = self._solve_variances(results.S,results.Z)
        variance_dict = dict()
        for i,k in enumerate(self._mixture_components):
            variance_dict[k] = abs(res_lsq[0][i][0])
            
        variance_dict['device'] = abs(res_lsq[0][-1][0])
        results.sigma_sq = variance_dict 
        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name].value

        results.P = param_vals
        
        return results

            
            
