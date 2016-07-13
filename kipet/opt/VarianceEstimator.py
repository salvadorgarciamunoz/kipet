from pyomo.environ import *
from pyomo.dae import *
from kipet.opt.Optimizer import *
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
import time
import copy
import sys
import os
import re

import fileinput

class VarianceEstimator(Optimizer):
    def __init__(self,model):
        super(VarianceEstimator, self).__init__(model)

        # add suffixes for warm start
        # Ipopt bound multipliers (obtained from solution)
        self.model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # Ipopt bound multipliers (sent to solver)
        self.model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        # Obtain dual solutions from first solve and send to warm start
        self.model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        # build additional models
        self._build_s_model()
        self._build_c_model()

        # used for scipy leas squares
        self._d_array = np.zeros((self._n_meas_times,self._n_meas_lambdas))
        for i,t in enumerate(self._meas_times):
            for j,l in enumerate(self._meas_lambdas):
                self._d_array[i,j] = self.model.D[t,l]

        self._s_array = np.ones(self._n_meas_lambdas*self._n_components)
        self._z_array = np.ones(self._n_meas_times*self._n_components)
        self._c_array = np.ones(self._n_meas_times*self._n_components)

        #tmp
        self._build_s_flat_model()
        self._build_c_flat_model()
        
        self._tmp2 = "tmp_solve_Z"
        self._tmp3 = "tmp_solve_S"
        self._tmp4 = "tmp_solve_C"

        f =open(self._tmp2,"w")
        f.close()
        f =open(self._tmp3,"w")
        f.close()
        f =open(self._tmp4,"w")
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

    def _build_s_flat_model(self):
        # smodel
        self.flat_smodel = ConcreteModel()
        self.flat_smodel.s_indices = Set(initialize=range(self._n_meas_lambdas*self._n_components))
        self.flat_smodel.z_indices = Set(initialize=range(self._n_meas_times*self._n_components))
        self.flat_smodel.s = Var(self.flat_smodel.s_indices,
                                 bounds=(0.0,None),
                                 initialize=1.0)

        self.flat_smodel.nc = self._n_components
        self.flat_smodel.nt = self._n_meas_times
        self.flat_smodel.nl = self._n_meas_lambdas
        #self.flat_smodel.z = Param(self.flat_smodel.z_indices,mutable=True,initialize=1.1)
        self.flat_smodel.z = Var(self.flat_smodel.z_indices,initialize=1.0)

        # D_array
        self.flat_smodel.d_array = d_array = np.zeros((self._n_meas_times,self._n_meas_lambdas))
        for i,t in enumerate(self._meas_times):
            for j,l in enumerate(self._meas_lambdas):
                self.flat_smodel.d_array[i,j] = self.model.D[t,l]

        
        def rule_s_objective(m):
            obj = 0.0
            for i in xrange(m.nt):
                for j in xrange(m.nl):
                    d_bar = sum(m.s[j*m.nc+k]*m.z[i*m.nc+k]for k in xrange(m.nc))
                    obj+=(d_bar-m.d_array[i,j])**2
            return obj

        self.flat_smodel.objective = Objective(rule=rule_s_objective)

    def _build_c_flat_model(self):
        # cmodel
        self.flat_cmodel = ConcreteModel()
        
        self.flat_cmodel.s_indices = Set(initialize=range(self._n_meas_lambdas*self._n_components))
        self.flat_cmodel.c_indices = Set(initialize=range(self._n_meas_times*self._n_components))
        self.flat_cmodel.c = Var(self.flat_cmodel.c_indices,
                                 bounds=(0.0,None),
                                 initialize=1.0)

        self.flat_cmodel.nc = self._n_components
        self.flat_cmodel.nt = self._n_meas_times
        self.flat_cmodel.nl = self._n_meas_lambdas
        #self.flat_cmodel.s = Param(self.flat_cmodel.s_indices,mutable=True,initialize=1.0)
        self.flat_cmodel.s = Var(self.flat_cmodel.s_indices,initialize=1.0)

        # D_array
        self.flat_cmodel.d_array = d_array = np.zeros((self._n_meas_times,self._n_meas_lambdas))
        for i,t in enumerate(self._meas_times):
            for j,l in enumerate(self._meas_lambdas):
                self.flat_cmodel.d_array[i,j] = self.model.D[t,l]

        def rule_c_objective(m):
            obj = 0.0
            for i in xrange(m.nt):
                for j in xrange(m.nl):
                    d_bar = sum(m.s[j*m.nc+k]*m.c[i*m.nc+k]for k in xrange(m.nc))
                    obj+=(m.d_array[i,j]-d_bar)**2
            return obj

        self.flat_cmodel.objective = Objective(rule=rule_c_objective)
        
        
    def _build_s_model(self):
        self.S_model = ConcreteModel()
        self.S_model.S = Var(self._meas_lambdas,
                             self._mixture_components,
                             bounds=(0.0,None),
                             initialize=1.0)
        
        # add suffixes for warm start
        self.S_model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.S_model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # Ipopt bound multipliers (sent to solver)
        self.S_model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.S_model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        # Obtain dual solutions from first solve and send to warm start
        self.S_model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        # for scaling factors
        self.S_model.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # initialization
        for k,v in self.model.S.iteritems():
            self.S_model.S[k].value = v.value

    def _build_c_model(self):
        
        self.C_model = ConcreteModel()
        self.C_model.C = Var(self._meas_times,
                           self._mixture_components,
                           bounds=(0.0,None),
                           initialize=1.0)

        # add suffixes for warm start
        self.C_model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.C_model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # Ipopt bound multipliers (sent to solver)
        self.C_model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.C_model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        # Obtain dual solutions from first solve and send to warm start
        self.C_model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

        # initializes the s and c models
        for k,v in self.model.C.iteritems():
            self.C_model.C[k].value = v.value

        # To pass scaling to the submodels
        self.C_model.scaling_factor = Suffix(direction=Suffix.EXPORT)

            
    def _solve_initalization(self,
                             optimizer,
                             subset_lambdas=None):

        if subset_lambdas:
            set_A = set(subset_lambdas)
        else:
            set_A = self._meas_lambdas
        
        # build objective
        obj = 0.0
        for t in self._meas_times:
            for l in set_A:
                D_bar = sum(self.model.Z[t,k]*self.model.S[l,k] for k in self.model.mixture_components)
                obj+= (self.model.D[t,l] - D_bar)**2
        self.model.init_objective = Objective(expr=obj)

        print("Solving Initialization Problem\n")

        solver_results = optimizer.solve(self.model, tee=True,report_timing=self._profile_time)

        for t in self._meas_times:
            for k in self._mixture_components:
                self.model.C[t,k].value = np.random.normal(self.model.Z[t,k].value,1e-6)
                self.C_model.C[t,k].value =  self.model.C[t,k].value
        self.model.del_component('init_objective')

    def _solve_Z(self,
                 optimizer,
                 C_trajectory=None):
        
        obj = 0.0
        reciprocal_ntp = 1.0/len(self.model.time)
        if C_trajectory is not None:
            for k in self.model.mixture_components:
                x = sum((C_trajectory[k][t]-self.model.Z[t,k])**2 for t in self._meas_times)
                #x*= reciprocal_ntp
                obj+= x
        else:
            # asume this value was computed beforehand
            for t in self._meas_times:
                for k in self._mixture_components:
                    self.model.C[t,k].fixed = True
            
            for k in self.model.mixture_components:
                x = sum((self.model.C[t,k]-self.model.Z[t,k])**2 for t in self._meas_times)
                #x*= reciprocal_ntp
                obj+= x

        self.model.z_objective = Objective(expr=obj)
        #self.model.z_objective.pprint()
        if self._profile_time:
            print('-----------------Solve_Z--------------------')
        
        solver_results = optimizer.solve(self.model,
                                         logfile=self._tmp2,
                                         #show_section_timing=True,
                                         report_timing=self._profile_time)
                                         

        if C_trajectory is not None:
            # unfixes all concentrations
            for t in self._meas_times:
                for k in self._mixture_components:
                    self.model.C[t,k].fixed = False
        
        self.model.del_component('z_objective')
        
    def _solve_flat_s(self,optimizer):
        n = self._n_components
        for j,l in enumerate(self._meas_lambdas):
            for k,c in enumerate(self._mixture_components):
                self.flat_smodel.s[j*n+k].value = self.model.S[l,c].value
                
        for j,t in enumerate(self._meas_times):
            for k,c in enumerate(self._mixture_components):
                self.flat_smodel.z[j*n+k].setub(self.model.Z[t,c].value)
                self.flat_smodel.z[j*n+k].setlb(self.model.Z[t,c].value)
                
        if self._profile_time:
            print('-----------------Solve_S--------------------')
            
        solver_results = optimizer.solve(self.flat_smodel,
                                         logfile=self._tmp3,
                                         #show_section_timing=True,
                                         report_timing=self._profile_time)
        
        # retrive solution to model
        for j,l in enumerate(self._meas_lambdas):
            for k,c in enumerate(self._mixture_components):
                self.model.S[l,c].value = self.flat_smodel.s[j*n+k].value

    def _solve_S(self,
                 optimizer,
                 Z_trajectory=None):
                
        obj = 0.0
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
                                         #show_section_timing=True,
                                         report_timing=self._profile_time)
                                         
        
        self.S_model.del_component('objective')
        
        #updates values in main model
        for k,v in self.S_model.S.iteritems():
            self.model.S[k].value = v.value
                
    def _solve_s_scipy(self):

        if self._profile_time:
            print('-----------------Solve_S--------------------')
            t0 = time.time()
        # assumes S have been computed in the model
        n = self._n_components
        for j,l in enumerate(self._meas_lambdas):
            for k,c in enumerate(self._mixture_components):
                if self.model.S[l,c].value<=0.0:
                    self._s_array[j*n+k] = 1e-15
                else:
                    self._s_array[j*n+k] = self.model.S[l,c].value

        for j,t in enumerate(self._meas_times):
            for k,c in enumerate(self._mixture_components):
                self._z_array[j*n+k] = self.model.Z[t,c].value

        def F(x,z_array,d_array,nl,nt,nc):
            diff = np.zeros(nt*nl)
            for i in xrange(nt):
                for j in xrange(nl):
                    diff[i*nl+j]=d_array[i,j]-sum(z_array[i*nc+k]*x[j*nc+k] for k in xrange(nc))
            return diff

        def JF(x,z_array,d_array,nl,nt,nc):
            row = []
            col = []
            data = []
            for i in xrange(nt):
                for j in xrange(nl):
                    for k in xrange(nc):
                        row.append(i*nl+j)
                        col.append(j*nc+k)
                        data.append(-z_array[i*nc+k])
            return coo_matrix((data, (row, col)),
                              shape=(nt*nl,nc*nl))

        # solve
        res = least_squares(F,
                            self._s_array,
                            JF,
                            #max_nfev=7,
                            #jac_sparsity=sparsity,
                            bounds=(0.0,np.inf),
                            #verbose=2,
                            args=(self._z_array,
                                  self._d_array,
                                  self._n_meas_lambdas,
                                  self._n_meas_times,
                                  self._n_components))
        if self._profile_time:
            t1 = time.time()
            print("Scipy.optimize.least_squares time={:.3f} seconds".format(t1-t0))


        # retrive solution
        for j,l in enumerate(self._meas_lambdas):
            for k,c in enumerate(self._mixture_components):
                self.model.S[l,c].value = res.x[j*n+k]
                

    def _solve_flat_c(self,optimizer):
        n = self._n_components
        for i,t in enumerate(self._meas_times):
            for k,c in enumerate(self._mixture_components):
                self.flat_cmodel.c[i*n+k] = self.model.C[t,c].value

        for j,l in enumerate(self._meas_lambdas):
            for k,c in enumerate(self._mixture_components):
                self.flat_cmodel.s[j*n+k].setlb(self.model.S[l,c].value)
                self.flat_cmodel.s[j*n+k].setub(self.model.S[l,c].value)
                
        if self._profile_time:
            print('-----------------Solve_C--------------------')
            
        solver_results = optimizer.solve(self.flat_cmodel,
                                         logfile=self._tmp4,
                                         report_timing=self._profile_time)
                
        # retrive solution
        for j,t in enumerate(self._meas_times):
            for k,c in enumerate(self._mixture_components):
                self.model.C[t,c].value = self.flat_cmodel.c[j*n+k].value

    def _solve_C(self,
                 optimizer):
        obj=0.0
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

                
    def _solve_c_scipy(self):

        if self._profile_time:
            print('-----------------Solve_C--------------------')
            t0 = time.time()
        # assumes S have been computed in the model
        n = self._n_components
        for i,t in enumerate(self._meas_times):
            for k,c in enumerate(self._mixture_components):
                if self.model.C[t,c].value <=0.0:
                    self._c_array[i*n+k] = 1e-15
                else:
                    self._c_array[i*n+k] = self.model.C[t,c].value

        for j,l in enumerate(self._meas_lambdas):
            for k,c in enumerate(self._mixture_components):
                self._s_array[j*n+k] = self.model.S[l,c].value

        def F(x,s_array,d_array,nl,nt,nc):
            diff = np.zeros(nt*nl)
            for i in xrange(nt):
                for j in xrange(nl):
                    diff[i*nl+j]=d_array[i,j]-sum(s_array[j*nc+k]*x[i*nc+k] for k in xrange(nc))
            return diff

        def JF(x,s_array,d_array,nl,nt,nc):
            row = []
            col = []
            data = []
            for i in xrange(nt):
                for j in xrange(nl):
                    for k in xrange(nc):
                        row.append(i*nl+j)
                        col.append(j*nc+k)
                        data.append(-s_array[j*nc+k])
            return coo_matrix((data, (row, col)),
                              shape=(nt*nl,nc*nt))

        # solve
        res = least_squares(F,
                            self._c_array,
                            JF,
                            #jac_sparsity=sparsity,
                            bounds=(0.0,np.inf),
                            #max_nfev=7,
                            #verbose=2,
                            args=(self._s_array,
                                  self._d_array,
                                  self._n_meas_lambdas,
                                  self._n_meas_times,
                                  self._n_components))
        if self._profile_time:
            t1 = time.time()
            print("Scipy.optimize.least_squares time={:.3f} seconds".format(t1-t0))

        # retrive solution
        for j,t in enumerate(self._meas_times):
            for k,c in enumerate(self._mixture_components):
                self.model.C[t,c].value = res.x[j*n+k]

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

        # deactivates objective functions                
        objectives_map = self.model.component_map(ctype=Objective,active=True)
        active_objectives_names = []
        for obj in objectives_map.itervalues():
            name = obj.cname()
            active_objectives_names.append(name)
            obj.deactivate()
            
        opt = SolverFactory(solver)
        for key, val in solver_opts.iteritems():
            opt.options[key]=val

        
        # solves formulation 18
        self._solve_initalization(opt,subset_lambdas=A)
        Z_i = np.array([v.value for v in self.model.Z.itervalues()])

        print("Solving Variance estimation")
        #perfoms the fisrt iteration 
        self._solve_Z(opt)
        #self._solve_flat_s(opt)
        #self._solve_flat_c(opt)
        self._solve_s_scipy()
        self._solve_c_scipy()
        #self._solve_S(opt)
        #self._solve_C(opt))
        
        
        """
        count=103189060359
        first_count=count
        dict_z = dict()
        n_z = len(self.flat_smodel.s)
        for v in self.flat_smodel.z.itervalues():
            v.setlb(count)
            v.setub(count)
            dict_z["{}".format(count)]=1.0
            count+=1
            
        self.flat_smodel.write("junk.nl")
        
        def multiple_replace(text, adict):
            rx = re.compile('|'.join(adict.iterkeys()))
            def one_xlat(match):
                return adict[match.group(0)]
            return rx.sub(one_xlat, text)
        
        t0 = time.time()
        f = file('junk.nl', 'r')
        all_string = f.read()
        print n_z
        start = all_string.find('{}'.format(first_count))
        subject = all_string[start:]
        #print subject
        f.close()
                  
        counter=0
        for k,v in dict_z.iteritems():
            dict_z[k]="np{}".format(counter)
            counter+=1
        
        subject2 = multiple_replace(subject,dict_z)
        f_out = file('second.nl', 'w')
        f_out.write(all_string[:start]+subject2)
        f_out.close()
        
        t1 = time.time()
        print t1-t0
        sys.exit()
        """
        if tee:
            filename = "iterations.log"
            if os.path.isfile(filename):
                os.remove(filename)
            self._log_iterations(filename,0)
            
        Z_i1 = np.array([v.value for v in self.model.Z.itervalues()])

        diff = Z_i1-Z_i
        norm_diff = np.linalg.norm(diff,norm_order)

        print("{: >11} {: >16}".format('Iter','|Zi-Zi+1|'))
        print("{: >10} {: >20}".format(0,norm_diff))
        count=1
        norm_diff=1
        while norm_diff>tol and count<max_iter:
            
            if count>2:
                self.model.ipopt_zL_in.update(self.model.ipopt_zL_out)
                self.model.ipopt_zU_in.update(self.model.ipopt_zU_out)
                opt.options['warm_start_init_point'] = 'yes'
                opt.options['warm_start_bound_push'] = 1e-6
                opt.options['warm_start_mult_bound_push'] = 1e-6
                opt.options['mu_init'] = 1e-6
            Z_i = Z_i1
            self._solve_Z(opt)
            #self._solve_S(opt)
            #self._solve_C(opt)
            self._solve_s_scipy()
            self._solve_c_scipy()
            #self._solve_flat_s(opt)
            #self._solve_flat_c(opt)
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

        # activates objective functions that were deactivated
        active_objectives_names = []
        objectives_map = self.model.component_map(ctype=Objective)
        for name in active_objectives_names:
            objectives_map[name].activate()
        
        return results

            
            
