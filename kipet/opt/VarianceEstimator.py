from pyomo.environ import *
from pyomo.dae import *
from pyomo.core import *
from pyomo.opt import (ReaderFactory,
                       ResultsFormat)
from kipet.opt.Optimizer import *
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
#import pyutilib.subprocess
import subprocess
import time
import copy
import sys
import os
import re


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
        
        self._tmp2 = "tmp_solve_Z"
        self._tmp3 = "tmp_solve_S"
        self._tmp4 = "tmp_solve_C"

        f = open(self._tmp2,'w')
        f.write("temporary file for ipopt output")
        f.close()

        f = open(self._tmp3,'w')
        f.write("temporary file for ipopt output")
        f.close()

        f = open(self._tmp4,'w')
        f.write("temporary file for ipopt output")
        f.close()
        
        self._profile_time = False
        
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
        
    def _build_s_model(self):
        self.S_model = ConcreteModel()
        self.S_model.S = Var(self._meas_lambdas,
                             self._mixture_components,
                             bounds=(0.0,None),
                             initialize=1.0)

        self.S_model.Z = Var(self._meas_times,
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

        obj = 0.0
        # asumes base model has been solved already for Z
        for t in self._meas_times:
            for l in self._meas_lambdas:
                D_bar = sum(self.S_model.S[l,k]*self.S_model.Z[t,k] for k in self._mixture_components)
                obj+=(D_bar-self.model.D[t,l])**2
                    
        self.S_model.objective = Objective(expr=obj)
        
    def _build_c_model(self):
        
        self.C_model = ConcreteModel()
        self.C_model.C = Var(self._meas_times,
                           self._mixture_components,
                           bounds=(0.0,None),
                           initialize=1.0)

        self.C_model.S = Var(self._meas_lambdas,
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
        
        obj=0.0
        # asumes that s model has been solved first
        for t in self._meas_times:
            for l in self._meas_lambdas:
                D_bar = sum(self.C_model.S[l,k]*self.C_model.C[t,k] for k in self._mixture_components)
                obj+=(self.model.D[t,l]-D_bar)**2
                    
        self.C_model.objective = Objective(expr=obj)

            
    def _solve_initalization(self,solver,**kwds):

        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee',False)
        set_A = kwds.pop('subset_lambdas',self._meas_lambdas)
                
        # build objective
        obj = 0.0
        for t in self._meas_times:
            for l in set_A:
                D_bar = sum(self.model.Z[t,k]*self.model.S[l,k] for k in self.model.mixture_components)
                obj+= (self.model.D[t,l] - D_bar)**2
        self.model.init_objective = Objective(expr=obj)

        print("Solving Initialization Problem\n")

        opt = SolverFactory(solver)

        for key, val in solver_opts.iteritems():
            opt.options[key]=val
        
        solver_results = opt.solve(self.model,
                                   tee=tee,
                                   report_timing=self._profile_time)

        for t in self._meas_times:
            for k in self._mixture_components:
                #self.model.C[t,k].value = np.random.normal(self.model.Z[t,k].value,1e-6)

                self.model.C[t,k].value = self.model.Z[t,k].value
                self.C_model.C[t,k].value =  self.model.C[t,k].value

        for l in self._meas_lambdas:
            for k in self._mixture_components:
                self.S_model.S[l,k].value = self.model.S[l,k].value
                
        self.model.del_component('init_objective')

    def _solve_Z(self,solver,**kwds):

        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee',False)
        
        obj = 0.0
        reciprocal_ntp = 1.0/len(self.model.time)
        
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

            
        opt = SolverFactory(solver)
        for key, val in solver_opts.iteritems():
            opt.options[key]=val

        solver_results = opt.solve(self.model,
                                   logfile=self._tmp2,
                                   tee=tee,
                                   #show_section_timing=True,
                                   report_timing=self._profile_time)
                                         
        self.model.del_component('z_objective')


    def _solve_S(self,solver,**kwds):

        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee',False)
        update_nl = kwds.pop('update_nl',False)

        if not update_nl:
            # assumes C has been solved already in the main  model
            for k,v in self.model.C.iteritems():
                self.S_model.Z[k].setlb(v.value)
                self.S_model.Z[k].setub(v.value)
        
            if self._profile_time:
                print('-----------------Solve_S--------------------')

            opt = SolverFactory(solver)

            for key, val in solver_opts.iteritems():
                opt.options[key]=val

            solver_results = opt.solve(self.S_model,
                                       logfile=self._tmp3,
                                       tee=tee,
                                       #keepfiles=True,
                                       #show_section_timing=True,
                                       report_timing=self._profile_time)
        else:
            
            self.write_s_nlfile(solver_opts=solver_opts)
            #self.write_s_nlfile2()
            print solver_opts
            call_ipopt("usmodel.nl",
                       opts=solver_opts,
                       stdout=self._tmp3)
            results_s = read_sol(self.S_model, "usmodel.sol", self._smodel_symbol_map)
            self.S_model.solutions.load_from(results_s)

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
                
    def _solve_C(self,solver,**kwds):

        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee',False)
        update_nl = kwds.pop('update_nl',False)
        
        if not update_nl:
            # assumes S has been solved already in the z model
            for k,v in self.model.S.iteritems():
                self.C_model.S[k].setlb(v.value)
                self.C_model.S[k].setub(v.value)

            if self._profile_time:
                print('-----------------Solve_C--------------------')

            opt = SolverFactory(solver)

            for key, val in solver_opts.iteritems():
                opt.options[key]=val
                
            solver_results = opt.solve(self.C_model,
                                       logfile=self._tmp4,
                                       tee=tee,
                                       #keepfiles=True,
                                       report_timing=self._profile_time)
        else:
            self.write_c_nlfile(solver_opts=solver_opts)
            print solver_opts
            call_ipopt("ucmodel.nl",
                       opts=solver_opts,
                       stdout=self._tmp4)
            results_c = read_sol(self.C_model, "ucmodel.sol", self._cmodel_symbol_map)
            self.C_model.solutions.load_from(results_c)
            
        #updates values in main model
        for k,v in self.C_model.C.iteritems():
            self.model.C[k].value = v.value

                
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
        with open(self._tmp2,'r') as tf:
            f.write("\n#######################Solve Z#######################\n")
            f.write(tf.read())
        
        tf = open(self._tmp3,'r')
        with open(self._tmp3,'r') as tf:
            f.write("\n#######################Solve S#######################\n")
            f.write(tf.read())
        with open(self._tmp4,'r') as tf:
            f.write("\n#######################Solve C#######################\n")
            f.write(tf.read())
        f.close()

    def parse_nlfiles2(self):
        #mark fixed variables
        count=103189060359
        first_count=count
        variable_map = dict()
        primer_mapa = dict()
        i=0
        self.ordered_z = dict()
        for k,v in self.S_model.Z.iteritems():
            v.value = count
            primer_mapeo['{}'.format(count)] = 'p{}'.format(i)
            self.ordered_z['p{}'.format(i)] = k
            i+=1
            count+=1

        count_mark = 203189060359
        old_values = dict()
        for k,v in self.S_model.S.iteritems():
            old_values[k] = v.value
            v.value = count_mark
            variable_map[count_mark]=k
            count_mark+=1
                        
        # write nlfile and symbol map 
        _, smap_id = self.S_model.write("s_model.nl")
        self._smodel_symbol_map = self.S_model.solutions.symbol_map[smap_id]

        # read the whole file
        f = file("s_model.nl",'r') 
        nl_string = f.read()
        f.close()

        # restore values of init vars
        for k,v in self.S_model.S.iteritems():
            v.value = old_values[k]

        # get string of variables
        regex = re.compile("0 2031890603*")
        r = regex.search(nl_string)
        first_s = r.group(0)
        start_x = nl_string.find('{}'.format(first_s))
        n_s = len(self.S_model.S)
        first_z = '{} 1.0'.format(n_s) #asumes bound-fixed vars initialze to 1.0
        end_x = nl_string.find(first_z,start_x)
        vars_string = nl_string[start_x:end_x]

        #print vars_string
        ordered_vars_s = list()
        var_count = 0
        for l in vars_string.split('\n'):
            if l:
                var_count = int(l.split()[1])
                ordered_vars_s.append(variable_map[var_count])

        self.ordered_vars_s = ordered_vars_s

        # get header 
        end_header = nl_string.find('common exprs: b,c,o,c1,o1')
        self.header_s = nl_string[:end_header] + 'common exprs: b,c,o,c1,o1\n'
        off_set = end_header+len('common exprs: b,c,o,c1,o1\n')
        self.suffix_str_s = ''
        obj1 = nl_string[off_set:start_x]
        self.objective_str_s = multiple_replace(obj1,primer_mapeo)  
        self.end_str_s = nl_string[end_x:]        
        

    def write_s_nlfile2(self,**kwds):
        solver_opts = kwds.pop('solver_opts', dict())
        
        segundo_mapeo = dict()
        for k,v in self.ordered_z.iteritems():
            segundo_mapeo[k] = self.model.C[k].value

        warm_start = 'warm_start_init_point'
        suffix_str = ''
        if solver_opts.has_key(warm_start):
            n_suffix = len(self.S_model.ipopt_zL_in)
            if n_suffix and solver_opts[warm_start]=='yes':
                suffix_str='S4 {} ipopt_zL_in\n'.format(n_suffix)
                for i,k in enumerate(self.ordered_vars_s):
                    suffix_str+="%d %r\n"%(i,self.S_model.ipopt_zL_in[self.S_model.S[k]])

        vars_str = ''
        for i,k in enumerate(self.ordered_vars_s):
            vars_str+="%d %r\n"%(i,self.S_model.S[k].value)

        obj2 = multiple_replace(self.objective_str_s,segundo_mapeo)  
        new_nl = self.header_s + suffix_str + obj2
        new_nl+= vars_str + self.rest_vars_str_s + bounds_str + self.end_str_s
        with open('usmodel.nl','w') as f:
            f.write(new_nl)
            
        
    def parse_nlfiles(self):
        #mark fixed variables
        count=103189060359
        first_count=count
        variable_map = dict()
        for k,v in self.S_model.Z.iteritems():
            v.setlb(count)
            v.setub(count)
            variable_map[count]=k
            count+=1

        count_mark = 203189060359
        old_values = dict()
        for k,v in self.S_model.S.iteritems():
            old_values[k] = v.value
            v.value = count_mark
            variable_map[count_mark]=k
            count_mark+=1
                        
        # write nlfile and symbol map 
        _, smap_id = self.S_model.write("s_model.nl")
        self._smodel_symbol_map = self.S_model.solutions.symbol_map[smap_id]

        # read the whole file
        f = file("s_model.nl",'r') 
        nl_string = f.read()
        f.close()

        # restore values of init vars
        for k,v in self.S_model.S.iteritems():
            v.value = old_values[k]

        # get string of variables
        regex = re.compile("0 2031890603*")
        r = regex.search(nl_string)
        first_s = r.group(0)
        start_x = nl_string.find('{}'.format(first_s))
        n_s = len(self.S_model.S)
        first_z = '{} 1.0'.format(n_s) #asumes bound-fixed vars initialze to 1.0
        end_x = nl_string.find(first_z,start_x)
        vars_string = nl_string[start_x:end_x]

        #print vars_string
        ordered_vars_s = list()
        var_count = 0
        for l in vars_string.split('\n'):
            if l:
                var_count = int(l.split()[1])
                ordered_vars_s.append(variable_map[var_count])

        self.ordered_vars_s = ordered_vars_s
        regex = re.compile("4 1031890603*")
        r = regex.search(nl_string)
        first_b = r.group(0)
        start_b = nl_string.find('{}'.format(first_b))
        self.nl_smodel1 = nl_string[:start_b]  
        bounds_string = nl_string[start_b:]
        end_b = bounds_string.find('{}'.format('k'))
        bounds_string = bounds_string[:end_b]

        # determines the order of the variables
        ordered_vars = list()
        var_count = 0
        for l in bounds_string.split('\n'):
            if l:
                var_count = int(l.split()[1])
                ordered_vars.append(variable_map[var_count])

        self.ordered_bounds_z = ordered_vars
        # end of file
        start_f = nl_string.find('{}'.format(var_count))
        end_string = nl_string[start_f:]
        end_string = end_string.replace(str(var_count)+'\n','')
        self.nl_smodel3 = end_string

        # get header 
        end_header = nl_string.find('common exprs: b,c,o,c1,o1')
        self.header_s = nl_string[:end_header] + 'common exprs: b,c,o,c1,o1\n'
        off_set = end_header+len('common exprs: b,c,o,c1,o1\n')
        self.suffix_str_s = ''
        self.objective_str_s = nl_string[off_set:start_x]
        self.rest_vars_str_s = nl_string[end_x:start_b]
        self.end_str_s = end_string
        #####################################################
        count=103189060359
        first_count=count
        variable_map = dict()
        for k,v in self.C_model.S.iteritems():
            v.setlb(count)
            v.setub(count)
            variable_map[count]=k
            count+=1

        count_mark = 203189060359
        old_values = dict()
        for k,v in self.C_model.C.iteritems():
            old_values[k] = v.value
            v.value = count_mark
            variable_map[count_mark]=k
            count_mark+=1

        # write nlfile and symbol map 
        _, smap_id = self.C_model.write("c_model.nl")
        self._cmodel_symbol_map = self.C_model.solutions.symbol_map[smap_id]

        f = file("c_model.nl",'r') 
        nl_string = f.read()
        f.close()

        # restore values of init vars
        for k,v in self.C_model.C.iteritems():
            v.value = old_values[k]

        # get string of variables
        regex = re.compile("0 2031890603*")
        r = regex.search(nl_string)
        first_c = r.group(0)
        start_x = nl_string.find('{}'.format(first_c))
        n_c = len(self.C_model.C)
        first_s = '{} 1.0'.format(n_c) #asumes bound-fixed vars initialze to 1.0
        end_x = nl_string.find(first_s,start_x)
        vars_string = nl_string[start_x:end_x]

        #print vars_string
        ordered_vars_c = list()
        var_count = 0
        for l in vars_string.split('\n'):
            if l:
                var_count = int(l.split()[1])
                ordered_vars_c.append(variable_map[var_count])

        self.ordered_vars_c = ordered_vars_c
        
        regex = re.compile("4 1031890603*")
        r = regex.search(nl_string)
        first_b = r.group(0)
        start_b = nl_string.find('{}'.format(first_b))
        self.nl_cmodel1 = nl_string[:start_b]  
        bounds_string = nl_string[start_b:]
        end_b = bounds_string.find('{}'.format('k'))
        bounds_string = bounds_string[:end_b]

        # determines the order of the variables
        ordered_vars = list()
        var_count = 0
        for l in bounds_string.split('\n'):
            if l:
                var_count = int(l.split()[1])
                ordered_vars.append(variable_map[var_count])

        self.ordered_bounds_c = ordered_vars
        # end of file
        start_f = nl_string.find('{}'.format(var_count))
        end_string = nl_string[start_f:]
        end_string = end_string.replace(str(var_count)+'\n','')
        self.nl_cmodel3 = end_string


        end_header = nl_string.find('common exprs: b,c,o,c1,o1')
        self.header_c = nl_string[:end_header] + 'common exprs: b,c,o,c1,o1\n'
        off_set = end_header+len('common exprs: b,c,o,c1,o1\n')
        self.suffix_str_c = ''
        self.objective_str_c = nl_string[off_set:start_x]
        self.rest_vars_str_c = nl_string[end_x:start_b]
        self.end_str_c = end_string

    def write_s_nlfile(self,**kwds):

        solver_opts = kwds.pop('solver_opts', dict())

        for k,v in self.model.C.iteritems():
            self.S_model.Z[k].value = v.value

        warm_start = 'warm_start_init_point'
        suffix_str = ''
        if solver_opts.has_key(warm_start):
            n_suffix = len(self.S_model.ipopt_zL_in)
            if n_suffix and solver_opts[warm_start]=='yes':
                suffix_str='S4 {} ipopt_zL_in\n'.format(n_suffix)
                for i,k in enumerate(self.ordered_vars_s):
                    suffix_str+="%d %r\n"%(i,self.S_model.ipopt_zL_in[self.S_model.S[k]])

        vars_str = ''
        for i,k in enumerate(self.ordered_vars_s):
            vars_str+="%d %r\n"%(i,self.S_model.S[k].value)

        bounds_str=''
        for k in self.ordered_bounds_z:
            bounds_str+="4 %r\n" % self.model.C[k].value
            
        new_nl = self.header_s + suffix_str + self.objective_str_s
        new_nl+= vars_str + self.rest_vars_str_s + bounds_str + self.end_str_s
        with open('usmodel.nl','w') as f:
            f.write(new_nl)
        
        

    def write_c_nlfile(self,**kwds):        

        solver_opts = kwds.pop('solver_opts', dict())
        
        for k,v in self.model.S.iteritems():
            self.C_model.S[k].value = v.value

        warm_start = 'warm_start_init_point'
        suffix_str = ''
        if solver_opts.has_key(warm_start):
            n_suffix = len(self.C_model.ipopt_zL_in)
            print "yes"
            if n_suffix and solver_opts[warm_start]=='yes':
                suffix_str='S4 {} ipopt_zL_in\n'.format(n_suffix)
                for i,k in enumerate(self.ordered_vars_c):
                    suffix_str+="%d %r\n"%(i,self.C_model.ipopt_zL_in[self.C_model.C[k]])

        vars_str = ''
        for i,k in enumerate(self.ordered_vars_c):
            vars_str+="%d %r\n"%(i,self.C_model.C[k].value)

        bounds_str=''
        for k in self.ordered_bounds_c:
            bounds_str+="4 %r\n" % self.model.S[k].value

        new_nl = self.header_c + suffix_str + self.objective_str_c
        new_nl+= vars_str + self.rest_vars_str_c + bounds_str + self.end_str_c
        with open('ucmodel.nl','w') as f:
            f.write(new_nl)
        
    def run_opt(self,solver,**kwds):

        solver_opts = kwds.pop('solver_opts', dict())
        variances = kwds.pop('variances',dict())
        tee = kwds.pop('tee',False)
        norm_order = kwds.pop('norm',np.inf)
        max_iter = kwds.pop('max_iter',400)
        tol = kwds.pop('tolerance',1e-5)
        A = kwds.pop('subset_lambdas',None)
        update_files = True
        
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
        self._solve_initalization(solver,
                                  tee=True,
                                  subset_lambdas=A)

        Z_i = np.array([v.value for v in self.model.Z.itervalues()])

        print("Solving Variance estimation")
        self.parse_nlfiles()
        #perfoms the fisrt iteration 
        self._solve_Z(solver)
        self._solve_s_scipy()
        self._solve_c_scipy()
        #self._solve_S(solver,update_nl=update_files,tee=True)
        #self._solve_C(solver,update_nl=update_files,tee=True)

        if tee:
            filename = "iterations.log"
            if os.path.isfile(filename):
                os.remove(filename)
            self._log_iterations(filename,0)

        
        Z_i1 = np.array([v.value for v in self.model.Z.itervalues()])

        diff = Z_i1-Z_i
        norm_diff = np.linalg.norm(diff,norm_order)

        print("{: >11} {: >16}".format('Iter','|Zi-Zi+1|'))
        #print("{: >10} {: >20}".format(0,norm_diff))
        count=1
        norm_diff=1
        options_dict=dict()
        while norm_diff>tol and count<max_iter:
            if count>2:
                self.model.ipopt_zL_in.update(self.model.ipopt_zL_out)
                self.model.ipopt_zU_in.update(self.model.ipopt_zU_out)

                # only update lower bounds since the c,s models do not have upper
                self.S_model.ipopt_zL_in.update(self.S_model.ipopt_zL_out)
                self.C_model.ipopt_zL_in.update(self.C_model.ipopt_zL_out)
                
                options_dict['warm_start_init_point'] = 'yes'
                options_dict['warm_start_bound_push'] = 1e-6
                options_dict['warm_start_mult_bound_push'] = 1e-6
                options_dict['mu_init'] = 1e-6
                
            Z_i = Z_i1
            self._solve_Z(solver,solver_opts=options_dict,tee=True)
            #self._solve_S(solver,solver_opts=options_dict,update_nl=update_files,tee=True)
            #self._solve_C(solver,solver_opts=options_dict,update_nl=update_files,tee=True)
            self._solve_s_scipy()
            self._solve_c_scipy()
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

            
            
def read_sol(model, sol_filename, symbol_map, suffixes=[".*"]):
    """
    Reads the solution from the SOL file and generates a
    results object with an appropriate symbol map for
    loading it into the given Pyomo model. By default all
    suffixes found in the NL file will be extracted. This
    can be overridden using the suffixes keyword, which
    should be a list of suffix names or regular expressions
    (or None).
    """
    if suffixes is None:
        suffixes = []


    # parse the SOL file
    with ReaderFactory(ResultsFormat.sol) as reader:
        results = reader(sol_filename, suffixes=suffixes)


    # tag the results object with the symbol_map
    results._smap = symbol_map

    return results
    
    
def call_ipopt(nlfilename,opts={},stdout=None):
    cmd=''
    cmd = 'ipopt -s '
    cmd+='{} '.format(nlfilename)
    for k,v in opts.iteritems():
        cmd+='{}={} '.format(k,v)
    
    #print cmd
    args = cmd.split()
    
    if stdout is not None:
        subprocess.call(args)
        
        #with open(stdout, 'w') as f:
        #    subprocess.call(args,stdout=f)
    else:
        
        subprocess.call(args)
    
def multiple_replace(text, adict):
    rx = re.compile('|'.join(adict.iterkeys()))
    def one_xlat(match):
        return adict[match.group(0)]
    return rx.sub(one_xlat, text)
