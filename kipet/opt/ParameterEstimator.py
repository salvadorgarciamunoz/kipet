from pyomo.environ import *
from pyomo.dae import *
from kipet.opt.Optimizer import *
from pyomo.core.base.expr import Expr_if

import StringIO
import copy
import re
import os


class ParameterEstimator(Optimizer):
    """Optimizer for parameter estimation.

    Parameters
    ----------
    model : Pyomo model
        Pyomo model to be used in the parameter estimation

    """
    def __init__(self,model):
        super(ParameterEstimator, self).__init__(model)
        # for reduce hessian
        self._idx_to_variable = dict()
        
    def run_sim(self,solver,**kdws):
        raise NotImplementedError("ParameterEstimator object does not have run_sim method. Call run_opt")       

    def _solve_extended_model(self, sigma_sq, optimizer, **kwds):
        """Solves estimation based on spectral data. (known variances)

           This method is not intended to be used by users directly
        Args:
            sigma_sq (dict): variances 
            
            optimizer (SolverFactory): Pyomo Solver factory object
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not

            with_d_vars (bool,optional): flag to the optimizer whether to add 
            variables and constraints for D_bar(i,j)
        
        Returns:
            None
        """
        
        tee = kwds.pop('tee',False)
        with_d_vars = kwds.pop('with_d_vars', False)
        weights = kwds.pop('weights', [1.0, 1.0])
        covariance = kwds.pop('covariance', False)
        species_list = kwds.pop('subset_components', None)

        list_components = []
        if species_list is None:
            list_components = [k for k in self._mixture_components]
        else:
            for k in species_list:
                if k in self._mixture_components:
                    list_components.append(k)
                else:
                    warnings.warn("Ignored {} since is not a mixture component of the model".format(k))
        if not self._spectra_given:
            raise NotImplementedError("Extended model requires spectral data model.D[ti,lj]")
            
        all_sigma_specified = True
        keys = sigma_sq.keys()
        for k in list_components:
            if k not in keys:
                all_sigma_specified = False
                sigma_sq[k] = max(sigma_sq.values())

        if not sigma_sq.has_key('device'):
            all_sigma_specified = False
            sigma_sq['device'] = 1.0
            
        m = self.model

        if with_d_vars:
            m.D_bar = Var(m.meas_times,
                          m.meas_lambdas)

            def rule_D_bar(m,t,l):
                return m.D_bar[t,l] == sum(m.C[t,k]*m.S[l,k] for k in m.mixture_components)
            m.D_bar_constraint = Constraint(m.meas_times,
                                            m.meas_lambdas,
                                            rule=rule_D_bar)

        # estimation
        def rule_objective(m):
            expr = 0
            for t in m.meas_times:
                for l in m.meas_lambdas:
                    if with_d_vars:
                        expr+= (m.D[t,l] - m.D_bar[t,l])**2/(sigma_sq['device'])
                    else:
                        D_bar = sum(m.C[t, k]*m.S[l, k] for k in list_components)
                        expr+= (m.D[t,l] - D_bar)**2/(sigma_sq['device'])

            expr*=weights[0]
            second_term = 0.0
            for t in m.meas_times:
                second_term += sum((m.C[t,k]-m.Z[t,k])**2/sigma_sq[k] for k in list_components)

            expr+=weights[1]*second_term
            return expr
        
        m.objective = Objective(rule=rule_objective)
        
        #solver_results = optimizer.solve(m,tee=True,
        #                                 report_timing=True)

        if covariance:
            self._tmpfile = "ipopt_hess"
            solver_results = optimizer.solve(m,tee=False,
                                             logfile=self._tmpfile,
                                             report_timing=True)

            print "Done solving building reduce hessian"
            output_string = ''
            with open(self._tmpfile,'r') as f:
                output_string = f.read()
            if os.path.exists(self._tmpfile):
                os.remove(self._tmpfile)
            #output_string = f.getvalue()
            ipopt_output,hessian_output = split_sipopt_string(output_string)
            #print hessian_output
            print "build strings"
            if tee==True:
                print ipopt_output
            
            if not all_sigma_specified:
                raise RuntimeError('All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            hessian = read_reduce_hessian(hessian_output,n_vars)
            #hessian = read_reduce_hessian2(hessian_output,n_vars)
            #print hessian
            self._compute_covariance(hessian,sigma_sq)
        else:
            solver_results = optimizer.solve(m,tee=tee)

        if with_d_vars:
            m.del_component('D_bar')
            m.del_component('D_bar_constraint')
        m.del_component('objective')

    def _define_reduce_hess_order(self):
        self.model.red_hessian = Suffix(direction=Suffix.IMPORT_EXPORT)

        count_vars = 1
        for t in self._meas_times:
            for c in self._mixture_components:
                v = self.model.C[t,c]
                self._idx_to_variable[count_vars] = v
                self.model.red_hessian[v] = count_vars
                count_vars+=1

        for l in self._meas_lambdas:
            for c in self._mixture_components:
                v = self.model.S[l,c]
                self._idx_to_variable[count_vars] = v
                self.model.red_hessian[v] = count_vars
                count_vars+=1

        for v in self.model.P.itervalues():
            self._idx_to_variable[count_vars] = v
            self.model.red_hessian[v] = count_vars
            count_vars+=1

    def _compute_covariance(self, hessian, variances):

        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_components
        nparams = len(self.model.P)
        nd = nw*nt
        ntheta = nc*(nw+nt)+ nparams

        print "Computing H matrix\n shape ({},{})".format(nparams,ntheta)
        all_H = hessian
        H = all_H[-nparams:,:]
        #H = hessian
        print "Computing B matrix\n shape ({},{})".format(ntheta,nd)
        self._compute_B_matrix(variances)
        B = self.B_matrix
        print "Computing Vd matrix\n shape ({},{})".format(nd,nd)
        self._compute_Vd_matrix(variances)
        Vd = self.Vd_matrix
        """
        Vd_dense = Vd.toarray()
        print "multiplying H*B"
        M1 = H.dot(B)
        print "multiplying H*B*Vd"
        M2 = M1.dot(Vd_dense)
        print "multiplying H*B*Vd*Bt"
        M3 = M2.dot(B.T)
        print "multiplying H*B*Vd*Bt*Ht"
        V_theta = M3.dot(H)
        """

        #R = B.T.dot(H)
        R = B.T.dot(H.T)
        A = Vd.dot(R)
        L = H.dot(B)
        Vtheta = A.T.dot(L.T)
        V_theta = Vtheta.T
        
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_components
        nparams = len(self.model.P)
        # this changes depending on the order of the suffixes passed to sipopt
        nd = nw*nt
        ntheta = nc*(nw+nt)
        #V_param = V_theta[ntheta:ntheta+nparams,ntheta:ntheta+nparams]
        V_param = V_theta
        variances_p = np.diag(V_param)
        print('\nConfindence intervals:')
        i=0
        for k,p in self.model.P.iteritems():
            print '{} ({},{})'.format(k, p.value-variances_p[i]**0.5, p.value+variances_p[i]**0.5)
            i=+1
        return 1
    
    def _compute_B_matrix(self, variances, **kwds):
        """Builds B matrix for calculation of covariances

           This method is not intended to be used by users directly

        Args:
            variances (dict): variances 
            
        Returns:
            None
        """

        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_components
        nparams = len(self.model.P)
        # this changes depending on the order of the suffixes passed to sipopt
        nd = nw*nt
        ntheta = nc*(nw+nt)+ nparams
        self.B_matrix = np.zeros((ntheta,nw*nt))
        for i,t in enumerate(self.model.meas_times):
            for j,l in enumerate(self.model.meas_lambdas):
                for k,c in enumerate(self.model.mixture_components):
                    #r_idx1 = k*nt+i
                    r_idx1 = i*nc+k
                    #r_idx2 = k*nw+j+nc*nt
                    r_idx2 = j*nc+k +nc*nw
                    #c_idx = i+j*nt
                    c_idx = i*nw+j
                    self.B_matrix[r_idx1,c_idx] = -2*self.model.S[l,c].value/variances['device']
                    self.B_matrix[r_idx2,c_idx] = -2*self.model.C[t,c].value/variances['device']

    def _compute_Vd_matrix(self, variances, **kwds):
        """Builds d covariance matrix

           This method is not intended to be used by users directly

        Args:
            variances (dict): variances 
            
        Returns:
            None
        """
        
        # add check for model already solved
        row =  []
        col =  []
        data = []
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_components
        """
        for i,t in enumerate(self.model.meas_times):
            for j,l in enumerate(self.model.meas_lambdas):
                for q,tt in enumerate(self.model.meas_times):
                    for p,ll in enumerate(self.model.meas_lambdas):
                        if i==q and j!=p:
                            val = sum(variances[c]*self.model.S[l,c].value*self.model.S[ll,c].value for c in self.model.mixture_components)
                            row.append(i*nw+j)
                            col.append(q*nw+p)
                            data.append(val)
                        if i==q and j==p:
                            val = sum(variances[c]*self.model.S[l,c].value**2 for c in self.model.mixture_components)+variances['device']
                            row.append(i*nw+j)
                            col.append(q*nw+p)
                            data.append(val)
        """
        s_array = np.zeros(nw*nc)
        v_array = np.zeros(nc)
        for k,c in enumerate(self.model.mixture_components):
            v_array[k] = variances[c]

        for j,l in enumerate(self.model.meas_lambdas):
            for k,c in enumerate(self.model.mixture_components):
                s_array[j*nc+k] = self.model.S[l,c].value
                
        row =  []
        col =  []
        data = []
        nd = nt*nw
        #Vd_dense = np.zeros((nd,nd))
        v_device = variances['device']
        for i in xrange(nt):
            for j in xrange(nw):
                val = sum(v_array[k]*s_array[j*nc+k]**2 for k in xrange(nc))+v_device
                row.append(i*nw+j)
                col.append(i*nw+j)
                data.append(val)
                #Vd_dense[i*nw+j,i*nw+j] = val
                for p in xrange(nw):
                    if j!=p:
                        val = sum(v_array[k]*s_array[j*nc+k]*s_array[p*nc+k] for k in xrange(nc))
                        row.append(i*nw+j)
                        col.append(i*nw+p)
                        data.append(val)
                        #Vd_dense[i*nw+j,i*nw+p] = val
        
        
        self.Vd_matrix = scipy.sparse.coo_matrix((data, (row, col)),
                                                 shape=(nd,nd)).tocsr()
        #self.Vd_matrix = Vd_dense

    def run_opt(self, solver, **kwds):

        """ Solves parameter estimation problem.
        
        Args:
            solver (str): name of the nonlinear solver to used
          
            solver_opts (dict, optional): options passed to the nonlinear solver
        
            variances (dict, optional): map of component name to noise variance. The
            map also contains the device noise variance
            
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
            
            with_d_vars (bool,optional): flag to the optimizer whether to add 
            variables and constraints for D_bar(i,j)

        Returns:
            Results object with loaded results

        """

        solver_opts = kwds.pop('solver_opts',  dict())
        variances = kwds.pop('variances', dict())
        tee = kwds.pop('tee', False)
        with_d_vars = kwds.pop('with_d_vars', False)
        covariance = kwds.pop('covariance', False)
        
        if not self.model.time.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')
        
        # Look at the output in results
        opt = SolverFactory(solver)

        if covariance:
            if solver!='ipopt_sens':
                raise RuntimeError('To get covariance matrix the solver needs to be ipopt_sens')
            if not solver_opts.has_key('compute_red_hessian'):
                solver_opts['compute_red_hessian'] = 'yes'
            
            self._define_reduce_hess_order()

        for key, val in solver_opts.iteritems():
            opt.options[key] = val

        active_objectives = [o for o in self.model.component_map(Objective, active=True)]
        if active_objectives:
            print("WARNING: The model has an active objective. Running optimization with models objective.\n To solve optimization with default objective (Weifengs) deactivate all objectives in the model.")
            solver_results = opt.solve(self.model, tee=tee)
        else:
            self._solve_extended_model(variances, opt,
                                       tee=tee,
                                       covariance=covariance,
                                       with_d_vars=with_d_vars,
                                       **kwds)
            
        results = ResultsObject()

        results.load_from_pyomo_model(self.model,
                                      to_load=['Z','dZdt','X','dXdt','C','S','Y'])
            
        self.compute_D_given_SC(results)
        
        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name].value

        results.P = param_vals

        return results


def split_sipopt_string(output_string):
    start_hess = output_string.find('DenseSymMatrix')
    ipopt_string = output_string[:start_hess]
    hess_string = output_string[start_hess:]
    return (ipopt_string,hess_string)


def read_reduce_hessian2(hessian_string, n_vars):
    hessian_string  = re.sub('RedHessian unscaled\[', '', hessian_string)
    hessian_string  = re.sub('\]=', ',', hessian_string)
    
    hessian = np.zeros((n_vars,n_vars))
    for i,line in enumerate(hessian_string.split('\n')):
        if i>0:
            hess_line = line.split(',') 
            if len(hess_line)==3:
                row = int(hess_line[0])
                col = int(hess_line[1])
                hessian[row,col] = float(hess_line[2])
                hessian[col,row] = float(hess_line[2])
    return hessian


def read_reduce_hessian(hessian_string,n_vars):
    
    hessian = np.zeros((n_vars,n_vars))
    for i,line in enumerate(hessian_string.split('\n')):
        if i>0: # ignores header
            if line not in ['',' ','\t']:
                hess_line = line.split(']=')
                if len(hess_line)==2:
                    value = float(hess_line[1])
                    column_line = hess_line[0].split(',')
                    col = int(column_line[1])
                    row_line = column_line[0].split('[')
                    row = int(row_line[1])
                    hessian[row,col] = float(value)
                    hessian[col,row] = float(value)
    return hessian
