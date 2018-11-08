# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
from kipet.library.Optimizer import *
from pyomo.core.base.expr import Expr_if
import numpy as np
import six
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

    def __init__(self, model):
        super(ParameterEstimator, self).__init__(model)
        # for reduce hessian
        self._idx_to_variable = dict()
        self._n_actual = self._n_components
        if hasattr(self.model, 'non_absorbing'):
            warnings.warn("Overriden by non_absorbing")
            list_components = [k for k in self._mixture_components if k not in self._non_absorbing]
            self._sublist_components = list_components
            self._n_actual = len(self._sublist_components)
        else:
            self._sublist_components = [k for k in self._mixture_components]

    def run_sim(self, solver, **kdws):
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

        tee = kwds.pop('tee', False)
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

        if hasattr(self.model, 'non_absorbing'):
            warnings.warn("Overriden by non_absorbing!!!")
            list_components = [k for k in self._mixture_components if k not in self._non_absorbing]

        all_sigma_specified = True
        print(sigma_sq)
        keys = sigma_sq.keys()
        for k in list_components:
            if k not in keys:
                all_sigma_specified = False
                sigma_sq[k] = max(sigma_sq.values())

        if not 'device' in sigma_sq.keys():
            all_sigma_specified = False
            sigma_sq['device'] = 1.0

        m = self.model

        if with_d_vars:
            m.D_bar = Var(m.meas_times,
                          m.meas_lambdas)

            def rule_D_bar(m, t, l):
                return m.D_bar[t, l] == sum(m.C[t, k] * m.S[l, k] for k in self._sublist_components)

            m.D_bar_constraint = Constraint(m.meas_times,
                                            m.meas_lambdas,
                                            rule=rule_D_bar)
        # estimation
        def rule_objective(m):
            expr = 0
            for t in m.meas_times:
                for l in m.meas_lambdas:
                    if with_d_vars:
                        expr += (m.D[t, l] - m.D_bar[t, l]) ** 2 / (sigma_sq['device'])
                    else:
                        D_bar = sum(m.C[t, k] * m.S[l, k] for k in list_components)
                        expr += (m.D[t, l] - D_bar) ** 2 / (sigma_sq['device'])

            expr *= weights[0]
            second_term = 0.0
            for t in m.meas_times:
                second_term += sum((m.C[t, k] - m.Z[t, k]) ** 2 / sigma_sq[k] for k in list_components)

            expr += weights[1] * second_term
            return expr

        m.objective = Objective(rule=rule_objective)

        # solver_results = optimizer.solve(m,tee=True,
        #                                 report_timing=True)

        if covariance and self.solver=='ipopt_sens':
            self._tmpfile = "ipopt_hess"
            solver_results = optimizer.solve(m, tee=False,
                                             logfile=self._tmpfile,
                                             report_timing=True)

            print("Done solving building reduce hessian")
            output_string = ''
            with open(self._tmpfile, 'r') as f:
                output_string = f.read()
            if os.path.exists(self._tmpfile):
                os.remove(self._tmpfile)
            # output_string = f.getvalue()
            ipopt_output, hessian_output = split_sipopt_string(output_string)
            # print hessian_output
            print("build strings")
            if tee == True:
                print(ipopt_output)

            if not all_sigma_specified:
                raise RuntimeError(
                    'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            hessian = read_reduce_hessian(hessian_output, n_vars)
            print(hessian.size, "hessian size")
            # hessian = read_reduce_hessian2(hessian_output,n_vars)
            # print hessian
            self._compute_covariance(hessian, sigma_sq)
            
        if covariance and self.solver == 'k_aug':            
            m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

            m.dof_v = Suffix(direction=Suffix.EXPORT)  #: SUFFIX FOR K_AUG
            m.rh_name = Suffix(direction=Suffix.IMPORT)  #: SUFFIX FOR K_AUG AS WELL
            
            count_vars = 1

            if not self._spectra_given:
                pass
            else:
                for t in self._meas_times:
                    for c in self._sublist_components:
                        m.C[t, c].set_suffix_value(m.dof_v,count_vars)
                        
                        count_vars += 1
        
            if not self._spectra_given:
                pass
            else:
                for l in self._meas_lambdas:
                    for c in self._sublist_components:
                        m.S[l, c].set_suffix_value(m.dof_v,count_vars)
                        count_vars += 1
                    
            for v in six.itervalues(self.model.P):
                if v.is_fixed():
                    continue
                m.P.set_suffix_value(m.dof_v,count_vars)
                count_vars += 1
           
            self._tmpfile = "k_aug_hess"
            ip = SolverFactory('ipopt')
            solver_results = ip.solve(m, tee=False,
                                             logfile=self._tmpfile,
                                             report_timing=True)
            k_aug = SolverFactory('k_aug')
            #k_aug.options["compute_inv"] = ""
            m.ipopt_zL_in.update(m.ipopt_zL_out)  #: be sure that the multipliers got updated!
            m.ipopt_zU_in.update(m.ipopt_zU_out)
            #m.write(filename="mynl.nl", format=ProblemFormat.nl)
            k_aug.solve(m, tee=False)
            print("Done solving building reduce hessian")

            if not all_sigma_specified:
                raise RuntimeError(
                    'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            print("n_vars",n_vars)
            #m.rh_name.pprint()
            var_loc = m.rh_name
            for v in six.itervalues(self._idx_to_variable):
                try:
                    var_loc[v]
                except:
                    #print(v, "is an error")
                    var_loc[v] = 0
                    #print(v, "is thus set to ", var_loc[v])
                    #print(var_loc[v])

            vlocsize = len(var_loc)
            #print("var_loc size, ", vlocsize) 
            unordered_hessian = np.loadtxt('result_red_hess.txt')
            if os.path.exists('result_red_hess.txt'):
                os.remove('result_red_hess.txt')
            #hessian = read_reduce_hessian_k_aug(hessian_output, n_vars)
            #hessian =hessian_output
            #print(hessian)
            print(unordered_hessian.size, "unordered hessian size")
            hessian = self._order_k_aug_hessian(unordered_hessian, var_loc)
            self._compute_covariance(hessian, sigma_sq)
        else:
            solver_results = optimizer.solve(m, tee=tee)

        if with_d_vars:
            m.del_component('D_bar')
            m.del_component('D_bar_constraint')
        m.del_component('objective')
        
        
    def _solve_model_given_c(self, sigma_sq, optimizer, **kwds):
        """Solves estimation based on concentration data. (known variances)

           This method is not intended to be used by users directly
        Args:
            sigma_sq (dict): variances 
            
            optimizer (SolverFactory): Pyomo Solver factory object
        
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
        
        Returns:
            None
        """

        tee = kwds.pop('tee', False)
        weights = kwds.pop('weights', [0.0, 1.0])
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

        if not self._concentration_given:
            raise NotImplementedError("Parameter Estimation from concentration data requires concentration data model.C[ti,cj]")

        #if hasattr(self.model, 'non_absorbing'):
        #    warnings.warn("Overriden by non_absorbing!!!")
        #    list_components = [k for k in self._mixture_components if k not in self._non_absorbing]

        all_sigma_specified = True
        print(sigma_sq)
        keys = sigma_sq.keys()
        for k in list_components:
            if k not in keys:
                all_sigma_specified = False
                sigma_sq[k] = max(sigma_sq.values())

        #if not 'device' in sigma_sq.keys():
        #    all_sigma_specified = False
        #    sigma_sq['device'] = 1.0

        m = self.model

        # estimation
        def rule_objective(m):
            obj = 0
            for t in m.meas_times:
                obj += sum((m.C[t, k] - m.Z[t, k]) ** 2 / sigma_sq[k] for k in list_components)

            return obj
            
        m.objective = Objective(rule=rule_objective)

        # solver_results = optimizer.solve(m,tee=True,
        #                                 report_timing=True)
        
        if covariance and self.solver == 'ipopt_sens':
            self._tmpfile = "ipopt_hess"
            solver_results = optimizer.solve(m, tee=False,
                                             logfile=self._tmpfile,
                                             report_timing=True)
            #self.model.red_hessian.pprint
            #m.P.pprint()
            print("Done solving building reduce hessian")
            output_string = ''
            with open(self._tmpfile, 'r') as f:
                output_string = f.read()
                
                print("output_string", output_string)
            if os.path.exists(self._tmpfile):
                os.remove(self._tmpfile)
            # output_string = f.getvalue()
            ipopt_output, hessian_output = split_sipopt_string(output_string)
            # print hessian_output
            print("build strings")
            #if tee == True:
            #    print(ipopt_output)

            if not all_sigma_specified:
                raise RuntimeError(
                    'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            hessian = read_reduce_hessian(hessian_output, n_vars)
            print(hessian.size, "hessian size")
            # hessian = read_reduce_hessian2(hessian_output,n_vars)
            if self._concentration_given:
                self._compute_covariance_C(hessian, sigma_sq)
            #else:
            #    self._compute_covariance(hessian, sigma_sq)
            
        if covariance and self.solver == 'k_aug':            
            m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

            m.dof_v = Suffix(direction=Suffix.EXPORT)  #: SUFFIX FOR K_AUG
            m.rh_name = Suffix(direction=Suffix.IMPORT)  #: SUFFIX FOR K_AUG AS WELL
            
            count_vars = 1

            if not self._spectra_given:
                pass
            else:
                for t in self._meas_times:
                    for c in self._sublist_components:
                        m.C[t, c].set_suffix_value(m.dof_v,count_vars)
                        
                        count_vars += 1
        
            if not self._spectra_given:
                pass
            else:
                for l in self._meas_lambdas:
                    for c in self._sublist_components:
                        m.S[l, c].set_suffix_value(m.dof_v,count_vars)
                        count_vars += 1
                    
            for v in six.itervalues(self.model.P):
                if v.is_fixed():
                    continue
                m.P.set_suffix_value(m.dof_v,count_vars)
                count_vars += 1
                
            self._tmpfile = "k_aug_hess"
            ip = SolverFactory('ipopt')
            solver_results = ip.solve(m, tee=True,
                                             logfile=self._tmpfile,
                                             report_timing=True)
            m.P.pprint()
            k_aug = SolverFactory('k_aug')
            
            #k_aug.options["no_scale"] = ""
            m.ipopt_zL_in.update(m.ipopt_zL_out)  #: be sure that the multipliers got updated!
            m.ipopt_zU_in.update(m.ipopt_zU_out)
            #m.write(filename="mynl.nl", format=ProblemFormat.nl)
            k_aug.solve(m, tee=True)
            print("Done solving building reduce hessian")

            if not all_sigma_specified:
                raise RuntimeError(
                    'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            print("n_vars",n_vars)
            #m.rh_name.pprint()
            var_loc = m.rh_name
            for v in six.itervalues(self._idx_to_variable):
                try:
                    var_loc[v]
                except:
                    print(v, "is an error")
                    var_loc[v] = 0
                    print(v, "is thus set to ", var_loc[v])
                    print(var_loc[v])

            vlocsize = len(var_loc)
            print("var_loc size, ", vlocsize) 
            unordered_hessian = np.loadtxt('result_red_hess.txt')
            if os.path.exists('result_red_hess.txt'):
                os.remove('result_red_hess.txt')
            #hessian = read_reduce_hessian_k_aug(hessian_output, n_vars)
            #hessian =hessian_output
            #print(hessian)
            print(unordered_hessian.size, "unordered hessian size")
            hessian = self._order_k_aug_hessian(unordered_hessian, var_loc)
            
            if self._concentration_given:
                self._compute_covariance_C(hessian, sigma_sq)
        else:
            solver_results = optimizer.solve(m, tee=tee)

        m.del_component('objective')

    def _define_reduce_hess_order(self):
        self.model.red_hessian = Suffix(direction=Suffix.IMPORT_EXPORT)
        count_vars = 1

        if not self._spectra_given:
            pass
        else:
            for t in self._meas_times:
                for c in self._sublist_components:
                    v = self.model.C[t, c]
                    self._idx_to_variable[count_vars] = v
                    self.model.red_hessian[v] = count_vars
                    count_vars += 1
        
        if not self._spectra_given:
            pass

        else:
            for l in self._meas_lambdas:
                for c in self._sublist_components:
                    v = self.model.S[l, c]
                    self._idx_to_variable[count_vars] = v
                    self.model.red_hessian[v] = count_vars
                    count_vars += 1
                    
        for v in six.itervalues(self.model.P):
            if v.is_fixed():
                print(v, end='\t')
                print("is fixed")
                continue
            self._idx_to_variable[count_vars] = v
            self.model.red_hessian[v] = count_vars
            count_vars += 1

    def _compute_covariance(self, hessian, variances):

        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_actual
        nparams = 0
        for v in six.itervalues(self.model.P):
            if v.is_fixed():  #: Skip the fixed ones
                print(str(v) + '\has been skipped for covariance calculations')
                continue
            nparams += 1
        # nparams = len(self.model.P)
        nd = nw * nt
        ntheta = nc * (nw + nt) + nparams

        print("Computing H matrix\n shape ({},{})".format(nparams, ntheta))
        all_H = hessian
        H = all_H[-nparams:, :]
        # H = hessian
        print("Computing B matrix\n shape ({},{})".format(ntheta, nd))
        self._compute_B_matrix(variances)
        B = self.B_matrix
        print("Computing Vd matrix\n shape ({},{})".format(nd, nd))
        self._compute_Vd_matrix(variances)
        Vd = self.Vd_matrix
        """
        Vd_dense = Vd.toarray()
        print("multiplying H*B")
        M1 = H.dot(B)
        print("multiplying H*B*Vd")
        M2 = M1.dot(Vd_dense)
        print("multiplying H*B*Vd*Bt")
        M3 = M2.dot(B.T)
        print("multiplying H*B*Vd*Bt*Ht")
        V_theta = M3.dot(H)
        """

        # R = B.T.dot(H)
        R = B.T.dot(H.T)
        A = Vd.dot(R)
        L = H.dot(B)
        Vtheta = A.T.dot(L.T)
        V_theta = Vtheta.T

        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_actual
        nparams = 0
        for v in six.itervalues(self.model.P):
            if v.is_fixed():  #: Skip the fixed ones ;)
                continue
            nparams += 1

        # this changes depending on the order of the suffixes passed to sipopt
        nd = nw * nt
        ntheta = nc * (nw + nt)
        # V_param = V_theta[ntheta:ntheta+nparams,ntheta:ntheta+nparams]
        V_param = V_theta
        variances_p = np.diag(V_param)
        print('\nConfidence intervals:')
        i = 0
        for k, p in self.model.P.items():
            if p.is_fixed():
                continue
            print('{} ({},{})'.format(k, p.value - variances_p[i] ** 0.5, p.value + variances_p[i] ** 0.5))
            i += 1
        return 1

    def _compute_covariance_C(self, hessian, variances):
        """
        Computes the covariance matrix for the paramaters taking in the Hessian
        matrix and the variances for the problem where only C data is provided.
        Outputs the parameter confidence intervals.
        
        This function is not intended to be used by the users directly
    
        """
        self._compute_residuals()
        res = self.residuals
        #print(res)
        #sets up matrix with variances in diagonals
        nc = self._n_actual
        nt = self._n_meas_times
        varmat = np.zeros((nc,nc))
        for c,k in enumerate(self._sublist_components):
            varmat[c,c]=variances[k]
        #print("varmat",varmat)
        #R=varmat.dot(res)
        #L = res.dot(varmat)
        E = 0
        for t in self._meas_times:
            for k in self._sublist_components:
                E += res[t,k]/(variances[k]**2)
        
        
        #Now we can use the E matrix with the hessian to estimate our confidence intervals
        nparams = 0
        for v in six.itervalues(self.model.P):
            if v.is_fixed():  #: Skip the fixed ones
                print(str(v) + '\has been skipped for covariance calculations')
                continue
            nparams += 1
        all_H = hessian
        H = all_H[-nparams:, :]

        #print(E_matrix)
        #covariance_C = E_matrix.dot(H.T)
        
        #print("value of the objective function (sum of squared residuals/sigma^2): ", E)
        #covari1 = res_in_vec.dot(H)
        #covariance_C =  2/(nt-2)*E*np.linalg.inv(H)
        #covariance_C = np.linalg.inv(H)
        
        covariance_C = H
        #print(covariance_C,"covariance matrix")
        variances_p = np.diag(covariance_C)
        print("Parameter variances: ", variances_p)
        print('\nConfidence intervals:')
        i = 0
        for k, p in self.model.P.items():
            if p.is_fixed():
                continue
            print('{} ({},{})'.format(k, p.value - variances_p[i] ** 0.5, p.value + variances_p[i] ** 0.5))
            i = +1


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
        nc = self._n_actual
        nparams = 0
        for v in six.itervalues(self.model.P):
            if v.is_fixed():  #: Skip the fixed parameters
                continue
            nparams += 1

        #nparams = len(self.model.P)
        # this changes depending on the order of the suffixes passed to sipopt
        nd = nw * nt
        ntheta = nc * (nw + nt) + nparams
        self.B_matrix = np.zeros((ntheta, nw * nt))
        for i, t in enumerate(self.model.meas_times):
            for j, l in enumerate(self.model.meas_lambdas):
                for k, c in enumerate(self._sublist_components):
                    # r_idx1 = k*nt+i
                    r_idx1 = i * nc + k
                    r_idx2 = j * nc + k + nc * nt
                    # r_idx2 = j * nc + k + nc * nw
                    # c_idx = i+j*nt
                    c_idx = i * nw + j
                    # print(j, k, r_idx2)
                    self.B_matrix[r_idx1, c_idx] = -2 * self.model.S[l, c].value / variances['device']
                    # try:
                    self.B_matrix[r_idx2, c_idx] = -2 * self.model.C[t, c].value / variances['device']
                    # except IndexError:
                    #     pass
        # sys.exit()

    def _compute_Vd_matrix(self, variances, **kwds):
        """Builds d covariance matrix

           This method is not intended to be used by users directly

        Args:
            variances (dict): variances 
            
        Returns:
            None
        """

        # add check for model already solved
        row = []
        col = []
        data = []
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nc = self._n_actual
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
        s_array = np.zeros(nw * nc)
        v_array = np.zeros(nc)
        for k, c in enumerate(self._sublist_components):
            v_array[k] = variances[c]

        for j, l in enumerate(self.model.meas_lambdas):
            for k, c in enumerate(self._sublist_components):
                s_array[j * nc + k] = self.model.S[l, c].value

        row = []
        col = []
        data = []
        nd = nt * nw
        # Vd_dense = np.zeros((nd,nd))
        v_device = variances['device']
        for i in range(nt):
            for j in range(nw):
                val = sum(v_array[k] * s_array[j * nc + k] ** 2 for k in range(nc)) + v_device
                row.append(i * nw + j)
                col.append(i * nw + j)
                data.append(val)
                # Vd_dense[i*nw+j,i*nw+j] = val
                for p in range(nw):
                    if j != p:
                        val = sum(v_array[k] * s_array[j * nc + k] * s_array[p * nc + k] for k in range(nc))
                        row.append(i * nw + j)
                        col.append(i * nw + p)
                        data.append(val)
                        # Vd_dense[i*nw+j,i*nw+p] = val

        self.Vd_matrix = scipy.sparse.coo_matrix((data, (row, col)),
                                                 shape=(nd, nd)).tocsr()
        # self.Vd_matrix = Vd_dense

    def _compute_residuals(self):
        """
        Computes the square of residuals between the optimal solution (Z) and the concentration data (C)
        Note that this returns a matrix of time points X components and it has not been divided by sigma^2
        
        This method is not intended to be used by users directly
        """
        nt = self._n_meas_times
        nc = self._n_actual
        self.residuals = dict()
        count_c = 0
        for c in self._sublist_components:
            count_t = 0
            for t in self._meas_times:
                a = self.model.C[t, c].value
                b = self.model.Z[t, c].value
                r = ((a-b)**2)
                self.residuals[t,c]=r
                count_t += 1
            count_c += 1
            
    def _order_k_aug_hessian(self, unordered_hessian, var_loc):
        """
        not meant to be used directly by users. Takes in the inverse of the reduced hessian
        outputted by k_aug and uses the rh_name to find the locations of the variables and then 
        re-orders the hessian to be in a format where the other functions are able to compute the
        confidence intervals in a way similar to that utilized by sIpopt.
        """
        vlocsize = len(var_loc)
        n_vars = len(self._idx_to_variable)
        hessian = np.zeros((n_vars,n_vars))
        i = 0
        for vi in six.itervalues(self._idx_to_variable):
            j = 0
            for vj in six.itervalues(self._idx_to_variable):
                h = unordered_hessian[(var_loc[vi]),(var_loc[vj])]
                hessian[i,j] = h
                j+=1
            i+=1                        
        print(hessian.size, "hessian size")
        return hessian 
      
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

        solver_opts = kwds.pop('solver_opts', dict())
        variances = kwds.pop('variances', dict())
        tee = kwds.pop('tee', False)
        with_d_vars = kwds.pop('with_d_vars', False)
        covariance = kwds.pop('covariance', False)
        self.solver = solver
        if not self.model.time.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')

        # Look at the output in results
        opt = SolverFactory(self.solver)

        if covariance:
            if self.solver != 'ipopt_sens' and self.solver != 'k_aug':
                raise RuntimeError('To get covariance matrix the solver needs to be ipopt_sens or k_aug')
            if self.solver == 'ipopt_sens':
                if not 'compute_red_hessian' in solver_opts.keys():
                    solver_opts['compute_red_hessian'] = 'yes'
            if self.solver == 'k_aug':
                solver_opts['compute_inv']=''
            self._define_reduce_hess_order()

        for key, val in solver_opts.items():
            opt.options[key] = val

        active_objectives = [o for o in self.model.component_map(Objective, active=True)]
        if active_objectives:
            print(
                "WARNING: The model has an active objective. Running optimization with models objective.\n"
                " To solve optimization with default objective (Weifengs) deactivate all objectives in the model.")
            solver_results = opt.solve(self.model, tee=tee)
        elif self._spectra_given:
            self._solve_extended_model(variances, opt,
                                       tee=tee,
                                       covariance=covariance,
                                       with_d_vars=with_d_vars,
                                       **kwds)
        elif self._concentration_given:
            print("This prints before the solve model given c")
            self._solve_model_given_c(variances, opt,
                                      tee=tee,
                                      covariance=covariance,
                                      **kwds)
        else:
            raise RuntimeError('Must either provide concentration data or spectra in order to solve the parameter estimation problem')
        
        results = ResultsObject()
        
        if self._spectra_given:
            results.load_from_pyomo_model(self.model,
                                      to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y'])
        elif self._concentration_given:
            results.load_from_pyomo_model(self.model,
                                      to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'Y'])
        else:
            raise RuntimeError('Must either provide concentration data or spectra in order to solve the parameter estimation problem')
           
        if self._spectra_given:
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
    #print(hess_string, ipopt_string)
    return (ipopt_string, hess_string)

def split_k_aug_string(output_string):
    start_hess = output_string.find('')
    ipopt_string = output_string[:start_hess]
    hess_string = output_string[start_hess:]
    #print(hess_string, ipopt_string)
    return (ipopt_string, hess_string)

def read_reduce_hessian2(hessian_string, n_vars):
    hessian_string = re.sub('RedHessian unscaled\[', '', hessian_string)
    hessian_string = re.sub('\]=', ',', hessian_string)

    hessian = np.zeros((n_vars, n_vars))
    for i, line in enumerate(hessian_string.split('\n')):
        if i > 0:
            hess_line = line.split(',')
            if len(hess_line) == 3:
                row = int(hess_line[0])
                col = int(hess_line[1])
                hessian[row, col] = float(hess_line[2])
                hessian[col, row] = float(hess_line[2])
    return hessian
    
def read_reduce_hessian(hessian_string, n_vars):
    hessian = np.zeros((n_vars, n_vars))
    for i, line in enumerate(hessian_string.split('\n')):
        if i > 0:  # ignores header
            if line not in ['', ' ', '\t']:
                hess_line = line.split(']=')
                if len(hess_line) == 2:
                    value = float(hess_line[1])
                    column_line = hess_line[0].split(',')
                    col = int(column_line[1])
                    row_line = column_line[0].split('[')
                    row = int(row_line[1])
                    hessian[row, col] = float(value)
                    hessian[col, row] = float(value)
    return hessian

def read_reduce_hessian_k_aug(hessian_string, n_vars):
    hessian = np.zeros((n_vars, n_vars))
    for i, line in enumerate(hessian_string.split('\n')):
        if i > 0:  # ignores header
            if line not in ['', ' ', '\t']:
                hess_line = line.split(']=')
                if len(hess_line) == 2:
                    value = float(hess_line[1])
                    column_line = hess_line[0].split(',')
                    col = int(column_line[1])
                    row_line = column_line[0].split('[')
                    row = int(row_line[1])
                    hessian[row, col] = float(value)
                    hessian[col, row] = float(value)
    return hessian
