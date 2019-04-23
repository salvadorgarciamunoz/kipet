# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
from kipet.library.Optimizer import *
from kipet.library.TemplateBuilder import *
import matplotlib.pyplot as plt
from pyomo import *
import numpy as np
import six
import copy
import re
import os
import time
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import NumericConstant

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
        self.hessian = None
        self._estimability = False
        self._idx_to_variable = dict()
        self._n_actual = self._n_components
        
        if hasattr(self.model, 'non_absorbing'):
            warnings.warn("Overriden by non_absorbing")
            list_components = [k for k in self._mixture_components if k not in self._non_absorbing]
            self._sublist_components = list_components
            self._n_actual = len(self._sublist_components)
            
        if hasattr(self.model, 'known_absorbance'):
            warnings.warn("Overriden by known_absorbance")
            list_components = [k for k in self._mixture_components if k not in self._known_absorbance]
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
            
            subset_lambdas (array_like,optional): Set of wavelengths to used in 
            the optimization problem (not yet fully implemented). Default all wavelengths.

        Returns:
            None
        """

        tee = kwds.pop('tee', False)
        with_d_vars = kwds.pop('with_d_vars', False)
        weights = kwds.pop('weights', [1.0, 1.0])
        warmstart = kwds.pop('warmstart', False)
        covariance = kwds.pop('covariance', False)
        species_list = kwds.pop('subset_components', None)
        set_A = kwds.pop('subset_lambdas', list())
        
        if not set_A:
            set_A = self._meas_lambdas
        
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
            warnings.warn("Overriden by non_absorbing!")
            list_components = [k for k in self._mixture_components if k not in self._non_absorbing]
        
        if hasattr(self.model, 'known_absorbance'):
            warnings.warn("Overriden by known_absorbance!")
            list_components = [k for k in self._mixture_components if k not in self._known_absorbance]

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
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
            if hasattr(self, '_abs_components'):
                def rule_D_bar(m, t, l):
                    return m.D_bar[t, l] == sum(m.Cs[t, k] * m.S[l, k] for k in self._abs_components)
            else:
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
                        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
                        if hasattr(self, '_abs_components'):
                            D_bar = sum(m.Cs[t, k] * m.S[l, k] for k in self._abs_components)
                            expr += (m.D[t, l] - D_bar) ** 2 / (sigma_sq['device'])
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

        if covariance and self.solver == 'ipopt_sens':
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
            #print('n_vars', n_vars)
            hessian = read_reduce_hessian(hessian_output, n_vars)
            print(hessian.size, "hessian size")
            # print(hessian.shape,"hessian shape")
            # hessian = read_reduce_hessian2(hessian_output,n_vars)
            # print hessian
            self._compute_covariance(hessian, sigma_sq)

        if warmstart==True:
            if hasattr(m,'dual') and hasattr(m,'ipopt_zL_out') and hasattr(m,'ipopt_zU_out') and hasattr(m,'ipopt_zL_in') and hasattr(m,'ipopt_zU_in'):
                m.ipopt_zL_in.update(m.ipopt_zL_out)  #: be sure that the multipliers got updated!
                m.ipopt_zU_in.update(m.ipopt_zU_out)
            else:
                m.dual= Suffix(direction=Suffix.IMPORT_EXPORT)
                m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
                m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
                m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
                m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

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
                if hasattr(self, '_abs_components'):
                    for t in self._meas_times:
                        for c in self._abs_components:
                            m.Cs[t, c].set_suffix_value(m.dof_v, count_vars)

                            count_vars += 1
                else:
                    for t in self._meas_times:
                        for c in self._sublist_components:
                            m.C[t, c].set_suffix_value(m.dof_v, count_vars)

                            count_vars += 1

            if not self._spectra_given:
                pass
            else:
                if hasattr(self, '_abs_components'):
                    for l in self._meas_lambdas:
                        for c in self._abs_components:
                            m.S[l, c].set_suffix_value(m.dof_v, count_vars)
                            count_vars += 1
                else:
                    for l in self._meas_lambdas:
                        for c in self._sublist_components:
                            m.S[l, c].set_suffix_value(m.dof_v, count_vars)
                            count_vars += 1

            for v in six.itervalues(self.model.P):
                if v.is_fixed():
                    continue
                m.P.set_suffix_value(m.dof_v, count_vars)
                count_vars += 1

            self._tmpfile = "k_aug_hess"
            ip = SolverFactory('ipopt')
            solver_results = ip.solve(m, tee=False,
                                      logfile=self._tmpfile,
                                      report_timing=True)
            k_aug = SolverFactory('k_aug')
            # k_aug.options["compute_inv"] = ""
            m.ipopt_zL_in.update(m.ipopt_zL_out)  #: be sure that the multipliers got updated!
            m.ipopt_zU_in.update(m.ipopt_zU_out)
            # m.write(filename="mynl.nl", format=ProblemFormat.nl)
            k_aug.solve(m, tee=False)
            print("Done solving building reduce hessian")

            if not all_sigma_specified:
                raise RuntimeError(
                    'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            print("n_vars", n_vars)
            # m.rh_name.pprint()
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
            # hessian = read_reduce_hessian_k_aug(hessian_output, n_vars)
            # hessian =hessian_output
            # print(hessian)
            #print(unordered_hessian.size, "unordered hessian size")
            hessian = self._order_k_aug_hessian(unordered_hessian, var_loc)
            if self._estimability == True:
                self.hessian = hessian
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
        warmstart = kwds.pop('warmstart', False)
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
            raise NotImplementedError(
                "Parameter Estimation from concentration data requires concentration data model.C[ti,cj]")

        # if hasattr(self.model, 'non_absorbing'):
        #    warnings.warn("Overriden by non_absorbing!!!")
        #    list_components = [k for k in self._mixture_components if k not in self._non_absorbing]

        all_sigma_specified = True
        #print(sigma_sq)
        keys = sigma_sq.keys()
        for k in list_components:
            if k not in keys:
                all_sigma_specified = False
                sigma_sq[k] = max(sigma_sq.values())

        # if not 'device' in sigma_sq.keys():
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
        if warmstart==True:
            if hasattr(m,'dual') and hasattr(m,'ipopt_zL_out') and hasattr(m,'ipopt_zU_out') and hasattr(m,'ipopt_zL_in') and hasattr(m,'ipopt_zU_in'):
                m.ipopt_zL_in.update(m.ipopt_zL_out)  #: be sure that the multipliers got updated!
                m.ipopt_zU_in.update(m.ipopt_zU_out)
            else:
                m.dual= Suffix(direction=Suffix.IMPORT_EXPORT)
                m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
                m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
                m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
                m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

        if covariance and self.solver == 'ipopt_sens':
            self._tmpfile = "ipopt_hess"
            solver_results = optimizer.solve(m, tee=False,
                                             logfile=self._tmpfile,
                                             report_timing=True)
            # self.model.red_hessian.pprint
            m.P.pprint()
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
            # if tee == True:
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
            # else:
            #    self._compute_covariance(hessian, sigma_sq)

        elif covariance and self.solver == 'k_aug':
            m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

            m.dof_v = Suffix(direction=Suffix.EXPORT)  #: SUFFIX FOR K_AUG
            m.rh_name = Suffix(direction=Suffix.IMPORT)  #: SUFFIX FOR K_AUG AS WELL
            m.dof_v.pprint()
            m.rh_name.pprint()
            count_vars = 1

            if not self._spectra_given:
                pass
            else:
                for t in self._meas_times:
                    for c in self._sublist_components:
                        m.C[t, c].set_suffix_value(m.dof_v, count_vars)

                        count_vars += 1

            if not self._spectra_given:
                pass
            else:
                for l in self._meas_lambdas:
                    for c in self._sublist_components:
                        m.S[l, c].set_suffix_value(m.dof_v, count_vars)
                        count_vars += 1

            # if self._concentration_given:
            #    for t in self._meas_times:
            #        for c in self._sublist_components:
            #            m.Z[t, c].set_suffix_value(m.dof_v,count_vars)

            #            count_vars += 1

            for v in six.itervalues(self.model.P):
                if v.is_fixed():
                    continue
                m.P.set_suffix_value(m.dof_v, count_vars)
                count_vars += 1

            self._tmpfile = "k_aug_hess"
            ip = SolverFactory('ipopt')
            solver_results = ip.solve(m, tee=False,
                                      logfile=self._tmpfile,
                                      report_timing=True)
            # m.P.pprint()
            k_aug = SolverFactory('k_aug')

            # k_aug.options["no_scale"] = ""
            m.ipopt_zL_in.update(m.ipopt_zL_out)  #: be sure that the multipliers got updated!
            m.ipopt_zU_in.update(m.ipopt_zU_out)
            # m.write(filename="mynl.nl", format=ProblemFormat.nl)
            k_aug.solve(m, tee=True)
            print("Done solving building reduce hessian")

            if not all_sigma_specified:
                raise RuntimeError(
                    'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            print("n_vars", n_vars)
            m.rh_name.pprint()
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
            print("var_loc size, ", vlocsize)
            unordered_hessian = np.loadtxt('result_red_hess.txt')
            if os.path.exists('result_red_hess.txt'):
                os.remove('result_red_hess.txt')
            # hessian = read_reduce_hessian_k_aug(hessian_output, n_vars)
            # hessian =hessian_output
            # print(hessian)
            print(unordered_hessian.size, "unordered hessian size")
            hessian = self._order_k_aug_hessian(unordered_hessian, var_loc)
            if self._estimability == True:
                self.hessian = hessian
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
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
            if hasattr(self, '_abs_components'):  # added for removing non absorbing ones from first term in obj
                for t in self._meas_times:
                    for c in self._abs_components:
                        v = self.model.Cs[t, c]
                        self._idx_to_variable[count_vars] = v
                        self.model.red_hessian[v] = count_vars
                        count_vars += 1
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
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
            if hasattr(self,'_abs_components'): #added for removing non absorbing ones from first term in obj
                for l in self._meas_lambdas:
                    for c in self._abs_components:
                        v = self.model.S[l, c]
                        self._idx_to_variable[count_vars] = v
                        self.model.red_hessian[v] = count_vars
                        count_vars += 1
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
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self, '_abs_components'):
            nabs=self._nabs_components #number of absorbing components (CS)
            nparams = 0
            for v in six.itervalues(self.model.P):
                if v.is_fixed():  #: Skip the fixed ones
                    print(str(v) + '\has been skipped for covariance calculations')
                    continue
                nparams += 1
            # nparams = len(self.model.P)
            nd = nw * nt
            ntheta = nabs * (nw + nt) + nparams

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
            nabs = self._nabs_components # #number of absorbing components (CS)
            nparams = 0
            for v in six.itervalues(self.model.P):
                if v.is_fixed():  #: Skip the fixed ones ;)
                    continue
                nparams += 1

            # this changes depending on the order of the suffixes passed to sipopt
            nd = nw * nt
            ntheta = nabs * (nw + nt)
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
        else:
            nc = self._n_actual
            print(nc)
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
        # print(res)
        # sets up matrix with variances in diagonals
        nc = self._n_actual
        nt = self._n_meas_times
        varmat = np.zeros((nc, nc))
        for c, k in enumerate(self._sublist_components):
            varmat[c, c] = variances[k]
        # print("varmat",varmat)
        # R=varmat.dot(res)
        # L = res.dot(varmat)
        E = 0
        for t in self._meas_times:
            for k in self._sublist_components:
                E += res[t, k] / (variances[k] ** 2)

        # Now we can use the E matrix with the hessian to estimate our confidence intervals
        nparams = 0
        for v in six.itervalues(self.model.P):
            if v.is_fixed():  #: Skip the fixed ones
                print(str(v) + '\has been skipped for covariance calculations')
                continue
            nparams += 1
        all_H = hessian
        H = all_H[-nparams:, :]

        # print(E_matrix)
        # covariance_C = E_matrix.dot(H.T)

        # print("value of the objective function (sum of squared residuals/sigma^2): ", E)
        # covari1 = res_in_vec.dot(H)
        # covariance_C =  2/(nt-2)*E*np.linalg.inv(H)
        # covariance_C = np.linalg.inv(H)

        covariance_C = H
        # print(covariance_C,"covariance matrix")
        variances_p = np.diag(covariance_C)
        print("Parameter variances: ", variances_p)
        print('\nConfidence intervals:')
        i = 0
        for k, p in self.model.P.items():
            if p.is_fixed():
                continue
            print('{} ({},{})'.format(k, p.value - variances_p[i] ** 0.5, p.value + variances_p[i] ** 0.5))
            i += 1

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

        nparams = 0
        for v in six.itervalues(self.model.P):
            if v.is_fixed():  #: Skip the fixed parameters
                continue
            nparams += 1

        # nparams = len(self.model.P)
        # this changes depending on the order of the suffixes passed to sipopt
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self,'_abs_components'):
            nabs=self._nabs_components
            nd = nw * nt
            ntheta = nabs * (nw + nt) + nparams
            self.B_matrix = np.zeros((ntheta, nw * nt))
            for i, t in enumerate(self.model.meas_times):
                for j, l in enumerate(self.model.meas_lambdas):
                    for k, c in enumerate(self._abs_components):
                        # r_idx1 = k*nt+i
                        r_idx1 = i * nabs + k
                        r_idx2 = j * nabs + k + nabs * nt
                        # r_idx2 = j * nc + k + nc * nw
                        # c_idx = i+j*nt
                        c_idx = i * nw + j
                        # print(j, k, r_idx2)
                        self.B_matrix[r_idx1, c_idx] = -2 * self.model.S[l, c].value / variances['device']
                        # try:
                        self.B_matrix[r_idx2, c_idx] = -2 * self.model.C[t, c].value / variances['device']
        else:
            nc = self._n_actual
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
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self,'_abs_components'):
            nabs=self._nabs_components
            s_array = np.zeros(nw * nabs)
            v_array = np.zeros(nabs)
            for k, c in enumerate(self._abs_components):
                v_array[k] = variances[c]

            for j, l in enumerate(self.model.meas_lambdas):
                for k, c in enumerate(self._abs_components):
                    s_array[j * nabs + k] = self.model.S[l, c].value

            row = []
            col = []
            data = []
            nd = nt * nw
            # Vd_dense = np.zeros((nd,nd))
            v_device = variances['device']
            for i in range(nt):
                for j in range(nw):
                    val = sum(v_array[k] * s_array[j * nabs + k] ** 2 for k in range(nabs)) + v_device
                    row.append(i * nw + j)
                    col.append(i * nw + j)
                    data.append(val)
                    # Vd_dense[i*nw+j,i*nw+j] = val
                    for p in range(nw):
                        if j != p:
                            val = sum(v_array[k] * s_array[j * nabs + k] * s_array[p * nabs + k] for k in range(nabs))
                            row.append(i * nw + j)
                            col.append(i * nw + p)
                            data.append(val)
            self.Vd_matrix = scipy.sparse.coo_matrix((data, (row, col)),
                                                     shape=(nd, nd)).tocsr()
        else:
            nc = self._n_actual
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
                r = ((a - b) ** 2)
                self.residuals[t, c] = r
                count_t += 1
            count_c += 1

######################### Added for using inputs model for parameter estimation, CS##########
    def load_discrete_jump(self):
        self.jump = True

        zeit = None
        for i in self.model.component_objects(ContinuousSet):
            zeit = i
            break
        if zeit is None:
            raise Exception('no continuous_set')
        self.time_set = zeit.name

        tgt_cts = getattr(self.model, self.time_set)  ## please correct me (not necessary!)
        self.ncp = tgt_cts.get_discretization_info()['ncp']
        fe_l = tgt_cts.get_finite_elements()
        fe_list = [fe_l[i + 1] - fe_l[i] for i in range(0, len(fe_l) - 1)]

        for i in range(0, len(fe_list)):  # test whether integer elements
            self.jump_constraints(i)

    ###########################
    def jump_constraints(self, fe):
        # type: (int) -> None
        """ Take the current state of variables of the initializing model at fe and load it into the tgt_model
        Note that this will skip fixed variables as a safeguard.

        Args:
            fe (int): The current finite element to be patched (tgt_model).
        """
        ###########################
        if not isinstance(fe, int):
            raise Exception  # wrong type
        ttgt = getattr(self.model, self.time_set)
        ##############################
        # Inclusion of discrete jumps: (CS)
        if self.jump:
            vs = ReplacementVisitor()  #: trick to replace variables
            kn = 0
            for ki in self.jump_times_dict.keys():
                if not isinstance(ki, str):
                    print("ki is not str")
                vtjumpkeydict = self.jump_times_dict[ki]
                for l in vtjumpkeydict.keys():
                    self.jump_time = vtjumpkeydict[l]
                    # print('jumptime:',self.jump_time)
                    self.jump_fe, self.jump_cp = fe_cp(ttgt, self.jump_time)
                    if self.jump_time not in self.feed_times_set:
                        raise Exception("Error: Check feed time points in set feed_times and in jump_times again.\n"
                                        "They do not match.\n"
                                        "Jump_time is not included in feed_times.")
                    # print('jump_el, el:',self.jump_fe, fe)
                    if fe == self.jump_fe + 1:
                        # print("jump_constraints!")
                        #################################
                        for v in self.disc_jump_v_dict.keys():
                            if not isinstance(v, str):
                                print("v is not str")
                            vkeydict = self.disc_jump_v_dict[v]
                            for k in vkeydict.keys():
                                if k == l:  # Match in between two components of dictionaries
                                    var = getattr(self.model, v)
                                    dvar = getattr(self.model, "d" + v + "dt")
                                    con_name = 'd' + v + 'dt_disc_eq'
                                    con = getattr(self.model, con_name)

                                    self.model.add_component(v + "_dummy_eq_" + str(kn), ConstraintList())
                                    conlist = getattr(self.model, v + "_dummy_eq_" + str(kn))
                                    varname = v + "_dummy_" + str(kn)
                                    self.model.add_component(varname, Var([0]))  #: this is now indexed [0]
                                    vdummy = getattr(self.model, varname)
                                    vs.change_replacement(vdummy[0])   #: who is replacing.
                                    # self.model.add_component(varname, Var())
                                    # vdummy = getattr(self.model, varname)
                                    jump_delta = vkeydict[k]
                                    self.model.add_component(v + '_jumpdelta' + str(kn),
                                                             Param(initialize=jump_delta))
                                    jump_param = getattr(self.model, v + '_jumpdelta' + str(kn))
                                    if not isinstance(k, tuple):
                                        k = (k,)
                                    exprjump = vdummy[0] - var[(self.jump_time,) + k] == jump_param  #: this cha
                                    # exprjump = vdummy - var[(self.jump_time,) + k] == jump_param
                                    self.model.add_component("jumpdelta_expr" + str(kn), Constraint(expr=exprjump))
                                    for kcp in range(1, self.ncp + 1):
                                        curr_time = t_ij(ttgt, self.jump_fe + 1, kcp)
                                        if not isinstance(k, tuple):
                                            knew = (k,)
                                        else:
                                            knew = k
                                        idx = (curr_time,) + knew
                                        con[idx].deactivate()
                                        e = con[idx].expr
                                        suspect_var = e.args[0].args[1].args[0].args[0].args[1]  #: seems that
                                        # e = con[idx].expr.clone()
                                        # e.args[0].args[1] = vdummy
                                        # con[idx].set_value(e)
                                        vs.change_suspect(id(suspect_var))  #: who to replace
                                        e_new = vs.dfs_postorder_stack(e)  #: replace
                                        con[idx].set_value(e_new)
                                        conlist.add(con[idx].expr)
                    kn = kn + 1

####################################################################

    def _order_k_aug_hessian(self, unordered_hessian, var_loc):
        """
        not meant to be used directly by users. Takes in the inverse of the reduced hessian
        outputted by k_aug and uses the rh_name to find the locations of the variables and then
        re-orders the hessian to be in a format where the other functions are able to compute the
        confidence intervals in a way similar to that utilized by sIpopt.
        """
        vlocsize = len(var_loc)
        n_vars = len(self._idx_to_variable)
        hessian = np.zeros((n_vars, n_vars))
        i = 0
        for vi in six.itervalues(self._idx_to_variable):
            j = 0
            for vj in six.itervalues(self._idx_to_variable):
                if n_vars ==1:
                    print("var_loc[vi]",var_loc[vi])
                    print(unordered_hessian)
                    h = unordered_hessian
                    hessian[i, j] = h
                else:
                    h = unordered_hessian[(var_loc[vi]), (var_loc[vj])]
                    hessian[i, j] = h
                j += 1
            i += 1
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
            
            report_time (bool, optional): flag as to whether to time the parameter estimation or not

            estimability (bool, optional): flag to tell the model whether it is
            being used by the estimability analysis and therefore will need to return the
            hessian for analysis.

        Returns:
            Results object with loaded results

        """

        solver_opts = kwds.pop('solver_opts', dict())
        variances = kwds.pop('variances', dict())
        tee = kwds.pop('tee', False)
        with_d_vars = kwds.pop('with_d_vars', False)
        covariance = kwds.pop('covariance', False)

        estimability = kwds.pop('estimability', False)
        report_time = kwds.pop('report_time', False)
        
        #additional arguments for inputs CS
        inputs = kwds.pop("inputs", None)
        inputs_sub = kwds.pop("inputs_sub", None)
        trajectories = kwds.pop("trajectories", None)
        fixedtraj = kwds.pop("fixedtraj", False)
        fixedy = kwds.pop("fixedy", False)
        yfix = kwds.pop("yfix", None)
        yfixtraj = kwds.pop("yfixtraj", None)

        jump = kwds.pop("jump", False)
        var_dic = kwds.pop("jump_states", None)
        jump_times = kwds.pop("jump_times", None)
        feed_times = kwds.pop("feed_times", None)

        self.solver = solver
        
        if not self.model.time.get_discretization_info():
            raise RuntimeError('apply discretization first before initializing')
        
        if report_time:
            start = time.time()
        # Look at the output in results
        opt = SolverFactory(self.solver)

        if covariance:
            if self.solver != 'ipopt_sens' and self.solver != 'k_aug':
                raise RuntimeError('To get covariance matrix the solver needs to be ipopt_sens or k_aug')
            if self.solver == 'ipopt_sens':
                if not 'compute_red_hessian' in solver_opts.keys():
                    solver_opts['compute_red_hessian'] = 'yes'
            if self.solver == 'k_aug':
                solver_opts['compute_inv'] = ''

            self._define_reduce_hess_order()

        for key, val in solver_opts.items():
            opt.options[key] = val

        if estimability == True:
            self._estimability = True
            # solver_opts['dsdp_mode'] = ""
        active_objectives = [o for o in self.model.component_map(Objective, active=True)]

        """inputs section""" # additional section for inputs from trajectory and fixed inputs, CS
        self.fixedtraj = fixedtraj
        self.fixedy = fixedy
        self.inputs_sub = inputs_sub
        self.yfix = yfix
        self.yfixtraj = yfixtraj
        if self.inputs_sub!=None:
            for k in self.inputs_sub.keys():
                if not isinstance(self.inputs_sub[k], list):
                    print("wrong type for inputs_sub {}".format(type(self.inputs_sub[k])))
                    # raise Exception
                for i in self.inputs_sub[k]:
                    # print(self.inputs_sub[k])
                    # print(i)
                    if self.fixedtraj==True or self.fixedy==True:
                        if self.fixedtraj==True:
                            for j in self.yfixtraj.keys():
                                for l in self.yfixtraj[j]:
                                    if i==l:
                                        # print('herel:fixedy', l)
                                        if not isinstance(self.yfixtraj[j], list):
                                            print("wrong type for yfixtraj {}".format(type(self.yfixtraj[j])))
                                        reft = trajectories[(k, i)]
                                        self.fix_from_trajectory(k, i, reft)
                        if self.fixedy==True:
                            for j in self.yfix.keys():
                                for l in self.yfix[j]:
                                    if i==l:
                                        # print('herel:fixedy',l)
                                        if not isinstance(self.yfix[j], list):
                                            print("wrong type for yfix {}".format(type(self.yfix[j])))
                                        for key in self.model.time.value:
                                            vark=getattr(self.model,k)
                                            vark[key, i].set_value(key)
                                            vark[key, i].fix()# since these are inputs we need to fix this
                    else:
                        print("A trajectory or fixed input is missing for {}\n".format((k, i)))
        """/end inputs section"""

        if jump:
            self.disc_jump_v_dict = var_dic
            self.jump_times_dict = jump_times  # now dictionary
            self.feed_times_set = feed_times
            if not isinstance(self.disc_jump_v_dict, dict):
                print("disc_jump_v_dict is of type {}".format(type(self.disc_jump_v_dict)))
                raise Exception  # wrong type
            if not isinstance(self.jump_times_dict, dict):
                print("disc_jump_times is of type {}".format(type(self.jump_times_dict)))
                raise Exception  # wrong type
            count = 0
            for i in six.iterkeys(self.jump_times_dict):
                for j in six.iteritems(self.jump_times_dict[i]):
                    count += 1
            if len(self.feed_times_set) > count:
                raise Exception("Error: Check feed time points in set feed_times and in jump_times again.\n"
                            "There are more time points in feed_times than jump_times provided.")
            self.load_discrete_jump()

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
            self._solve_model_given_c(variances, opt,
                                      tee=tee,
                                      covariance=covariance,
                                      **kwds)
        else:
            raise RuntimeError(
                'Must either provide concentration data or spectra in order to solve the parameter estimation problem')

        results = ResultsObject()

        if self._spectra_given:
            results.load_from_pyomo_model(self.model,
                                          to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y'])
            # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
            if hasattr(self, '_abs_components'):
                results.load_from_pyomo_model(self.model,
                                              to_load=['Cs'])
                # results.load_from_pyomo_model(self.model,
                #                               to_load=['Ss'])
        elif self._concentration_given:
            results.load_from_pyomo_model(self.model,
                                          to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'Y'])
        else:
            raise RuntimeError(
                'Must either provide concentration data or spectra in order to solve the parameter estimatiD-on problem')

        if self._spectra_given:
            self.compute_D_given_SC(results)

        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name].value

        results.P = param_vals

        if report_time:
            end = time.time()
            print("Total execution time in seconds for variance estimation:", end - start)

        if self._estimability == True:
            return self.hessian, results
        else:
            return results        
        
    def run_param_est_with_subset_lambdas(self, builder_clone, end_time, subset, nfe, ncp, sigmas):
        """ Performs the parameter estimation with a specific subset of wavelengths.
            At the moment, this is performed as a totally new Pyomo model, based on the 
            original estimation. Initialization strategies for this will be needed.
                        
                Args:
                    builder_clone (TemplateBuidler): Template builder class of complete model
                                without the data added yet
                    end_time (float): the end time for the data and simulation
                    subset(list): list of selected wavelengths
                    nfe (int): number of finite elements
                    ncp (int): number of collocation points
                    sigmas(dict): dictionary containing the variances, as used in the ParameterEstimator class
            
                Returns:
                    results (Pyomo model solved): The solved pyomo model
            
        """ 
        if not isinstance(subset, (list, dict)):
            raise RuntimeError("subset must be of type list or dict!")
             
        if isinstance(subset, dict):
            lists1 = sorted(subset.items())
            x1, y1 = zip(*lists1)
            subset = list(x1)
        #should check whether the list contains wavelengths or not 
        
        #This is the filter for creating the new data subset
        new_D = pd.DataFrame(np.nan,index=self._meas_times, columns = subset)
        for t in self._meas_times:
            for l in self._meas_lambdas:
                if l in subset:
                    new_D.at[t,l] = self.model.D[t,l]
        #print(new_D)   
        #Now that we have a new DataFrame, we need to build the entire problem from this
        #An entire new ParameterEstimation problem should be set up, on the outside of 
        #this function and class structure, from the model already developed by the user. 
        new_template = construct_model_from_reduced_set(builder_clone,end_time, new_D)
        #need to put in an optional running of the variance estimator for the new 
        #parameter estiamtion run, or just use the previous full model run to initialize... 
            
        results, lof = run_param_est(new_template, nfe, ncp, sigmas) 
        
        return results
        
    def run_lof_analysis(self, builder_before_data, end_time, correlations, lof_full_model, nfe, ncp, sigmas, step_size = 0.2, search_range = (0, 1)):
        """ Runs the lack of fit minimization problem used in the Michael's Reaction paper
        from Chen et al. (submitted). To use this function, the full parameter estimation
        problem should be solved first and the correlations for wavelngths from this optimization
        need to be supplied to the function as an option.
                        
                Args:
                    builder_before_data (TemplateBuilder): Template builder class of complete model
                                without the data added yet
                    end_time (int): the end time for the data and simulation

                    correlations (dict): dictionary containing the wavelengths and their correlations
                                to the concentration profiles
                    lof_full_model(int): the value of the lack of fit of the full model (with all wavelengths)
            
                Returns:
                    *****final model results.
            
        """ 
        if not isinstance(step_size, float):
            raise RuntimeError("step_size must be a float between 0 and 1")
        elif step_size >= 1 or step_size <= 0:
            return RuntimeError("step_size must be a float between 0 and 1")
        
        if not isinstance(search_range, tuple):
            raise RuntimeError("search range must be a tuple")
        elif search_range [0] < 0 or search_range [0] > 1 and not (isinstance(search_range, float) or isinstance(search_range, int)):
            raise RuntimeError("search range lower value must be between 0 and 1 and must be type float")
        elif search_range [1] < 0 or search_range [1] > 1 and not  (isinstance(search_range, float) or isinstance(search_range, int)):
            raise RuntimeError("search range upper value must be between 0 and 1 and must be type float")
        elif search_range [1] <= search_range [0]:
            raise RuntimeError("search_range[1] must be bigger than search_range[0]!")
        #firstly we will run the initial search from at increments of 20 % for the correlations
        # we already have lof(0) so we want 10,30,50,70, 90.
        count = 0
        filt = 0.0
        initial_solutions = list()
        initial_solutions.append((0, lof_full_model))
        while filt < search_range[1]:
            filt += step_size
            if filt > search_range[1]:
                break
            elif filt == 1:
                break
            new_subs = wavelength_subset_selection(correlations = correlations, n = filt)
            lists1 = sorted(new_subs.items())
            x1, y1 = zip(*lists1)
            x = list(x1)            
            
            new_D = pd.DataFrame(np.nan,index=self._meas_times, columns = new_subs)
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    if l in new_subs:
                        new_D.at[t,l] = self.model.D[t,l]
            
            #opt_model, nfe, ncp = construct_model_from_reduced_set(builder_before_data, end_time, new_D)
            # Now that we have a new DataFrame, we need to build the entire problem from this
            # An entire new ParameterEstimation problem should be set up, on the outside of 
            # this function and class structure, from the model already developed by the user. 
            new_template = construct_model_from_reduced_set(builder_before_data,end_time, new_D)
            # need to put in an optional running of the variance estimator for the new 
            # parameter estimation run, or just use the previous full model run to initialize...             
            results, lof = run_param_est(new_template, nfe, ncp, sigmas) 
            initial_solutions.append((filt, lof))
            
            count += 1
        
        count = 0
        for x in initial_solutions:
            print("When wavelengths of less than ", x[0], "correlation are removed")
            print("The lack of fit is: ", x[1])
        #print(initial_solutions)
                            
    #=============================================================================
    #--------------------------- DIAGNOSTIC TOOLS ------------------------
    #=============================================================================
        
    def lack_of_fit(self):
        """ Runs basic post-processing lack of fit analysis
        
            Args:
                None
    
            Returns:
                lack of fit (int): percentage lack of fit
    
        """        
        nt = self._n_meas_times
        nc = self._n_components #changed from n_actual
        nw = self._n_meas_lambdas
        
        D_model = np.zeros((nt,nw))
        
        c_count = -1
        t_count = 0
        # added due to new structure for non_abs species, non-absorbing species not included in S and Cs as subset of C (CS):
        if hasattr(self,'_abs_components'):
            nabs = self._nabs_components #number of absorbing components (CS)
            Cs = np.zeros((nt, nabs))
            Ss = np.zeros((nw, nabs))
            for c in self._abs_components:
                c_count += 1
                t_count = 0
                for t in self._meas_times:
                    Cs[t_count, c_count] = self.model.Cs[t, c].value
                    t_count += 1

            c_count = -1
            l_count = 0
            for c in self._abs_components:
                c_count += 1
                l_count = 0
                for l in self._meas_lambdas:
                    Ss[l_count, c_count] = self.model.S[l, c].value
                    l_count += 1
            D_model = Cs.dot(Ss.T)
        else:
            C = np.zeros((nt, nc))
            S = np.zeros((nw, nc))
            for c in self._sublist_components:
                c_count += 1
                t_count = 0
                for t in self._meas_times:
                    C[t_count,c_count] = self.model.C[t,c].value
                    t_count += 1

            c_count = -1
            l_count = 0
            for c in self._sublist_components:
                c_count += 1
                l_count = 0
                for l in self._meas_lambdas:
                    S[l_count,c_count] = self.model.S[l,c].value
                    l_count += 1
            D_model = C.dot(S.T)

        
        sum_e = 0
        sum_d = 0
        t_count = -1
        l_count = 0 
        
        for t in self._meas_times:
            t_count += 1
            l_count = 0
            for l in self._meas_lambdas:
                sum_e += (D_model[t_count,l_count] - self.model.D[t,l])**2
                l_count += 1
         
        t_count = -1
        l_count = 0
        for t in self._meas_times:
            for l in self._meas_lambdas:
                sum_d += (self.model.D[t,l])**2
        
        lof = ((sum_e/sum_d)**0.5) * 100
        
        print("The lack of fit is ", lof, " %")
        return lof

    def wavelength_correlation(self):
        """ determines the degree of correlation between the individual wavelengths and 
        the and the concentrations. 
        
            Args:
                None
    
            Returns:
                dictionary of correlations with wavelength
    
        """        
        nt = self._n_meas_times
        
        cov_d_l = dict()
        #calculating the covariance dl with ck
        for c in self._sublist_components:
            for l in self._meas_lambdas:
                mean_d = (sum(self.model.D[t,l] for t in self._meas_times)/nt)
                mean_c = (sum(self.model.C[t,c].value for t in self._meas_times)/nt)
                cov_d_l[l,c] = 0
                for t in self._meas_times:
                    cov_d_l[l,c] += (self.model.D[t,l] - mean_d)*(self.model.C[t,c].value - mean_c)
        
                cov_d_l[l,c] = cov_d_l[l,c]/(nt-1)
        
        #calculating the standard devs for dl and ck over time  
        s_dl = dict()
        
        for l in self._meas_lambdas:
            s_dl[l] = 0
            mean_d = (sum(self.model.D[t,l] for t in self._meas_times)/nt)
            error = 0
            for t in self._meas_times:
                error += (self.model.D[t,l] - mean_d)**2
            s_dl[l] = (error/(nt-1))**0.5
        
        s_ck = dict()
        
        for c in self._sublist_components:
            s_ck[c] = 0
            mean_c = (sum(self.model.C[t,c].value for t in self._meas_times)/nt)
            error = 0
            for t in self._meas_times:
                error += (self.model.C[t,c].value - mean_c)**2
            s_ck[c] = (error/(nt-1))**0.5
        
        cor_lc = dict()
        
        for c in self._sublist_components:
            for l in self._meas_lambdas:
                cor_lc[l,c] = cov_d_l[l,c]/(s_ck[c]*s_dl[l])
        
        cor_l = dict()
        for l in self._meas_lambdas:
            cor_l[l]=max(cor_lc[l,c] for c in self._sublist_components)

        return cor_l

    
def split_sipopt_string(output_string):
    start_hess = output_string.find('DenseSymMatrix')
    ipopt_string = output_string[:start_hess]
    hess_string = output_string[start_hess:]
    # print(hess_string, ipopt_string)
    return (ipopt_string, hess_string)


def split_k_aug_string(output_string):
    start_hess = output_string.find('')
    ipopt_string = output_string[:start_hess]
    hess_string = output_string[start_hess:]
    # print(hess_string, ipopt_string)
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


#######################additional for inputs###CS
def t_ij(time_set, i, j):
    # type: (ContinuousSet, int, int) -> float
    """Return the corresponding time(continuous set) based on the i-th finite element and j-th collocation point
    From the NMPC_MHE framework by @dthierry.

    Args:
        time_set (ContinuousSet): Parent Continuous set
        i (int): finite element
        j (int): collocation point

    Returns:
        float: Corresponding index of the ContinuousSet
    """
    if i < time_set.get_discretization_info()['nfe']:
        h = time_set.get_finite_elements()[i + 1] - time_set.get_finite_elements()[i]  #: This would work even for 1 fe
    else:
        h = time_set.get_finite_elements()[i] - time_set.get_finite_elements()[i - 1]  #: This would work even for 1 fe
    tau = time_set.get_discretization_info()['tau_points']
    fe = time_set.get_finite_elements()[i]
    time = fe + tau[j] * h
    return round(time, 6)


def fe_cp(time_set, feedtime):
    # type: (ContinuousSet, float) -> tuple
    # """Return the corresponding fe and cp for a given time
    # Args:
    #    time_set:
    #    t:
    # """
    fe_l = time_set.get_lower_element_boundary(feedtime)
   # print("fe_l", fe_l)
    fe = None
    j = 0
    for i in time_set.get_finite_elements():
        if fe_l == i:
            fe = j
            break
        j += 1
    h = time_set.get_finite_elements()[1] - time_set.get_finite_elements()[0]
    tauh = [i * h for i in time_set.get_discretization_info()['tau_points']]
    j = 0  #: Watch out for LEGENDRE
    cp = None
    for i in tauh:
        if round(i + fe_l, 6) == feedtime:
            cp = j
            break
        j += 1
    return fe, cp

#: This class can replace variables from an expression
class ReplacementVisitor(EXPR.ExpressionReplacementVisitor):
    def __init__(self):
        super(ReplacementVisitor, self).__init__()
        self._replacement = None
        self._suspect = None

    def change_suspect(self, suspect_):
        self._suspect = suspect_

    def change_replacement(self, replacement_):
        self._replacement = replacement_

    def visiting_potential_leaf(self, node):
        #
        # Clone leaf nodes in the expression tree
        #
        if node.__class__ in native_numeric_types:
            return True, node

        if node.__class__ is NumericConstant:
            return True, node


        if node.is_variable_type():
            if id(node) == self._suspect:
                d = self._replacement
                return True, d
            else:
                return True, node

        return False, None
################################################

def wavelength_subset_selection(correlations = None, n = None):
    """ identifies the subset of wavelengths that needs to be chosen, based
    on the minimum correlation value set by the user (or from the automated 
    lack of fit minimization procedure)
     
        Args:
            correlations (dict): dictionary obtained from the wavelength_correlation
                    function, containing every wavelength from the original set and
                    their correlations to the concentration profile.
                   
            n (int): a value between 0 - 1 that slects the minimum amount
                    correlation between the wavelength and the concentrations.
    
        Returns:
            dictionary of correlations with wavelength
    
    """      
    if not isinstance(correlations, dict):
        raise RuntimeError("correlations must be of type dict()! Use wavelength_correlation function first!")
        
    #should check whether the dictionary contains all wavelengths or not
    if not isinstance(n, float):
        raise RuntimeError("n must be of type int!")
    elif n > 1 or n < 0:
        raise RuntimeError("n must be a number between 0 and 1")             
       
    subset_dict = dict()
    for l in six.iterkeys(correlations):
        if correlations[l] >= n:
            subset_dict[l] = correlations[l]
    return subset_dict

#=============================================================================
#----------- PARAMETER ESTIMATION WITH WAVELENGTH SELECTION ------------------
#=============================================================================

def construct_model_from_reduced_set(builder_clone, end_time, D):
    """ constructs the new pyomo model based on the selected wavelengths.
     
        Args:
            builder_clone (TemplateBuilder): Template builder class of complete model
                            without the data added yet 
            end_time (int): the end time for the data and simulation
            D (dataframe): the new, reduced dataset with only the selected wavelengths.
    
        Returns:
            opt_mode(TemplateBuilder): new Pyomo model from TemplateBuilder, ready for
                    parameter estimation
    
    """ 

    if not isinstance(builder_clone, TemplateBuilder):
        raise RuntimeError('builder_clone needs to be of type TemplateBuilder')
    
    if not isinstance(D, pd.DataFrame):
        raise RuntimeError('Spectral data format not supported. Try pandas.DataFrame')
        
    if not isinstance(end_time, int):
        raise RuntimeError('nfe needs to be type int. Number of finite elements must be defined')

    builder_clone.add_spectral_data(D)
    opt_model = builder_clone.create_pyomo_model(0.0,end_time)
    
    return opt_model
    
def run_param_est(opt_model, nfe, ncp, sigmas):
    """ Runs the parameter estimator for the selected subset
     
        Args:
            opt_model (pyomo model): The model that we wish to run the 
            nfe (int): number of finite elements
            ncp (int): number of collocation points
            sigmas(dict): dictionary containing the variances, as used in the ParameterEstimator class
            
        Returns:
            results_pyomo (results of optimization): Parameter Estimation results
            lof (float): lack of fit results
    
    """ 
    
    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
    options = dict()
    
    # These may not always solve, so we need to come up with a decent initialization strategy here
    results_pyomo = p_estimator.run_opt('ipopt',
                                      tee=False,
                                      solver_opts = options,
                                      variances=sigmas)
    lof = p_estimator.lack_of_fit()

    return results_pyomo, lof
    
