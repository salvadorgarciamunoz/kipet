# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
from kipet.library.ParameterEstimator import *
from pyomo.core.base.expr import Expr_if
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import numpy as np
import scipy
import six
import copy
import re
import os

__author__ = 'Michael Short'  #: November 2018

class EstimabilityAnalyzer(ParameterEstimator):
    """This class is for estimability analysis. For now it will be used to select the parameter set that
    is suitable for estimation based on a mean squared error (MSE) based approach first described by
    Wu, McLean, Harris, and McAuley (2011). The class will contain a number of functions that will perform the 
    estimability analysis. This should eventually be expanded to include a host of functions and methods.

    Parameters
    ----------
    model : TemplateBuilder
        The full model TemplateBuilder problem needs to be fed into the Estimability Analyzer as this is needed
        in order to build the sensitivities for ranking parameters as well as for constructing the simplified models
    """

    def __init__(self, model):
        super(EstimabilityAnalyzer, self).__init__(model)
        self.param_ranks = dict()
        
    def run_sim(self, solver, **kdws):
        raise NotImplementedError("EstimabilityAnalyzer object does not have run_sim method. Call run_analyzer")

    def get_sensitivities_for_params(self, **kwds):
        """ Obtains the sensitivities (dsdp) using k_aug. This function only works for
        concentration-only problems and obtains the sensitivities based on the initial parameter
        values and how they affect the output (Z).
        
        Args:        
            sigmasq (dict): map of component name to noise variance. The
            map also contains the device noise variance
            
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not
            
        Returns:
            dsdp (numpy matrix):  sensitivity matrix with columns being parameters and rows the Z vars
            idx_to_params (dict): dictionary that maps the columns to the parameters
            (This should probably be a global variable, not a return)
        """       
        
        if not self.model.time.get_discretization_info():
            raise RuntimeError('apply discretization first before running the estimability')
            
        sigma_sq = kwds.pop('sigmasq', dict())
        tee = kwds.pop('tee', False)
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
            raise NotImplementedError("In order to use the estimability analysis from concentration data requires concentration data model.C[ti,cj]")

        all_sigma_specified = True
        #print(sigma_sq)
        keys = sigma_sq.keys()
        for k in list_components:
            if k not in keys:
                all_sigma_specified = False
                sigma_sq[k] = max(sigma_sq.values())
                
        if not all_sigma_specified:
            raise RuntimeError(
                'All variances must be specified to determine sensitivities.\n Please pass variance dictionary to run_opt')
        
        m = self.model

        # estimation
        def rule_objective(m):
            obj = 0
            for t in m.meas_times:
                obj += sum((m.C[t, k] - m.Z[t, k]) ** 2 / sigma_sq[k] for k in list_components)

            return obj
            
        m.objective = Objective(rule=rule_objective)

        #set dummy variables for k_aug to do sensitivities
        paramcount = 0
        paramlist=list()
        varcount = 0
        varlist = list()
        for k,v in six.iteritems(m.P):
            if v.is_fixed():
                paramcount +=1
                paramlist.append(k)
            else:
                varcount += 1
                varlist.append(k)
                
        if paramcount >= 1:
            m.dpset = Set(initialize = paramlist)
            m.dummyP= Var(m.dpset)
            for i in paramlist:
                #print("dummyP", i)
                m.dummyP[i] = m.P[i].value
                
        if varcount >= 1:
            m.dvset = Set(initialize = varlist)
            m.dummyV= Param(m.dvset, mutable =True)
            for i in varlist:
                #print("dummyV", i)
                #print(m.P[i].value)
                m.dummyV[i] = m.P[i].value
        
        #set dummy constraints   
        def dummy_constraints(m,p):
            if p in varlist:
                return 0 == m.dummyV[p] - m.P[p] 
            if p in paramlist:
                return 0 == m.dummyP[p] - m.P[p] 
           
        m.dummyC = Constraint(m.parameter_names, rule=dummy_constraints)     
        
        #set up suffixes for Ipopt that are required for k_aug
        m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

        #: K_AUG SUFFIXES  
        m.dcdp = Suffix(direction=Suffix.EXPORT)  #: the dummy constraints
        m.var_order = Suffix(direction=Suffix.EXPORT)  #: Important variables (primal)
        
        # set which are the variables and which are the parameters for k_aug
        
        count_vars = 1
        #print("count_vars:",count_vars)
        if not self._spectra_given:
            pass
        else:
            for t in self._meas_times:
                for c in self._sublist_components:
                    m.C[t, c].set_suffix_value(m.var_order,count_vars)
                        
                    count_vars += 1
        
        if not self._spectra_given:
            pass
        else:
            for l in self._meas_lambdas:
                for c in self._sublist_components:
                    m.S[l, c].set_suffix_value(m.var_order,count_vars)
                    count_vars += 1
                        
        if self._concentration_given:
            for t in self._meas_times:
                for c in self._sublist_components:
                    m.Z[t, c].set_suffix_value(m.var_order,count_vars)
                        
                    count_vars += 1
        
        count_dcdp = 1

        idx_to_param = dict()
        for p in m.parameter_names:
            
            m.dummyC[p].set_suffix_value(m.dcdp,count_dcdp)
            idx_to_param[count_dcdp]=p
            count_dcdp+=1
          
        #: Clear this file
        with open('ipopt.opt', 'w') as f:
            f.close()
                
        #first solve with Ipopt
        ip = SolverFactory('ipopt')
        solver_results = ip.solve(m, tee=False,
                                  report_timing=True)

        m.ipopt_zL_in.update(m.ipopt_zL_out)
        m.ipopt_zU_in.update(m.ipopt_zU_out) 
        
        k_aug = SolverFactory('k_aug')
        k_aug.options['dsdp_mode'] = ""  #: sensitivity mode!
        #solve with k_aug in sensitivity mode
        k_aug.solve(m, tee=True)
        print("Done solving sensitivities")
            
        #print('k_aug dsdp')
        #m.dcdp.pprint()

        dsdp = np.loadtxt('dxdp_.dat')
        
        if os.path.exists('dxdp_.dat'):
            os.remove('dxdp_.dat')
        print(idx_to_param)
        return dsdp , idx_to_param

    def rank_params_yao(self, param_scaling = None, meas_scaling = None, sigmas = None):
        """This function ranks parameters in the method described in Yao (2003) by obtaining the sensitivities related
        to the parameters in the model through solving the original NLP model for concentrations, getting the sensitivities
        relating to each paramater, and then using them to predict the next sensitivity. User must provide scaling factors
        as defined in the paper. These are in the form of dictionaries, relating the confidences to the initial
        guesses for the parameters as well as for the confidence in the measurements.

        Args:
        ----------
        param_scaling (dictionary): dictionary including each parameter and their relative uncertainty. e.g. a value of 
        0.5 means that the value for the real parameter is within 50% of the guessed value
    
        meas_scaling (scalar): scalar value showing the certainty of the measurement obtained from the device 
        manufacturer or general knowledge of process
        
        sigmasq (dict): map of component name to noise variance. The map also contains the device noise variance.
        
        returns:
            list with order of parameters
        """
        
        if param_scaling == None:
            param_scaling ={}
            print("WARNING: No scaling provided by user, so uncertainties based on the bounds provided by the user is assumed.")
            # uncertainties calculated based on bounds given
            for p in self.model.P:
                lb = self.model.P[p].lb
                ub = self.model.P[p].ub
                init = (ub-lb)/2
                param_scaling[p] = init/(ub-lb)
                print("automated param_scaling", param_scaling)
        elif param_scaling != None:
            if type(param_scaling) is not dict:
                raise RuntimeError('The param_scaling must be type dict')
        
        if meas_scaling == None:
            meas_scaling = 0.001
            print("WARNING: No scaling for measurments provided by user, so uncertainties based on measurements will be set to 0.01")
        elif meas_scaling != None:
            if isinstance(meas_scaling, int) or isinstance(meas_scaling, float):
                print("meas_scaling", meas_scaling)
            else:
                raise RuntimeError('The meas_scaling must be type int')
                
        # k_aug is used to get the sensitivities. The full model is solved with dummy
        # parameters and variables at the initial values for the parameters
        dsdp, idx_to_param = self.get_sensitivities_for_params(tee=True, sigmasq=sigmas)

        nvars = np.size(dsdp,0)
        nparams = 0
        for v in six.itervalues(self.model.P):
            if v.is_fixed():
                print(v, end='\t')
                print("is fixed")
                continue
            nparams += 1

        dsdp_scaled = dsdp

        # scale the sensitivities
        i=0
        for k, p in self.model.P.items():
            if p.is_fixed():
                continue
            
            for row in range(len(dsdp)):
                dsdp_scaled[row][i] = dsdp[row][i]*param_scaling[k]/meas_scaling
            i += 1

        # euclidean norm for each column of Hessian relating parameters to outputs
        eucnorm = dict()
        count=0
        for i in range(nparams):
            total = 0
            for row in range(len(dsdp)):
                total += dsdp[row][count]**2           
            float(total)
            total = np.asscalar(total)
            sqr = (total)**(0.5)
            eucnorm[count]=sqr
            count+=1
        
        # sort the norms and link them to the relevant parameters
        sorted_euc = sorted(eucnorm.values(), reverse=True)

        count=0
        ordered_params = dict()
        for p in idx_to_param:
            for t in idx_to_param:
                if sorted_euc[p-1]==eucnorm[t-1]:
                    ordered_params[count] = t-1
            count +=1
        
        # set the first ranked parameter as the one with highest norm
        iter_count=0
        self.param_ranks[1] = idx_to_param[ordered_params[0]+1]
            
        #The ranking strategy of Yao, where the X and Z matrices are formed
        next_est = dict()
        X= None
        kcol = None
        for i in range(nparams-1):
            if i==0:
                X = np.zeros((nvars,1))
            
            # Form the appropriate matrix
            for x in range(i+1):
                paramhere = ordered_params[x]
                kcol = dsdp[:,ordered_params[x]].T
                recol= kcol.reshape((nvars,1))

                if x >= 1:
                    X = np.append(X,np.zeros([len(X),1]),1)

                for n in range(nvars):
                    X[n][x] = recol[n][0]

            # Use Ordinary Least Squares to use X to predict Z
            # try is here to catch any error resulting from a singular matrix
            try:
                A = X.T.dot(X)
                B= np.linalg.inv(A)
                C = B.dot(X.T)
                D=X.dot(C)
                Z = dsdp.T
                Zbar=D.dot(Z.T)

                #Get residuals of prediction
                Res = Z.T - Zbar
            except:
                print("There was an error during the OLS prediction. Most likely caused by a singular matrix. Unable to continue the procedure")
                break
            
            # Calculate the magnitude of residuals
            magres = dict()
            counter=0
            for i in range(nparams):
                total = 0
                for row in range(len(Res)):
                    total += Res[row][counter]**2
                float(total)
                total = np.asscalar(total)
                sqr = (total)**(0.5)
                magres[counter]=sqr
                counter +=1

            # Sort the residuals and ensure the params are correctly assigned
            sorted_magres = sorted(magres.values(), reverse=True)
            count2=0
            for p in idx_to_param:
                for t in idx_to_param:
                    if sorted_magres[p-1]==magres[t-1]:
                        next_est[count2] = t
                count2 += 1

            # Add next most estimable param to the ranking list  
            self.param_ranks[(iter_count+2)]=idx_to_param[next_est[0]]
            iter_count += 1
            
            print("======================PARAMETER RANKED======================")
            if len(self.param_ranks) == nparams - 1:
                print("All parameters have been ranked")
                break
        
        #adding the unranked parameters to the list
        #NOTE: if param appears here then it was not evaluated (i.e. it was the least estimable)
        count = 0
        self.unranked_params = {}
        for v,p in six.iteritems(self.model.P):
            if p.is_fixed():
                print(v, end='\t')
                print("is fixed")
                continue
            
            if v in self.param_ranks.values():
                continue
            else:
                self.unranked_params[count]=v
                count += 1

        print("The parameters are ranked in the following order from most estimable to least estimable:")
        count = 0
        for i in self.param_ranks:
            print("Number ", i, "is ", self.param_ranks[i])
            count+=1
        
        print("The unranked parameters are the follows: ")
        if len(self.unranked_params) == 0:
            print("All parameters ranked")
            
        for i in self.unranked_params:
            count+=1
            print("unranked ", (count), "is ", self.unranked_params[i])
        
        #preparing final list to return to user
        self.ordered_params = list()
        count = 0
        for i in self.param_ranks:
            self.ordered_params.append(self.param_ranks[i])
            count += 1
        for i in self.unranked_params:
            self.ordered_params.append(self.unranked_params[i])
            count += 1

        return self.ordered_params
