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
        print(sigma_sq)
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
                print("dummyP", i)
                m.dummyP[i] = m.P[i].value
                
        if varcount >= 1:
            m.dvset = Set(initialize = varlist)
            m.dummyV= Param(m.dvset, mutable =True)
            for i in varlist:
                print("dummyV", i)
                print(m.P[i].value)
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
        print("count_vars:",count_vars)
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

        for i,k in six.iteritems(m.dcdp):
            print(i,k)
            print(type(i))

        dsdp = np.loadtxt('dxdp_.dat')
        
        if os.path.exists('dxdp_.dat'):
            os.remove('dxdp_.dat')
        print(idx_to_param)
        return dsdp , idx_to_param

    def rank_params_yao(self, param_scaling = None, meas_scaling = None, sigmas = None):
        """This function ranks parameters in the method described in Yao (2003) by obtaining the sensitivities related
        to the parameters in the model through solving the original NLP model for concentrations, getting the sensitivities
        relating to each paramater, and then using them to predict the next sensitivity. User must provide scaling factors
        as defined in the paper. These are in the form of dictionaries, relating to the confidences relating to the initial
        guesses for the parameters as well as for the confidence inthe measurements.


        Args:
        ----------
        param_scaling: dictionary
        dictionary including each parameter and their relative uncertainty. e.g. a value of 0.5 means that the value
        for the real parameter is within 50% of the guessed value
    
        meas_scaling: scalar
        scalar value showing the certainty of the measurement obtained from the device manufacturer or general 
        knowledge of process
        
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
        
        dsdp, idx_to_param = self.get_sensitivities_for_params(tee=True,sigmasq=sigmas)

        print(dsdp.size)
        nvars = np.size(dsdp,0)
        print("dsdp", dsdp)
        nparams = 0
        #idx_to_param = {}
        for v in six.itervalues(self.model.P):
            if v.is_fixed():
                print(v, end='\t')
                print("is fixed")
                continue
            print("v", v)
            #idx_to_param[nparams]=v
            nparams += 1
        print("nparams", nparams)    
        #all_H = dsdp
        #H = all_H[:,-nparams:]
        print("dsdp", dsdp)
        dsdp_scaled = dsdp

        i=0
        for k, p in self.model.P.items():
            if p.is_fixed():
                continue
            print(k,p)
            print(param_scaling[k])
            for row in range(len(dsdp)):
                dsdp_scaled[row][i] = dsdp[row][i]*param_scaling[k]/meas_scaling
            i += 1
            
        print("dsdp: ", dsdp)
        print("dsdp scaled: ", dsdp_scaled)
        #euclidean norm for each column of Hessian relating parameters to outputs
        eucnorm = dict()
        #paramdict = dict()
        count=0
        for i in range(nparams):
            print(i)
            total = 0
            for row in range(len(dsdp)):
                total += dsdp[row][count]**2
            print(idx_to_param[1 + i])            
            float(total)
            total = np.asscalar(total)
            print(total)
            sqr = (total)**(0.5)
            eucnorm[count]=sqr
            #paramdict[count]=idx_to_param[i]
            count+=1
           
        print("Euclidean Norms: ", eucnorm)
        
        sorted_euc = sorted(eucnorm.values(), reverse=True)
        print("Sorted Norms: ",sorted_euc)

        count=0
        ordered_params = dict()
        for p in idx_to_param:
            for t in idx_to_param:
                if sorted_euc[p-1]==eucnorm[t-1]:
                    ordered_params[count] = t-1
            count +=1
        print("Euclidean Norms, sorted: ",sorted_euc)
        print("params: ", idx_to_param)
        #print("ordered param dict: ", paramdict)
        print("ordered params:", ordered_params)
        for i in idx_to_param:
            print(i)
            print("idx_to_param[i]:", idx_to_param[i])
            
        for i in ordered_params:
            print(i)
            print("orderedparams[i]:", ordered_params[i])
            
        iter_count=0
        self.param_ranks[1] = idx_to_param[ordered_params[0]+1]
        for i in self.param_ranks:
            print(i)
            print("parameter ranked first:", self.param_ranks[i])
            
        #The ranking strategy of Yao, where the X and Z matrices are formed
        next_est = dict()
        X= None
        kcol = None
        for i in range(nparams-1):
            print("i", i)
            print(iter_count)
            print("nvars:",nvars)
            if i==0:
                print("hi there")
                X = np.zeros((nvars,1))
                #X = X.reshape((nvars,1))
            print(X)
    
            for k in range(i+1):
                print("iter_count",iter_count)
                
            for x in range(i+1):
                paramhere = ordered_params[x]
                print("paramhere:", paramhere)
                print("paramhere proper:", idx_to_param[ordered_params[x]+1])
                print(x)
                print("H shape: ", dsdp.shape)
                print("Hcol", dsdp[:,ordered_params[x]])
                print(dsdp[:,ordered_params[x]].shape)
                
                kcol = dsdp[:,ordered_params[x]].T
                print("X size: ", X.shape)
                print("kcol size: ", kcol.shape)
                print(kcol)
                recol= kcol.reshape((nvars,1))
                print("recol",recol)
                print("recolshape: ", recol.shape)
                if x >= 1:
                    X = np.append(X,np.zeros([len(X),1]),1)
                print("X",X)
                print(X.shape)
                for n in range(nvars):
                    #print("x",x)
                    #print("ordered param x",ordered_params[x])
                    #print("n",n)
                    #print(X[n][x])
                    #print(recol[n][0])
                    X[n][x] = recol[n][0]
                print(X)
                print(X.shape)
                #Use Ordinary Least Squares to use X to predict Z
            try:
                A = X.T.dot(X)
                print("A",A)
                print("Ashape:", A.shape)
                B= np.linalg.inv(A)
                print("B",B)
                print("B shape: ", B.shape)
                C = B.dot(X.T)
                print(C)
                print(C.shape)
                D=X.dot(C)
                print("D",D)
                print("D shape",D.shape)
                Z = dsdp.T
                print("Z.shape", Z.shape)
                Zbar=D.dot(Z.T)
                print(dsdp)
                print(dsdp.shape)
                print("Zbar:", Zbar)
                print("Zbar shape: ", Zbar.shape)
                #Get residuals of prediction
                Res = Z.T - Zbar
            except:
                print("Singular matrix, unable to continue the procedure")
                break
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
            print("magres: ", magres)
            print(ordered_params)
            for i in ordered_params:
                print(i)
                print("ordered_params[i]:", ordered_params[i])
            sorted_magres = sorted(magres.values(), reverse=True)
            print("sorted_magres",sorted_magres)
            count2=0
            for p in idx_to_param:
                for t in idx_to_param:
                    print(t)
                    if sorted_magres[p-1]==magres[t-1]:
                        next_est[count2] = t
                count2 += 1
            print("next_est",next_est)  
            self.param_ranks[(iter_count+2)]=idx_to_param[next_est[0]]
            iter_count += 1
            for i in self.param_ranks:
                print(i)
                print("self.param_ranks:", self.param_ranks[i])
            print("======================PARAMETER RANKED======================")
            print("len(self.param_ranks)", len(self.param_ranks))
            print("nparam-1", nparams - 1)
            if len(self.param_ranks) == nparams - 1:
                print(len(self.param_ranks))
                print(nparams-1)
                print("All parameters have been ranked")
                break
        
        #adding the unranked parameters to the list
        #NOTE: if param appears here then it was not evaluated (i.e. it was the least estimable)
        count = 0
        self.unranked_params = {}
        for v,p in six.iteritems(self.model.P):
            print(v,p)
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
        print(count)
        for i in self.ordered_params:
            print(i)
        return self.ordered_params
            
    def run_analyzer(self, method = None, parameter_rankings = None):
        """This function performs the estimability analysis. The user selects the method to be used. The default will
        be selected based on the type of data selected. For now, only the method of Wu, McLean, Harris, and McAuley 
        (2011) using the means squared error is used. Other estimability analysis tools will be added in time. 
        The parameter rankings need to be included as well and this can be done using various methods, however for now, 
        only the Yao (2003) method is used.


        Args:
        ----------
        method: function
            The estimability method to be used. Default is Wu, et al (2011) for concentrations. Others to be added
    
        parameter_rankings: list
            A list containing the parameter rankings in order from most estimable to least estimable. Can be obtained using
            one of Kipet's parameter ranking functions.
        
        returns:
            list of parameters that should remain in the parameter estimation, while all other parameters should be fixed.
        """
        if method == None:
            method = "Wu"
            print("The method to be used is that of Wu, et al. 2011")
        elif method != "Wu":
            print("The only supported method for estimability analysis is tht of Wu, et al., 2011, at the moment")
        else:
            method = "Wu"
            
        if parameter_rankings == None:
            raise RuntimeError('The parameter rankings need to be provided in order to run the estimability analysis chosen')
            
        elif parameter_rankings != None:
            if type(parameter_rankings) is not dict:
                raise RuntimeError('The parameter_rankings must be type dict')   
                
        for v in six.itervalues(self.model.P): 
            if v in parameter_rankings.values():
                continue
        
        
        if method == "Wu":
            self.wu_estimability()

    def wu_estimability(self, parameter_rankings = None):
        """This function performs the estimability analysis of Wu, McLean, Harris, and McAuley (2011) 
        using the means squared error. 

        Args:
        ----------
        parameter_rankings: list
            A list containing the parameter rankings in order from most estimable to least estimable. Can be obtained using
            one of Kipet's parameter ranking functions.
        
        returns:
            list of parameters that should remain in the parameter estimation, while all other parameters should be fixed.
        """
        
        J = dict()
        
        
    def run_lsq_given_some_P(self,solver,parameters,**kwds):
        
        """Determines the minimised weighted sum of squared residuals based on
        solving the problem with certain parameters fixed and others left as variables
        
        Args:
            parameters(list): which parameters are variable
            solver (str): name of the nonlinear solver to used
          
            solver_opts (dict, optional): options passed to the nonlinear solver
        
            variances (dict, optional): map of component name to noise variance. The
            map also contains the device noise variance
            
            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not

            initialization (bool, optional): flag indicating whether result should be 
            loaded or not to the pyomo model
        
        Returns:
            Results object with loaded results

        """
        solver_opts = kwds.pop('solver_opts', dict())
        variances = kwds.pop('variances',dict())
        tee = kwds.pop('tee',False)
        initialization = kwds.pop('initialization',False)
        wb = kwds.pop('with_bounds',True)
        max_iter = kwds.pop('max_lsq_iter',200)
        
        if not self.model.time.get_discretization_info():
            raise RuntimeError('apply discretization first before running simulation')

        base_values = ResultsObject()
        base_values.load_from_pyomo_model(self.model,
                                          to_load=['Z','dZdt','X','dXdt','Y'])

        # fixes parameters not being estimated
        old_values = {}   
        for k,v in self.model.P.items():
            for k1,v1 in parameters.items():
                if k == k1:
                    print("Still variable = ", k)
                    continue
                elif self.model.P[k].fixed ==False:
                    old_values[k] = self.model.P[k].value
                    self.model.P[k].value = v
                    print(self.model.P[k])
                    print(v)
                    self.model.P[k].fixed = True

        for k,v in self.model.P.items():
            if not v.fixed:
                print('parameter {} is not fixed for this estimation'.format(k))
            
        # deactivates objective functions for simulation                
        objectives_map = self.model.component_map(ctype=Objective,active=True)
        active_objectives_names = []
        for obj in six.itervalues(objectives_map):
            name = obj.cname()
            active_objectives_names.append(name)
            obj.deactivate()

            
        opt = SolverFactory(solver)
        for key, val in solver_opts.items():
            opt.options[key]=val

        solver_results = opt.solve(self.model,tee=tee)

        #unfixes the parameters that were fixed
        for k,v in old_values.items():
            if not initialization:
                self.model.P[k].value = v 
            self.model.P[k].fixed = False
            self.model.P[k].stale = False
        # activates objective functions that were deactivated
        active_objectives_names = []
        objectives_map = self.model.component_map(ctype=Objective)
        for name in active_objectives_names:
            objectives_map[name].activate()

        # unstale variables that were marked stale
        for var in six.itervalues(self.model.component_map(ctype=Var)):
            if not isinstance(var,DerivativeVar):
                for var_data in six.itervalues(var):
                    var_data.stale=False
            else:
                for var_data in six.itervalues(var):
                    var_data.stale=True

        # retriving solutions to results object  
        #results = ResultsObject()
        #results.load_from_pyomo_model(self.model,
        #                              to_load=['Z','dZdt','X','dXdt','Y'])

        #c_array = np.zeros((self._n_meas_times,self._n_components))
        #for i,t in enumerate(self._meas_times):
        #    for j,k in enumerate(self._mixture_components):
        #        c_array[i,j] = results.Z[k][t]

        #results.C = pd.DataFrame(data=c_array,
        #                         columns=self._mixture_components,
        #                         index=self._meas_times)
        
        #D_data = self.model.D
        
        #if self._n_meas_times and self._n_meas_times<self._n_components:
        #    raise RuntimeError('Not enough measurements num_meas>= num_components')

        # solves over determined system
        #s_array = self._solve_S_from_DC(results.C,
        #                                tee=tee,
        #                                with_bounds=wb,
        #                                max_iter=max_iter)

        #d_results = []
        #for t in self._meas_times:
        #    for l in self._meas_lambdas:
        #        d_results.append(D_data[t,l])
        #d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
                        
        #results.S = pd.DataFrame(data=s_array,
        #                         columns=self._mixture_components,
        #                         index=self._meas_lambdas)

        #results.D = pd.DataFrame(data=d_array,
        #                         columns=self._meas_lambdas,
        #                         index=self._meas_times)        

        #if initialization:
        #    for t in self.model.meas_times:
        #        for k in self.mixture_components:
        #            self.model.C[t,k].value = self.model.Z[t,k].value

            #for l in self.model.meas_lambdas:
            #    for k in self.mixture_components:
            #        self.model.S[l,k].value =  results.S[k][l]
        #else:
        #    if not base_values.Z.empty:
        #        self.initialize_from_trajectory('Z',base_values.Z)
        #        self.initialize_from_trajectory('dZdt',base_values.dZdt)
        #    if not base_values.X.empty:
        #        self.initialize_from_trajectory('X',base_values.X)
        #        self.initialize_from_trajectory('dXdt',base_values.dXdt)
        
        return results

    
def read_dxdp(dxdp_string, dxdp, n_vars):
    """
    Args:
        
        
    """
    dxdp = np.zeros((n_vars, n_vars))
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
