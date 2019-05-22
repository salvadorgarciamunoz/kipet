# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.dae import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.Optimizer import *
from pyomo import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy
import six
import copy
import re
import os
from pyomo.opt import ProblemFormat

__author__ = 'Michael Short'  #: February 2019

class MultipleExperimentsEstimator():
    """This class is for Estimation of Variances and parameters when we have multiple experimental datasets.
    This class relies heavily on the Pyomo block class as we put each experimental class into its own block.
    This blocks are first run individually in order to find good initializations and then they are linked and
    run together in a large optimization problem.

    Parameters
    ----------
    model : TemplateBuilder
        The full model TemplateBuilder problem needs to be fed into the MultipleExperimentsEstimator. This
        pyomo model will form the basis for all the optimization tasks
    """

    def __init__(self,datasets):
        super(MultipleExperimentsEstimator, self).__init__()
        self.block_models = dict()
        self._idx_to_variable = dict()
        
        if datasets != None:
            if isinstance(datasets, dict):
                for key, val in datasets.items():
                    if not isinstance(key, str):
                        raise RuntimeError('The key for the dictionary must be a str')
                    if not isinstance(val, pd.DataFrame):
                        raise RuntimeError('The value in the dictionary must be the experimental dataset as a pandas DataFrame')

        else:
            raise RuntimeError("datasets not given, add datasets as a dict to use this class")
            
        self.datasets = datasets
        self.experiments = list()
        for key,val in self.datasets.items():
            self.experiments.append(key)
        
        self._variance_solved = False
        
        self.variances= dict()
        self.variance_results = dict()
        self.start_time =dict()
        self.end_time = dict()
        self.builder = dict()
        self.opt_model = dict()
        
        self.initialization_model = dict()
        self._sublist_components = dict()

        self._n_meas_times = 0
        self._n_meas_lambdas = 0
        self._n_actual = 0
        self._n_params = 0
        
        self._spectra_given = True
        self._concentration_given = False
        
        self.global_params = None
        
        # set of flags to mark the how many times and wavelengths are in each dataset
        self.l_mark = dict()
        self.t_mark = dict()
        self.n_mark = dict()
        self.p_mark = dict()
          
    def _define_reduce_hess_order_mult(self):
        """This function is used to link the variables to the columns in the reduced
           hessian for multiple experiments.   
           
           Currently this is not functional
        """
        self.model.red_hessian = Suffix(direction=Suffix.IMPORT_EXPORT)
        count_vars = 1

        for i in self.experiments:
            if not self._spectra_given:
                pass
            else:
                for t in self.model.experiment[i].meas_times:
                    for c in self._sublist_components[i]:
                        v = self.model.experiment[i].C[t, c]
                        self._idx_to_variable[count_vars] = v
                        self.model.red_hessian[v] = count_vars
                        count_vars += 1                        
            #print("count after C exp",i, count_vars)             
        #print("count after C",i, count_vars) 
        
        for i in self.experiments:
            if not self._spectra_given:
                pass    
            else:
                for l in self.model.experiment[i].meas_lambdas:
                    for c in self._sublist_components[i]:
                        v = self.model.experiment[i].S[l, c]
                        self._idx_to_variable[count_vars] = v
                        self.model.red_hessian[v] = count_vars
                        count_vars += 1
            #print("count after S exp",i, count_vars)             
        #print("count after S",i, count_vars) 

        for i in self.experiments:
            for k,v in six.iteritems(self.model.experiment[i].P):
                print(k,v)
                if v.is_fixed():
                    print(v, end='\t')
                    print("is fixed")
                    continue
                self._idx_to_variable[count_vars] = v
                #print("count_vars:", count_vars, "self._idx_to_variable[count_vars]")
                self.model.red_hessian[v] = count_vars
                count_vars += 1
            
    def _order_k_aug_hessian(self, unordered_hessian, var_loc):
        """
        not meant to be used directly by users. Takes in the inverse of the reduced hessian
        outputted by k_aug and uses the rh_name to find the locations of the variables and then
        re-orders the hessian to be in a format where the other functions are able to compute the
        confidence intervals in a way similar to that utilized by sIpopt.
        
        Currently not functional
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
                    #print(unordered_hessian)
                    h = unordered_hessian
                    hessian[i, j] = h
                else:
                    h = unordered_hessian[(var_loc[vi]), (var_loc[vj])]
                    hessian[i, j] = h
                j += 1
            i += 1
        print(hessian.size, "hessian size")
        return hessian
           
    def _compute_covariance(self, hessian, variances):
        
        nt = 0
        exp_count = 0
        for i in self.experiments:
            for t in self.model.experiment[i].meas_times:
                nt+=1
            self.t_mark[exp_count] = nt
            exp_count += 1
                
        self._n_meas_times = nt
        
        nw = 0
        exp_count = 0
        for i in self.experiments:
            for t in self.model.experiment[i].meas_lambdas:
                nw+=1
            self.l_mark[exp_count] = nw
            exp_count += 1
            
        self._n_meas_lambdas = nw
        
        nc = 0
        exp_count = 0
        for i in self.experiments:
            nc += len(self._sublist_components[i])
            self.n_mark[exp_count] = nc
            exp_count +=1
            
        print(nc)        
        self._n_actual = nc
        
        exp_count = 0
        nparams = 0
        for i in self.experiments:
            for v in six.itervalues(self.model.experiment[i].P):
                if v.is_fixed():  #: Skip the fixed ones
                    print(str(v) + '\has been skipped for covariance calculations')
                    continue
                nparams += 1
            self.p_mark[exp_count] = nparams
            exp_count +=1
        # nparams = len(self.model.P)
        self._n_params = nparams
        
        nd = nw * nt
        ntheta = 0
        exp_count = 0
        
        for i in self.experiments:
            
            if exp_count == 0:
                print(self.n_mark[exp_count],self.t_mark[exp_count], self.l_mark[exp_count], self.p_mark[exp_count])
                ntheta += self.n_mark[exp_count] * (self.t_mark[exp_count] + self.l_mark[exp_count]) + self.p_mark[exp_count]
            else:
                ncompx = (self.n_mark[exp_count]-self.n_mark[exp_count - 1])
                ntimex = (self.t_mark[exp_count] - self.t_mark[exp_count - 1] )
                nwavex = (self.l_mark[exp_count] - self.l_mark[exp_count - 1])
                nparmx = self.p_mark[exp_count] - self.p_mark[exp_count - 1]
                print(ncompx,ntimex, nwavex, nparmx)
                ntheta += ncompx * (ntimex+ nwavex) + nparmx
            exp_count += 1
            print("ntheta:", ntheta)
            
        #ntheta = nc * (nw + nt) + nparams

        print("Computing H matrix\n shape ({},{})".format(nparams, ntheta))
        all_H = hessian
        H = all_H[-nparams:, :]
        #print(H)
        print(H.shape)
        # H = hessian
        print("Computing B matrix\n shape ({},{})".format(ntheta, nd))
        self._compute_B_matrix(variances)
        B = self.B_matrix
        print("Computing Vd matrix\n shape ({},{})".format(nd, nd))
        self._compute_Vd_matrix(variances)
        Vd = self.Vd_matrix
        
        R = B.T.dot(H.T)
        A = Vd.dot(R)
        L = H.dot(B)
        Vtheta = A.T.dot(L.T)
        V_theta = Vtheta.T
        V_param = V_theta
        variances_p = np.diag(V_param)
        print(variances_p)
        print('\nConfidence intervals:')
        i = 0
        for exp in self.experiments:
            for k, p in self.model.experiment[exp].P.items():
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
        nt = 0
        exp_count = 0
        for i in self.experiments:
            for t in self.model.experiment[i].meas_times:
                nt+=1
            self.t_mark[exp_count] = nt
            exp_count += 1
                
        self._n_meas_times = nt
        
        nc = 0
        exp_count = 0
        for i in self.experiments:
            nc += len(self._sublist_components[i])
            self.n_mark[exp_count] = nc
            exp_count +=1
            
        #print(nc)        
        self._n_actual = nc
        
        exp_count = 0
        nparams = 0
        for i in self.experiments:
            for v in six.itervalues(self.model.experiment[i].P):
                if v.is_fixed():  #: Skip the fixed ones
                    print(str(v) + '\has been skipped for covariance calculations')
                    continue
                nparams += 1
            self.p_mark[exp_count] = nparams
            exp_count +=1
        # nparams = len(self.model.P)
        self._n_params = nparams
        
        
        self._compute_residuals()
        res = self.residuals
        # sets up matrix with variances in diagonals
        nc = self._n_actual
        nt = self._n_meas_times

        varmat = np.zeros((nc, nc))
        for i in self.experiments:
            for c, k in enumerate(self._sublist_components[i]):
                varmat[c, c] = variances[i][k]
        #print("varmat",varmat)
        # R=varmat.dot(res)
        # L = res.dot(varmat)
        E = 0
        
        for i in self.experiments:
            for t in self.model.experiment[i].meas_times:
                for c, k in enumerate(self._sublist_components[i]):
            
                    E += res[i, t, k] / (variances[i][k] ** 2)
        #print(E)
        # Now we can use the E matrix with the hessian to estimate our confidence intervals
        all_H = hessian
        H = all_H[-nparams:, :]

        covariance_C = H
        print(covariance_C,"covariance matrix")
        variances_p = np.diag(covariance_C)
        print("Parameter variances: ", variances_p)
        print('\nConfidence intervals:')
        i = 0
        for exp in self.experiments:
            for k, p in self.model.experiment[exp].P.items():
        
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
        nc = self._n_actual
        nparams = self._n_params
        print(nt,nw,nc,nparams)
        
        nd = nw * nt
        ntheta = 0
        exp_count = 0
        rindmark = 0
        for i in self.experiments:
            if exp_count == 0:
                print(self.n_mark[exp_count],self.t_mark[exp_count], self.l_mark[exp_count], self.p_mark[exp_count])
                ntheta += self.n_mark[exp_count] * (self.t_mark[exp_count] + self.l_mark[exp_count]) + self.p_mark[exp_count]
                rindmark += self.n_mark[exp_count]*self.t_mark[exp_count]
            else:
                ncompx = (self.n_mark[exp_count]-self.n_mark[exp_count - 1])
                ntimex = (self.t_mark[exp_count] - self.t_mark[exp_count - 1] )
                nwavex = (self.l_mark[exp_count] - self.l_mark[exp_count - 1])
                nparmx = self.p_mark[exp_count] - self.p_mark[exp_count - 1]
                print(ncompx,ntimex, nwavex, nparmx)
                rindmark += ncompx*ntimex
                print("rindmark", rindmark)
                ntheta += ncompx * (ntimex+ nwavex) + nparmx
            exp_count += 1
        #print(nwdict)
        #print(ntdict)
        #print(variances)
        self.B_matrix = np.zeros((ntheta, nw * nt))
        
        # This part here is equivalent to the section underneath
        # I am not deleting it as it may still be useful as it is similar 
        # to the original implementation that we use for a single experiment
        '''
        count = 0
        exp_count = 0
        countk = 0
        countj = 0
        counti = 0
        ishift,kshift,jshift = 0,0,0
        knum = 0
        jnum=0
        for x in self.experiments:
            ishift = counti
            kshift = knum
            jshift = jnum
            print("shifts:", ishift,kshift,jshift)
            for i, t in enumerate(self.model.experiment[x].meas_times):
                for j, l in enumerate(self.model.experiment[x].meas_lambdas):
                    for k, c in enumerate(self._sublist_components[x]):
                        # r_idx1 = k*nt+i
                        if exp_count == 0:
                            nc = self.n_mark[exp_count]
                            #nt = self.t_mark[exp_count]
                            #nw = self.l_mark[exp_count]
                        else: 
                            nc = (self.n_mark[exp_count] - self.n_mark[exp_count - 1])
                            #nt = (self.t_mark[exp_count] - self.t_mark[exp_count - 1] )
                            #nw = (self.l_mark[exp_count] - self.l_mark[exp_count - 1])

                        
                        r_idx1 = ((i + ishift)* nc + (k+ kshift))
                        r_idx2 = ((j + jshift) * nc + (k+ kshift) + nc * nt)
                        # r_idx2 = j * nc + k + nc * nw
                        # c_idx = i+j*nt
                        c_idx = ((i +ishift ) * nw + (j + jshift))
                        # print(j, k, r_idx2)
                        #print(r_idx1,r_idx2,c_idx, i,j,k,nc,nw,nt)
                        self.B_matrix[r_idx1, c_idx] = -2 * self.model.experiment[x].S[l, c].value / (self.variances[x]['device'])
                        # try:
                        self.B_matrix[r_idx2, c_idx] = -2 * self.model.experiment[x].C[t, c].value / (self.variances[x]['device'])
                        #matrixdict[exp_count][r_idx1, c_idx]= -2 * self.model.experiment[x].S[l, c].value / (self.variances[x]['device'])
                       # matrixdict[exp_count][r_idx2, c_idx] = -2 * self.model.experiment[x].C[t, c].value / (self.variances[x]['device'])
                        # except IndexError:
                        #     pass
                        knum = max(knum,k)
                        count += 1
                    countj += 1
                    jnum = max(jnum,j)
                counti += 1
                #print("indices, ", r_idx1,r_idx2,c_idx,i,j,k)
                #print(ishift,kshift,jshift)
            exp_count += 1
            print("nc,nt,nw",nc,nt,nw)
            print(ishift,kshift,jshift)
            print("indices, ", r_idx1,r_idx2,c_idx,i,j,k)
        exp_count = exp_count    
        #for p in range(exp_count):
        #    print("p in range here:", p)
        #    if p == 0:
        #        A = matrixdict[p]
        #        print("A matrix shape = ",A.shape)
        #    else:
        #        print(matrixdict[p].shape)
        #        A = np.concatenate((A,matrixdict[p])) 
        #        print("A matrix shape = ", A.shape)
                
        #self.B_matrix = A
        print("number of times in B loop:", count)
        print("B matrix shape should be = ", ntheta, "X", nd)
        print("B matrix shape = ",self.B_matrix.shape)
        print("B matrix size ",self.B_matrix.size)
        print(self.B_matrix)
        # sys.exit()
        '''
        exp_count = 0

        timeshift, waveshift = 0,0
        nc_prev = 0
        minusr1 = 0
        r_idx1_old = int()
        r_idx2_old = int()
        for i in range(nt):
            for j in range(nw):
                #NEED TO PUT A WAY TO COUNT THE EXPERIMENTS BASED ON THE TOTAL NUMBERS HERE
                if exp_count == 0:
                    nc = self.n_mark[exp_count]
                    nc_prev = nc
                    
                if i == self.t_mark[exp_count] and j == self.l_mark[exp_count]:
                    exp_count += 1
                    timeshift = i
                    waveshift = j
                    nc_prev = nc
                    nc = (self.n_mark[exp_count] - self.n_mark[exp_count - 1])
                    print("HERE IS THE EXP CHANGE")
                
                    #timeshift = self.t_mark[exp_count]
                    #waveshift = self.l_mark[exp_count]
                            
                #elif exp_count != 0 and i == self.t_mark[exp_count] and j == self.l_mark[exp_count]:
                    #nc_prev = nc
                    #nc = (self.n_mark[exp_count] - self.n_mark[exp_count - 1])
                    #nt = (self.t_mark[exp_count] - self.t_mark[exp_count - 1] )
                    #nw = (self.l_mark[exp_count] - self.l_mark[exp_count - 1])
                
                for x, exp in enumerate(self.experiments):
                    #print(x,exp)
                    if x == exp_count:
                        break
                    else:
                        pass
                        
                if exp_count >= 1:
                    minusr1 = nc - nc_prev
                    #print(minusr1)
                
                for k, c in enumerate(self._sublist_components[exp]):
                    for ii, t in enumerate(self.model.experiment[exp].meas_times):
                        if ii + timeshift == i:
                            time = t
                            break
                        else:
                            pass
                    for jj, l in enumerate(self.model.experiment[exp].meas_lambdas):
                        if jj + waveshift == j:
                            wave = l
                            break
                        else:
                            pass 
                        
                    r_idx1 = ((i)* (nc) + (k))
                    #r_idx1 = ((i)* (nc) + (k) - (minusr1*i))
                    #r_idxwhat1 = ((i)* (4) + (k) )   
                    #print(nc,nc_prev)
                    
                    #r_idx2 = ((j) * (nc) + (k) + (rindmark) - (minusr1*j))
                    r_idx2 = ((j) * (nc) + (k) + (nc*nt))
                    #r_idxwhat2 = ((j) * (4) + (k) + (rindmark-1) - (minusr1*j))
                    #(nc_prev) * nt
                    c_idx = ((i) * nw + (j))
                    #print("indices, ", r_idx1,r_idx2,c_idx,i,j,k,wave,time,nc,nc_prev,nt,nw)
                    self.B_matrix[r_idx1, c_idx] = -2 * self.model.experiment[exp].S[wave, c].value / (self.variances[exp]['device'])
                    try:
                        self.B_matrix[r_idx2, c_idx] = -2 * self.model.experiment[exp].C[time, c].value / (self.variances[exp]['device'])
                    except:
                        #print("indices,**** ", r_idx1,r_idx2,c_idx,i,j,k,wave,time,nc,nt,nw) 
                        df = pd.DataFrame(self.B_matrix)
                        df.to_csv('failB.csv')
                        sys.exit()
            r_idx1_old = r_idx1
            r_idx2_old = r_idx2
            #print("indices, **", r_idx1,r_idx2,c_idx,i,j,k,wave,time,nc)
                        
                    
        #print("indices, ", r_idx1,r_idx2,c_idx,i,j,k)        
        print("B matrix shape should be = ", ntheta, "X", nd)
        print("B matrix shape = ",self.B_matrix.shape)
        #print("B matrix size ",self.B_matrix.size)
        print(self.B_matrix)
        df = pd.DataFrame(self.B_matrix)
        df.to_csv('leB.csv')
        

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
        print("self._n_actual", self._n_actual)

        v_array = np.zeros(nc)
                
        #nc = nc/len(self.experiments)
        #nc = int(math.ceil(nc))
        #print("how many components: " , nc)
        s_array = np.zeros(nw * nc)
        
        count = 0
        for x in self.experiments:
            for k, c in enumerate(self._sublist_components[x]):
                #print("v_array: ", k, c)
                v_array[count] = variances[x][c]
                count += 1
        
        #print("v_array full: ", v_array)
        
        kshift,jshift = 0,0
        knum = 0
        jnum=0
        count=0
        exp_count = 0
        for x in self.experiments:
            kshift += knum
            jshift += jnum
            if exp_count != 0:
                kshift+=1
            #print("shifts:", ishift,kshift,jshift)
            for j, l in enumerate(self.model.experiment[x].meas_lambdas):
                for k, c in enumerate(self._sublist_components[x]):
                    
                    if exp_count == 0:
                        nc = self.n_mark[exp_count]
                        nt = self.t_mark[exp_count]
                        nw = self.l_mark[exp_count]
                    else: 
                        #nc = (self.n_mark[exp_count] - self.n_mark[exp_count - 1])
                        nt = (self.t_mark[exp_count] - self.t_mark[exp_count - 1] )
                        nw = (self.l_mark[exp_count] - self.l_mark[exp_count - 1])

                    s_array[(j+jshift) * nc + (k+kshift)] = self.model.experiment[x].S[l, c].value
                    idx = (j+jshift) * nc + (k+kshift)
                    #print(idx)
                    knum = max(knum,k)
                    count += 1
                
                jnum = max(jnum,j)
            #print("nc", nc)
            exp_count += 1

        #print("times in loop:", count)
        row = []
        col = []
        data = []
        nt = self._n_meas_times
        nw = self._n_meas_lambdas
        nd = nt * nw
        # Vd_dense = np.zeros((nd,nd))
        v_device = list()
        
        #print("s_array:", s_array)
        #print(s_array.size)
        #print(s_array.shape)        

        #exp_count = 0
        for x in self.experiments:
            v_device.append(variances[x]['device']) 
            #exp_count += 1
        
        exp_count = 0
        v_device_exp = v_device[exp_count]
        for i in range(nt):
            for j in range(nw):
                if i == self.t_mark[exp_count] and j == self.l_mark[exp_count]:
                    exp_count += 1
                    v_device_exp = v_device[exp_count]

                    
                val = sum(v_array[k] * s_array[j * nc + k] ** 2 for k in range(nc)) + v_device_exp
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
        print(self.Vd_matrix)
        df = pd.DataFrame(self.Vd_matrix)
        df.to_csv('leVd.csv')
        
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
        for x in self.experiments:
            for c in self._sublist_components[x]:
                count_t = 0
                for i, t in enumerate(self.model.experiment[x].meas_times):
                    a = self.model.experiment[x].C[t, c].value
                    b = self.model.experiment[x].Z[t, c].value
                    r = ((a - b) ** 2)
                    self.residuals[x, t, c] = r
                    count_t += 1
                count_c += 1
         
    def run_variance_estimation(self, builder, **kwds):
        """Solves the Variance Estimation procedure described in Chen et al 2016. Here, we call the VarianceEstimator
            seperately on each dataset in order to not only get the model noise and the 
           This method solved a sequence of optimization problems
           to determine variances and initial guesses for parameter estimation.

        Args:
            solver (str,optional): solver to be used, default is ipopt
            
            solver_opts (dict, optional): options passed to the nonlinear solver
            
            start_time (dict): dictionary with key of experiment name and value float
            
            end_time (dict): dictionary with key of experiment name and value float
            
            nfe (int): number of finite elements
            
            ncp (int): number of collocation points
            
            builder (TemplateBuilder): need to add the TemplateBuilder before create_pyomo_model

            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not

            norm (optional): norm for checking convergence. The default value is the infinity norm,
            it uses same options as scipy.linalg.norm

            max_iter (int,optional): maximum number of iterations for Weifengs procedure. Default 400.

            tolerance (float,optional): Tolerance for termination by the change Z. Default 5.0e-5

            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.

            lsq_ipopt (bool,optional): Determines whether to use ipopt for solving the least squares 
            problems in Weifengs procedure. Default False. The default used scipy.least_squares.

            init_C (DataFrame,optional): Dataframe with concentration data used to start Weifengs procedure.

        Returns:

            None

        """
        #Require the same arguments as the VarianceEstimator as these will be applied in the same
        #function, just for each individual dataset
        solver = kwds.pop('solver',str)
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', False)
        norm_order = kwds.pop('norm',np.inf)
        max_iter = kwds.pop('max_iter', 400)
        tol = kwds.pop('tolerance', 5.0e-5)
        A = kwds.pop('subset_lambdas', None)
        lsq_ipopt = kwds.pop('lsq_ipopt', False)
        init_C = kwds.pop('init_C', None)
        
        start_time = kwds.pop('start_time', dict())
        end_time = kwds.pop('end_time', dict())
        nfe = kwds.pop('nfe', 50)
        ncp = kwds.pop('ncp', 3)
        
        # additional arguments for inputs
        # These will need to be made into dicts if there are different 
        # conditions in each experiment
        inputs = kwds.pop("inputs", None)
        inputs_sub = kwds.pop("inputs_sub", None)
        trajectories = kwds.pop("trajectories", None)
        fixedtraj = kwds.pop('fixedtraj', False)
        fixedy = kwds.pop('fixedy', False)
        yfix = kwds.pop("yfix", None)
        yfixtraj = kwds.pop("yfixtraj", None)

        jump = kwds.pop("jump", False)
        var_dic = kwds.pop("jump_states", None)
        jump_times = kwds.pop("jump_times", None)
        feed_times = kwds.pop("feed_times", None)

        species_list = kwds.pop('subset_components', None)
        
        if not isinstance(start_time, dict):
            raise RuntimeError("Must provide start_times as dict with each experiment")
        else:
            for key, val in start_time.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for start_time need to be the same as the experimental datasets")
        self.start_time = start_time    
        if not isinstance(end_time, dict):
            raise RuntimeError("Must provide end_times as dict with each experiment")    
        else:
            for key, val in end_time.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for end_time need to be the same as the experimental datasets")
        self.end_time = end_time
        
        #This is to make sure that for either cases of 1 model for all experiments or different models, 
        #we ensure we have the correct builders
        
        if isinstance(builder, dict):
            for key, val in builder.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for builder need to be the same as the experimental datasets")   
                    
                if not isinstance(val, TemplateBuilder):
                    raise RuntimeError('builder needs to be type TemplateBuilder')
                    
        elif isinstance(builder, TemplateBuilder):
            builder_dict = {}
            for item in self.experiments:
                builder_dict[item] = builder
            
            builder = builder_dict
            
        else:
            raise RuntimeError("builder added needs to be a dictionary of TemplateBuilders or a TemplateBuilder")
        
        if solver == '':
            solver = 'ipopt'
            
        v_est_dict = dict()
        results_variances = dict()
        sigmas = dict()
        print("SOLVING VARIANCE ESTIMATION FOR INDIVIDUAL DATASETS")
        for l in self.experiments:
            print("solving for dataset ", l)
            self.builder[l] = builder[l]
            self.builder[l].add_spectral_data(self.datasets[l])
            self.opt_model[l] = self.builder[l].create_pyomo_model(start_time[l],end_time[l])
            
            v_est_dict[l] = VarianceEstimator(self.opt_model[l])
            v_est_dict[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
            results_variances[l] = v_est_dict[l].run_opt(solver,
                                            tee=tee,
                                            solver_opts=solver_opts,
                                            max_iter=max_iter,
                                            tol=tol,
                                            subset_lambdas = A)
            print("\nThe estimated variances are:\n")
            for k,v in six.iteritems(results_variances[l].sigma_sq):
                print(k, v)
            self.variance_results[l] = results_variances[l]
            # and the sigmas for the parameter estimation step are now known and fixed
            sigmas[l] = results_variances[l].sigma_sq
            self.variances[l] = sigmas[l] 
        self._variance_solved = True
        
        return results_variances

    def solve_full_problem(self, solver, **kwds):
        """Sets up the reduced hessian and all other calculations for the full problem solve
        Include the covariance calculations
        
        INCOMPLETE - NOT WORKING
        """
        #Check for whether solver is sipopt or kaug
        
        tee = kwds.pop('tee', False)
        weights = kwds.pop('weights', [0.0, 1.0])
        covariance = kwds.pop('covariance', False)
        warmstart = kwds.pop('warmstart', False)
        species_list = kwds.pop('subset_components', None)
        solver_opts = kwds.pop('solver_opts', dict())
        
        if solver != 'ipopt_sens' and solver != 'k_aug':
            raise RuntimeError('To get covariance matrix the solver needs to be ipopt_sens or k_aug')
        if solver == 'ipopt_sens':
            if not 'compute_red_hessian' in solver_opts.keys():
                solver_opts['compute_red_hessian'] = 'yes'
        if solver == 'k_aug':
            solver_opts['compute_inv'] = ''

        optimizer = SolverFactory(solver)
        for key, val in solver_opts.items():
            optimizer.options[key] = val
        
        self._define_reduce_hess_order_mult()
        m = self.model
                    
        if solver == 'ipopt_sens':
            self._tmpfile = "ipopt_hess"
            solver_results = optimizer.solve(m, tee=m.tee,
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
            #tee = True
            if tee == True:
                print(ipopt_output)

            #if not all_sigma_specified:
            #    raise RuntimeError(
            #        'All variances must be specified to determine covariance matrix.\n Please pass variance dictionary to run_opt')

            n_vars = len(self._idx_to_variable)
            #print('n_vars', n_vars)
            hessian = read_reduce_hessian(hessian_output, n_vars)
            print(hessian.size, "hessian size")
            print(hessian.shape,"hessian shape")
            # hessian = read_reduce_hessian2(hessian_output,n_vars)
            # print hessian
            self._compute_covariance(hessian, self.variances)

        elif solver == 'k_aug':
            m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    
            m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
            m.dof_v = Suffix(direction=Suffix.EXPORT)  #: SUFFIX FOR K_AUG
            m.rh_name = Suffix(direction=Suffix.IMPORT)  #: SUFFIX FOR K_AUG AS WELL
    
            count_vars = 1
            for i in self.experiments:
                if not self._spectra_given:
                    pass
                else:
                    for t in m.experiment[i].meas_times:
                        for c in self._sublist_components[i]:
                            m.experiment[i].C[t, c].set_suffix_value(m.dof_v, count_vars)
    
                            count_vars += 1
    
                if not self._spectra_given:
                    pass
                else:
                    for l in m.experiment[i].meas_lambdas:
                        for c in self._sublist_components[i]:
                            m.experiment[i].S[l, c].set_suffix_value(m.dof_v, count_vars)
                            count_vars += 1
    
                for v in six.itervalues(self.model.experiment[i].P):
                    if v.is_fixed():
                        continue
                    m.experiment[i].P.set_suffix_value(m.dof_v, count_vars)
                    count_vars += 1
            
            print("count_vars:", count_vars)
            self._tmpfile = "k_aug_hess"
            ip = SolverFactory('ipopt')
            with open("ipopt.opt", "w") as f:
                f.write("print_info_string yes")
                f.close()
                
            m.write(filename="ip.nl", format=ProblemFormat.nl)
            solver_results = ip.solve(m, tee=m.tee, 
                                      options = solver_opts,
                                      logfile=self._tmpfile,
                                      report_timing=True)
            
            m.write(filename="ka.nl", format=ProblemFormat.nl)
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
            #if self._estimability == True:
            #    self.hessian = hessian
            self._compute_covariance(hessian, sigma_sq)        

    def solve_conc_full_problem(self, solver, **kwds):
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
        solver_opts = kwds.pop('solver_opts', dict())
        
        #solver_opts = dict()
        if solver != 'ipopt_sens' and solver != 'k_aug':
            raise RuntimeError('To get covariance matrix the solver needs to be ipopt_sens or k_aug')
        if solver == 'ipopt_sens':
            if not 'compute_red_hessian' in solver_opts.keys():
                solver_opts['compute_red_hessian'] = 'yes'
        if solver == 'k_aug':
            solver_opts['compute_inv'] = ''
        print(solver_opts)
        optimizer = SolverFactory(solver)
        for key, val in solver_opts.items():
            optimizer.options[key] = val
            
        m = self.model
        
        self._define_reduce_hess_order_mult()
        
        if covariance and solver == 'ipopt_sens':
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

            ipopt_output, hessian_output = split_sipopt_string1(output_string)
            #print (hessian_output)
            print("build strings")
            if tee == True:
                print(ipopt_output)

            n_vars = len(self._idx_to_variable)

            hessian = read_reduce_hessian(hessian_output, n_vars)
            print(hessian.size, "hessian size")
            # hessian = read_reduce_hessian2(hessian_output,n_vars)
            sigma_sq = self.variances
            if self._concentration_given:
                self._compute_covariance_C(hessian, sigma_sq)
            # else:
            #    self._compute_covariance(hessian, sigma_sq)

        elif covariance and solver == 'k_aug':
                
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
                for t in m.experiment[i].meas_times:
                    for c in self._sublist_components[i]:
                        m.experiment[i].C[t, c].set_suffix_value(m.dof_v, count_vars)

                        count_vars += 1
                        

            if not self._spectra_given:
                pass
            else:
                for l in self._meas_lambdas:
                    for c in self._sublist_components:
                        m.S[l, c].set_suffix_value(m.dof_v, count_vars)
                        count_vars += 1

            var_counted = list()
            
            print("globs", self.global_params)
            
            for i in self.experiments:
                for k,v in six.iteritems(self.model.experiment[i].P):
                    #print(k,v)                    
                    if k not in var_counted:
                        #print(count_vars)
                        #print(k,v)
                        if v.is_fixed():  #: Skip the fixed ones
                            print(str(v) + '\has been skipped for covariance calculations')
                            continue
                        m.experiment[i].P.set_suffix_value(m.dof_v, count_vars)
                    count_vars += 1
                    
                    var_counted.append(k)
                    
            self._tmpfile = "k_aug_hess"
            ip = SolverFactory('ipopt')
            solver_results = ip.solve(m, tee=tee,
                                      logfile=self._tmpfile,
                                      report_timing=True)
            # m.P.pprint()
            k_aug = SolverFactory('k_aug')
            k_aug.options['compute_inv'] = ""
            # k_aug.options["no_scale"] = ""
            m.ipopt_zL_in.update(m.ipopt_zL_out)  #: be sure that the multipliers got updated!
            m.ipopt_zU_in.update(m.ipopt_zU_out)
            # m.write(filename="mynl.nl", format=ProblemFormat.nl)
            print("do we get here?")
            k_aug.solve(m, tee=False)
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
        
    def run_parameter_estimation(self, builder, **kwds):
        """Solves the Parameter Estimation procedure described in Chen et al 2016. Here, 
            we call the ParameterEstimator separately on each dataset before adding them
            to blocks and solving simultaneously.

        Args:
            solver (str): name of the nonlinear solver to used

            solver_opts (dict, optional): options passed to the nonlinear solver

            variances (dict, optional): map of component name to noise variance. The
            map also contains the device noise variance

            tee (bool,optional): flag to tell the optimizer whether to stream output
            to the terminal or not

            with_d_vars (bool,optional): flag to the optimizer whether to add
            variables and constraints for D_bar(i,j)
            
            start_time (dict): dictionary with key of experiment name and value float
            
            end_time (dict): dictionary with key of experiment name and value float
            
            nfe (int): number of finite elements
            
            ncp (int): number of collocation points
            
            builder (TemplateBuilder): need to add the TemplateBuilder before create_pyomo_model

            subset_lambdas (array_like,optional): Set of wavelengths to used in initialization problem 
            (Weifeng paper). Default all wavelengths.
            
            sigma_sq (dict): variances
            
            spectra_problem (bool): tells whether we have spectral data or concentration data (False if not specified)

        Returns:

            None

        """
        #Require the same arguments as the VarianceEstimator as these will be applied in the same
        #function, just for each individual dataset
        
        solver = kwds.pop('solver', str)
        solver_opts = kwds.pop('solver_opts', dict())
        tee = kwds.pop('tee', False)
        A = kwds.pop('subset_lambdas', None)
        
        start_time = kwds.pop('start_time', dict())
        end_time = kwds.pop('end_time', dict())
        nfe = kwds.pop('nfe', 50)
        ncp = kwds.pop('ncp', 3)
        
        sigma_sq = kwds.pop('sigma_sq', dict())
        # additional arguments for inputs
        # These will need to be made into dicts if there are different 
        # conditions in each experiment
        inputs = kwds.pop("inputs", None)
        inputs_sub = kwds.pop("inputs_sub", None)
        trajectories = kwds.pop("trajectories", None)
        fixedtraj = kwds.pop('fixedtraj', False)
        fixedy = kwds.pop('fixedy', False)
        yfix = kwds.pop("yfix", None)
        yfixtraj = kwds.pop("yfixtraj", None)

        jump = kwds.pop("jump", False)
        var_dic = kwds.pop("jump_states", None)
        jump_times = kwds.pop("jump_times", None)
        feed_times = kwds.pop("feed_times", None)

        species_list = kwds.pop('subset_components', None)
        
        covariance = kwds.pop('covariance', False)
        
        spectra_problem = kwds.pop('spectra_problem', True)
        
        if covariance:
            if solver != 'ipopt_sens' and solver != 'k_aug':
                raise RuntimeError('To get covariance matrix the solver needs to be ipopt_sens or k_aug')
            #if solver == 'ipopt_sens':
            #    raise RuntimeError('To get covariance matrix for multiple experiments, the solver needs to be k_aug')
        
        if not isinstance(start_time, dict):
            raise RuntimeError("Must provide start_times as dict with each experiment")
        else:
            for key, val in start_time.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for start_time need to be the same as the experimental datasets")
            
        if not isinstance(end_time, dict):
            raise RuntimeError("Must provide end_times as dict with each experiment")    
        else:
            for key, val in end_time.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for end_time need to be the same as the experimental datasets")
         
            
        if isinstance(builder, dict):
            for key, val in builder.items():
                if key not in self.experiments:
                    raise RuntimeError("The keys for builder need to be the same as the experimental datasets")   
                    
                if not isinstance(val, TemplateBuilder):
                    raise RuntimeError('builder needs to be type TemplateBuilder')
                    
        elif isinstance(builder, TemplateBuilder):
            builder_dict = {}
            for item in self.experiments:
                builder_dict[item] = builder
            
            builder = builder_dict
            
        else:
            raise RuntimeError('builder needs to be type dictionary of TemplateBuilder or TemplateBuilder')
            
        p_est_dict = dict()
        results_pest = dict()

        list_components = {}
        for k in self.experiments:            
            list_components[k] = [k for k in builder[k]._component_names]
        self._sublist_components = list_components
        
        if bool(sigma_sq) == False:
            if bool(self.variances) == True: 
                sigma_sq = self.variances
            else:
                raise RuntimeError('Need to add variances in order to run parameter estimation')
        else:
            for key, val in sigma_sq.items():

                if key not in self.experiments:
                    raise RuntimeError("The keys for sigma_sq need to be the same as the experimental datasets")
                    
                all_sigma_specified = True
                keys = sigma_sq[key].keys()
                expsigma = sigma_sq[key]
                
                for k in list_components:
                    if k not in keys:
                        all_sigma_specified = False
                        expsigma[k] = max(expsigma.values())
        
                if not 'device' in val.keys():
                    all_sigma_specified = False
                    expsigma['device'] = 1.0
                
                
        if self._variance_solved == False:
            self.variances = sigma_sq
            
        print("SOLVING PARAMETER ESTIMATION FOR INDIVIDUAL DATASETS - For initialization")
        
        ind_p_est = dict()
        list_params_across_blocks = list()
        all_params = list()
        global_params = list()
        for l in self.experiments:
            print("solving for dataset ", l)
            if self._variance_solved == True and spectra_problem:
                #then we already have inits
                ind_p_est[l] = ParameterEstimator(self.opt_model[l])
                #ind_p_est[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
                               
                ind_p_est[l].initialize_from_trajectory('Z',self.variance_results[l].Z)
                ind_p_est[l].initialize_from_trajectory('S',self.variance_results[l].S)
                ind_p_est[l].initialize_from_trajectory('C',self.variance_results[l].C)
                #NOTICE here that we may need to add X and Y variables and DZdt vars here depending on the situtation
                #This needs to be done based on their existence.
                ind_p_est[l].scale_variables_from_trajectory('Z',self.variance_results[l].Z)
                ind_p_est[l].scale_variables_from_trajectory('S',self.variance_results[l].S)
                ind_p_est[l].scale_variables_from_trajectory('C',self.variance_results[l].C)
                
                results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                      tee=tee,
                                                      solver_opts = solver_opts,
                                                      variances = self.variances[l])
                
                self.initialization_model[l] = ind_p_est[l]
                
                print("The estimated parameters are:")
                for k,v in six.iteritems(results_pest[l].P):
                    print(k, v)
                    if k not in all_params:
                        all_params.append(k)
                    else:
                        global_params.append(k)
                    
                    if k not in list_params_across_blocks:
                        list_params_across_blocks.append(k)
                
                print("all_params:" , all_params)
                print("global_params:", global_params)
            
            else:
                #we do not have inits
                if spectra_problem == True:
                    self._spectra_given = True
                    self.builder[l]=builder[l]
                    self.builder[l].add_spectral_data(self.datasets[l])
                    self.opt_model[l] = self.builder[l].create_pyomo_model(start_time[l],end_time[l])
                    ind_p_est[l] = ParameterEstimator(self.opt_model[l])
                    ind_p_est[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
                    
                    results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                         tee=tee,
                                                          solver_opts = solver_opts,
                                                          variances = sigma_sq[l])
    
                    self.initialization_model[l] = ind_p_est[l]
                
                elif spectra_problem == False:
                    self._spectra_given = False
                    self._concentration_given = True
                    self.builder[l]=builder[l]
                    self.builder[l].add_concentration_data(self.datasets[l])
                    self.opt_model[l] = self.builder[l].create_pyomo_model(start_time[l],end_time[l])
                    ind_p_est[l] = ParameterEstimator(self.opt_model[l])
                    ind_p_est[l].apply_discretization('dae.collocation',nfe=nfe,ncp=ncp,scheme='LAGRANGE-RADAU')
                    
                    results_pest[l] = ind_p_est[l].run_opt('ipopt',
                                                         tee=tee,
                                                          solver_opts = solver_opts,
                                                          variances = sigma_sq[l])
    
    
                    self.initialization_model[l] = ind_p_est[l]                    
                
                print("The estimated parameters are:")
                for k,v in six.iteritems(results_pest[l].P):
                    print(k, v)
                    if k not in all_params:
                        all_params.append(k)
                    else:
                        if k not in global_params:
                            global_params.append(k)
                        else:
                            pass
                    if k not in list_params_across_blocks:
                        list_params_across_blocks.append(k)
                
                print("all_params:" , all_params)
                print("global_params:", global_params)
                self.global_params = global_params
                # I want to build a list of the parameters here that we will use to link the models later on
                
        #Now that we have all our datasets solved individually we can build our blocks and use
        #these solutions to initialize
        m = ConcreteModel()
        m.solver_opts = solver_opts
        m.tee = tee
        
        def build_individual_blocks(m, exp):
            """This function forms the rule for the construction of the individual blocks 
            for multiple experiments, referenced in run_parameter_estimation. This function 
            is not meant to be used by users directly.
            
            Args:
                m (pyomo Concrete model): the concrete model that we are adding the block to
                exp (list): a list containing the experiments
                
            Returns:
                Pyomo model: Pyomo model inside the block
                
            """
            species_list=None
            #if not set_A:
            #    set_A = before_solve_extend._meas_lambdas
            
            list_components = []
            if species_list is None:
                list_components = [k for k in self.initialization_model[exp]._mixture_components]
            else:
                for k in species_list:
                    if k in self.initialization_model[exp]._mixture_components:
                        list_components.append(k)
                    else:
                        warnings.warn("Ignored {} since is not a mixture component of the model".format(k))
                        
            #WITH_D_VARS as far as I can tell should always be True, unless we ignore the device noise
            with_d_vars= True
            
            m = self.initialization_model[exp].model
            if with_d_vars and self._spectra_given:
                m.D_bar = Var(m.meas_times,
                              m.meas_lambdas)
    
                def rule_D_bar(m, t, l):
                    return m.D_bar[t, l] == sum(m.C[t, k] * m.S[l, k] for k in self.initialization_model[exp]._sublist_components)
    
                m.D_bar_constraint = Constraint(m.meas_times,
                                                m.meas_lambdas,
                                                rule=rule_D_bar)
                
            m.error = Var(bounds = (0, None))
            
            if self._spectra_given:
                def rule_objective(m):
                    expr = 0
                    for t in m.meas_times:
                        for l in m.meas_lambdas:
                            if with_d_vars:
                                expr += (m.D[t, l] - m.D_bar[t, l]) ** 2 / (self.variances[exp]['device'])
                            else:
                                D_bar = sum(m.C[t, k] * m.S[l, k] for k in list_components)
                                expr += (m.D[t, l] - D_bar) ** 2 / (self.variances[exp]['device'])
                    
                    #If we require weights then we would add them back in here
                    #expr *= weights[0]
                    second_term = 0.0
                    for t in m.meas_times:
                        second_term += sum((m.C[t, k] - m.Z[t, k]) ** 2 / self.variances[exp][k] for k in list_components)
        
                    #expr += weights[1] * second_term
                    expr += second_term
                    return m.error == expr
        
                m.obj_const = Constraint(rule=rule_objective)
            elif self._concentration_given:
                def rule_objective(m):
                    obj = 0
                    for t in m.meas_times:
                        obj += sum((m.C[t, k] - m.Z[t, k]) ** 2 / self.variances[exp][k] for k in list_components)
        
                    return m.error == obj
        
                m.obj_const = Constraint(rule=rule_objective)
            return m  
        
        m.experiment = Block(self.experiments, rule = build_individual_blocks)
        count = 0
        m.dataset_count = list()
        m.map_exp_to_count = dict()
        m.first_exp = None
        for i in self.experiments:
            m.dataset_count.append(count)
            m.map_exp_to_count[count] = i
            if count == 0:
                m.first_exp = i
            count += 1
        def param_linking_rule(m, exp, param):
            prev_exp = None
            if exp == m.first_exp:

                return Constraint.Skip
            else:
                for key, val in m.map_exp_to_count.items():
                    print(key,val)
                    if val == exp:
                        prev_exp = m.map_exp_to_count[key-1]
                if param in global_params and prev_exp != None:
                    #This here is to check that the correct linking constraints are constructed
                    print("this constraint is written:")
                    print(m.experiment[exp].P[param],"=", m.experiment[prev_exp].P[param])
                    return m.experiment[exp].P[param] == m.experiment[prev_exp].P[param]
                    
                else:
                    return Constraint.Skip
            
        m.parameter_linking = Constraint(self.experiments, list_params_across_blocks, rule = param_linking_rule)
        
        m.obj = Objective(sense = minimize, expr=sum(b.error for b in m.experiment[:]))
        self.model = m
        
        if covariance and solver == 'k_aug' and self._spectra_given:
            #Not yet working
            
            self.solve_full_problem(solver, tee = tee, solver_opts = solver_opts)
            #solver_opts['compute_inv'] = ''
            
        elif covariance and solver == 'ipopt_sens' and self._spectra_given:
            #Not yet working
            
            self.solve_full_problem(solver, tee = tee, solver_opts = solver_opts)
            #solver_opts['compute_inv'] = ''    
            
        elif self._spectra_given:
            #Working
            optimizer = SolverFactory('ipopt')  
            solver_results = optimizer.solve(m, options = solver_opts,tee=tee)
        
        elif covariance and solver == 'k_aug' and self._concentration_given:   
            self.solve_conc_full_problem(solver, covariance = covariance, tee=tee)
            
        elif covariance and solver == 'ipopt_sens' and self._concentration_given:   
            self.solve_conc_full_problem(solver, covariance = covariance, tee=tee)     
            
        elif self._concentration_given:
            #Working
            optimizer = SolverFactory('ipopt')
            solver_results = optimizer.solve(m, options = solver_opts,tee=tee)
            
        solver_results = dict()   
        
        # loading the results, notice that we return a dictionary
        for i in m.experiment:
            solver_results[i] = ResultsObject()
            solver_results[i].load_from_pyomo_model(m.experiment[i],to_load=['Z', 'dZdt', 'X', 'dXdt', 'C', 'S', 'Y', 'P'])
        
        return solver_results
    
def split_sipopt_string1(output_string):
    start_hess = output_string.find('DenseSymMatrix')
    ipopt_string = output_string[:start_hess]
    hess_string = output_string[start_hess:]
    # print(hess_string, ipopt_string)
    return (ipopt_string, hess_string)
