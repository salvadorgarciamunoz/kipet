#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 08:27:42 2020

@author: kevin
"""

#%%
model = opt_model

A = model.C
B = model.Z

E = pyomo_squared_diff(A, B)
Ed = dict_squared_diff(A, B)

#%%
def pyomo_squared_diff(A, B, sigma=1, override_sets=None):
    
    if sigma <= 0 or sigma is None:
        sigma = 1
        print('Warning: invalid sigma provided; set to one.')
    
    # handle sigma sets
    
    if override_sets is not None:
        if len(override_sets) != 2:
            raise ValueError('You need to provide two sets for override.')
    
    expr_value = 0
    
    if override_sets is None:
     
        for index, v in A.items():
            expr_i = (A[index] - B[index])**2 / sigma**2
            expr_value += expr_i
    
    else:
        for ind1 in override_sets[0]:
            for ind2 in override_sets[1]:
                expr_i = (A[ind1, ind2] - B[ind1, ind2])**2 / sigma**2
        
    return expr_value
    
def dict_squared_diff(A, B, sigma=1, override_sets=None):
    
    if sigma <= 0 or sigma is None:
        sigma = 1
        print('Warning: invalid sigma provided; set to one.')
    
    # handle sigma sets
    
    if override_sets is not None:
        if len(override_sets) != 2:
            raise ValueError('You need to provide two sets for override.')
    
    residuals = {}
    
    if override_sets is None:
     
        for index, v in A.items():
            expr_i = (A[index].value - B[index].value)**2 / sigma**2
            residuals[index] = expr_i
    
    else:
        for ind1 in override_sets[0]:
            for ind2 in override_sets[1]:
                expr_i = (A[ind1, ind2].value - B[ind1, ind2].value)**2 / sigma**2
                residuals[ind1, ind2] = expr_i
            
    return residuals
    




#%%
