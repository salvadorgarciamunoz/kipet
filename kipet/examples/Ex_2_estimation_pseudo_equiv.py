
#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with unknow variancesof spectral data using pyomo discretization 
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2*Z_b	                Z_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 
#       Initial concentration 

from __future__ import print_function
from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.data_tools import *
import matplotlib.pyplot as plt
import os
import sys
import inspect
import six

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
        
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================
    
    # Load spectral data from the relevant file location. As described in section 4.3.1
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename =  os.path.join(dataDirectory,'Dij.txt')
    D_frame = read_spectral_data_from_txt(filename)

    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_spectral_data
    #################################################################################    
    builder = TemplateBuilder()    
    components = {'A':1e-3,'B':0,'C':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1', init=2.0, bounds=(0.05,5.0)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder.add_parameter('k2',init = 0.2, bounds=(0.0,1.0))
    builder.add_spectral_data(D_frame)

    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    builder.set_odes_rule(rule_odes)
    opt_model = builder.create_pyomo_model(0.0,10.0)

    #=========================================================================
    #USER INPUT SECTION - PRE-PROCESSING
    #=========================================================================
    # to determine the species present and whether we have unwanted contributions
    # we can represent our problem in a different way: the pseudo-equivalency matrix
    # First we define the matrix using the stoichiometry with columns representing 
    # Each component and the first rows being each reaction:
    #             A , B , C
    reaction1 = [-1 , 1 , 0]
    reaction2 = [ 0 , -1, 1]
    # Next we define the initial conditions
    initial =   [1e-3,0 , 0]
    
    # In the case of inputs we would add those as extra rows, with the time
    # of input being the row, and the components fed being the columns
    
    # Before we can perform the analysis on the pseudo-equivalency matrix we need to 
    # first determine the number of components that are likely to be absorbing.
    # We can do this by performing PCA and carfeully selecting and weighing up
    # the pros and cons of each potential number of components that we can select
    # Our advice is to first perform the PCA with a large number of components to 
    # see where the scree plot has an elbow. If there is a clear elbow, as there is 
    # this example, then it is advised to select this number of components

    basic_pca(D_frame,n=100,with_plots=True)
    #After performing the PCA we can see that we should use 3 components
    
    # Now we would like to determine the rank of this matrix (which will be 3):
    rank_data = rank(D_frame, eps = 1e-1)
    # Here, eps represents the cut-off for the singular values from our plot. This
    # Will differ from problem to problem and needs to be chosen based on the scree plot    
    print(rank_data)

    # Now that we have defined out pseudo-equivalency matrix rows we can perform:
    perform_data_analysis(D_frame, [reaction1,reaction2,initial], rank_data)  
    
    # The output from this should give you whether the number of chosen components 
    # and the number of absorbing components match and whether it may be necessary to include
    # unwanted contributions or not
    sys.exit()