"""
Kipet: Kinetic parameter estimation toolkit
Copyright (c) 2016 Eli Lilly.
 
Example 7 - Concentration as input with missing or incongruent data
This also shows some other new features of Kipet as well

"""
import inspect
import os
import sys
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from kipet.library.TemplateBuilder import *
from kipet.library.ParameterEstimator import *
from kipet.library.data_tools import *
from kipet.library.common.charts import make_plot

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================
    data_file = 'missing_data.txt'
    directory = 'data_sets'
    
    dataDirectory = os.path.abspath(
    os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
        inspect.currentframe() ) ) ), directory))
    filename =  os.path.join(dataDirectory, data_file)
    
    # New generic data read method
    C_frame = read_file(filename)
    
    # Also with optional directory if not in the default
    #C_frame = read_file(filename, directory='data_sets')

    # Make the template
    builder = TemplateBuilder()
    
    # Option to clear all data from the template
    builder.clear_data()
    
    # Add individual species to the template
    builder.add_concentration_data(C_frame['A'])
    print(f'\n\nThis is in the TB after A\n\n: {builder._concentration_data}')
    
    # Add additional species - use overwrite=False to add - default is to overwrite
    #builder.add_concentration_data(C_frame['B'], overwrite=False)
    builder.add_concentration_data(C_frame['C'], overwrite=False)
    print(builder._concentration_data)
    print(f'\n\nThis is in the TB after B\n\n: {builder._concentration_data}')
    
    #This is the same as adding the data frame using add_experimental_data:
    # NOTE: This can only be used after adding components to the template!
    #builder.clear_data()
    #builder.add_experimental_data(C_frame)
    print(builder._concentration_data)
    
    # You can also add the entire data frame using add_concentration_data
    #builder.clear_data()
    #builder.add_concentration_data(C_frame)
    print(builder._concentration_data)
    
    # I want to remove the other methods
    # and use a single generic method to add data - add the type as arg (or not)
    #builder.clear_data()
    #builder.add_data(C_frame, 'concentration')
    print(builder._concentration_data)
    
    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_spectral_data
    #################################################################################    
    components = {'A':1e-3,'B':0,'C':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1',bounds=(0.0,5.0))
    builder.add_parameter('k2',bounds=(0.0,1.0))
    
    # New generic method to add data - add the type as arg (or not)
    #builder.add_data(C_frame) #, 'concentration')

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
    #USER INPUT SECTION - VARIANCE GIVEN
    #=========================================================================
    sigmas = {'A':1e-10,'B':1e-11,'C':1e-10}
    
    #=========================================================================
    # USER INPUT SECTION - PARAMETER ESTIMATION 
    #=========================================================================
    # In order to run the paramter estimation we create a pyomo model as described in section 4.3.4

    # and define our parameter estimation problem and discretization strategy
    p_estimator = ParameterEstimator(opt_model)
    p_estimator.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')
    
    # Again we provide options for the solver, this time providing the scaling that we set above
    options = dict()
#    options['nlp_scaling_method'] = 'user-scaling'

    # finally we run the optimization
    results_pyomo = p_estimator.run_opt('ipopt',
                                        variances=sigmas,
                                        tee=True,
                                        solver_opts = options)

    # And display the results
    print("The estimated parameters are:")
    for k, v in results_pyomo.P.items():
        print(k, v)

    #C_Dataframe = pd.DataFrame(data=results_pyomo.C)
    #write_file('Ex_1_C_data.csv',C_Dataframe)
    
    
    # New method for making plots simpler, look at the func for some options
    if with_plots:
        make_plot(results_pyomo, 'C')