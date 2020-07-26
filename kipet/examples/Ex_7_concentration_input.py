"""
Kipet: Kinetic parameter estimation toolkit
Copyright (c) 2016 Eli Lilly.
 
Example 7 - Concentration as input

"""
import matplotlib.pyplot as plt
import os
import sys
import pathlib
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
    # Load spectral data from the relevant file location. As described in section 4.3.1
    #################################################################################
    data_dir = Path.cwd().joinpath('data_sets')
    filename = data_dir.joinpath(Path('issue_data', 'missing_data.txt'))
    # New generic data read method
    C_frame = read_file(filename)
    
    # Then we build dae block for as described in the section 4.2.1. Note the addition
    # of the data using .add_spectral_data
    #################################################################################    
    builder = TemplateBuilder()    
    components = {'A':1e-3,'B':0,'C':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1',bounds=(0.0,5.0))
    builder.add_parameter('k2',bounds=(0.0,1.0))
    builder.add_concentration_data(C_frame)

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
    #write_concentration_data_to_csv('Ex_1_C_data.csv',C_Dataframe)
        
    # display results
    if with_plots:
        make_plot(results_pyomo, 'C')