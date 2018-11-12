#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Parameter estimation with fixed variances of concentration data with inputs using pyomo discretization and fe-factory


from kipet.library.TemplateBuilder import *
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
from kipet.library.FESimulator import *
from kipet.library.data_tools import *
import matplotlib.pyplot as plt

import inspect
import sys
import os
import six

if __name__ == "__main__":
    
    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
    
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================

    ######################################  
     # create template model 
    species = {'A':6.7, 'B':20.2, 'C':0.0}
    params = {'k_p':3.734e7}

    builder = TemplateBuilder()   

    builder = TemplateBuilder(concentrations=species,
                              parameters=params)    
    #extra_states=dict()
    #extra_states['T']=290.0
    builder.add_complementary_state_variable('T',290.0)

    # define explicit system of ODEs
    def rule_odes(m,t):
        r = m.P['k_p']*exp(-15400.0/(1.987*m.X[t,'T']))*m.Z[t,'A']*m.Z[t,'B']
        T1 = 45650.0*(r*0.01)/28.0
        T2 = 1+(328.0-m.X[t,'T'])/((328.0-m.X[t,'T'])**2+1e-5**2)**0.5
        exprs = dict()
        exprs['A'] = -r
        exprs['B'] = -r
        exprs['C'] = r
        exprs['T'] = T1+T2
        return exprs

    builder.set_odes_rule(rule_odes)
    
    # create an instance of a pyomo model template
    # the template includes
    #      - Z variables indexed over time and components names e.g. m.Z[t,'A']
    #      - P parameters indexed over the parameter names e.g. m.P['k']
    feed_times=[2.51, 10.0, 15.0]#set with declared feeding points
    builder.add_feed_times(feed_times)#have to add feed times first!

    # Load concentration data
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename =  os.path.join(dataDirectory,'Ad5_C_data_input_noise2.csv')
    C_frame = read_concentration_data_from_csv(filename)
    builder.add_concentration_data(C_frame) #has to be added here already to use for initialization as well
    pyomo_model = builder.create_pyomo_model(0.0,20.0)

    #=========================================================================
    #USER INPUT SECTION - FE Factory
    #=========================================================================
     
    # call FESimulator
    # FESimulator re-constructs the current TemplateBuilder into fe_factory syntax
    # there is no need to call PyomoSimulator any more as FESimulator is a child class 
    sim = FESimulator(pyomo_model)
    
    # defines the discrete points wanted in the concentration profile
    sim.apply_discretization('dae.collocation', nfe=40, ncp=3, scheme='LAGRANGE-RADAU')

    #New Inputs for discrete feeds
    Z_step = {'A': 0.5, 'C': 1.0} #Which component and which amount is added
    X_step = {'T': 10.0}
    jump_states = {'Z': Z_step, 'X': X_step}
    jump_points1 = {'A': 15.0, 'C': 2.51}#Which component is added at which point in time
    jump_points2 = {'T': 10.0}
    jump_times = {'Z': jump_points1, 'X': jump_points2}


    init = sim.call_fe_factory(jump_states=jump_states, jump_times=jump_times, feed_times=feed_times)
    #They should be added in this way in case some arguments of all the ones available are not necessary here.

#=========================================================================
    #USER INPUT SECTION - SIMULATION
#========================================================================
    #: now the final run, just as before
    # simulate
    options = {}
    # options = { 'acceptable_constr_viol_tol': 0.1, 'constr_viol_tol':0.1,# 'required_infeasibility_reduction':0.999999,
    #                'print_user_options': 'yes'}#,
    results_pyomo = sim.run_sim('ipopt',
                          tee=True,
                          solver_opts=options)

    if with_plots:
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile Without Noise")

        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile with Noise ")
    
        plt.show()    
        
#=========================================================================
    #USER INPUT SECTION - Parameter Estimation
#========================================================================
    #Here with fixed variances!
    sigmas={'C': 1e-10,
            'B': 1e-10,
            'A': 1e-10,
            'device': 1e-10}

    #Now introduce parameters as non fixed
    pyomo_model.del_component(params)
    builder.add_parameter('k_p',bounds=(0.0,8e7))
    model = builder.create_pyomo_model(0.0,20.0)
    p_estimator = ParameterEstimator(model)
    p_estimator.apply_discretization('dae.collocation', nfe=40, ncp=3, scheme='LAGRANGE-RADAU')


    # # Again we provide options for the solver
    options = dict()
    #p_estimator.initialize_from_trajectory('Z', results.Z)
    #p_estimator.initialize_from_trajectory('S', results.S)
    #p_estimator.initialize_from_trajectory('dZdt', results.dZdt)
    #p_estimator.initialize_from_trajectory('C', results.C)
    # finally we run the optimization
    
    results_pyomo = p_estimator.run_opt('ipopt_sens',
                                        tee=True,
                                        solver_opts=options,
                                        variances=sigmas,
                                        covariance=True,
                                        #inputs_sub=inputs_sub,
                                        #trajectories=trajs,
                                        jump=True,
                                        jump_times=jump_times,
                                        jump_states=jump_states,
                                        feed_times=feed_times#has to be added here for error checks in ParameterEstimator
                                        #fixedy=True,
                                        #fixedtraj=True,
                                        #yfix=yfix,
                                        #yfixtraj=yfixtraj
                                        )
    #The commented arguments are available as well but not necessary here.

    
    print("The estimated parameters are:")
    for k,v in six.iteritems(results_pyomo.P):
        print(k, v)

    # display results
    if with_plots:
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")

        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
    
        plt.show()
