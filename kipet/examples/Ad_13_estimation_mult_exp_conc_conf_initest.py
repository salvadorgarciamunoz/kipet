#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with estimating initial values for complementary states as local parameters of concentration data using pyomo discretization
# Including different initialization strategies
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
from kipet.library.MultipleExperimentsEstimator import *
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
       
    # Load concentration data from the relevant file location. As described in section 4.3.1
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename =  os.path.join(dataDirectory,'Ex_1_C_data_withoutA.csv')
    C_frame1 = read_concentration_data_from_csv(filename)
    C_frame2 = add_noise_to_signal(C_frame1, 0.0001)

    #################################################################################    
    builder = TemplateBuilder()
    builder2 = TemplateBuilder()
    components = {'B':0,'C':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1', init=1.0, bounds=(0.00,10)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder.add_parameter('k2',init = 0.224, bounds=(0.0,10))

    builder2.add_mixture_component(components)
    builder2.add_parameter('k1', init=1.0, bounds=(0.00, 10))
    # There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder2.add_parameter('k2', init=0.224, bounds=(0.0, 10))
    #add complementary state variable:
    extra_states1 = dict()
    extra_states1['A'] = 1e-3
    builder.add_complementary_state_variable(extra_states1)

    extra_states2 = dict()
    extra_states2['A2'] = 1e-3
    builder2.add_complementary_state_variable(extra_states2)
    
    # If you have multiple experiments, you need to add your experimental datasets to a dictionary:
    datasets = {'Exp1': C_frame1, 'Exp2': C_frame2}
    #Notice that we do not add the data to the model as we did in the past, rather we pass this as 
    #an argument into the function later on.
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.X[t,'A']
        exprs['B'] = m.P['k1']*m.X[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs

    def rule_odes2(m,t):
        exprs = dict()
        exprs['A2'] = -m.P['k1']*m.X[t,'A2']
        exprs['B'] = m.P['k1']*m.X[t,'A2']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    builder.set_odes_rule(rule_odes)
    builder2.set_odes_rule(rule_odes2)
    start_time1=0.0
    start_time2=0.0
    end_time1=10.0
    end_time2=10.0

    modelexp1 = builder.create_pyomo_model(start_time1, end_time1)

    builder22 = TemplateBuilder()
    builder22.add_mixture_component(components)
    builder22.add_parameter('k1', init=1.0, bounds=(0.0, 10))
    # There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder22.add_parameter('k2', init=0.224, bounds=(0.0, 10))
    # add complementary state variable:
    builder22.add_complementary_state_variable(extra_states2)
    builder22.set_odes_rule(rule_odes2)
    modelexp2 = builder22.create_pyomo_model(start_time2, end_time2)


    #Define our start and end times for the experiment, as we had done in the past, now related to
    #the dataset names
    initextra ={'Exp1':'A', 'Exp2':'A2'}
    start_time = {'Exp1':start_time1, 'Exp2':start_time2}
    end_time = {'Exp1':end_time1, 'Exp2':end_time2}
    builder_dict = {'Exp1':builder, 'Exp2':builder22}
    
    options = dict()
    options['linear_solver'] = 'ma57'
    #options['mu_init']=1e-6
    
    # ============================================================================
    #   USER INPUT SECTION - MULTIPLE EXPERIMENTAL DATASETS       
    # ===========================================================================
    # Here we use the class for Multiple experiments, notice that we add the dictionary
    # Containing the datasets here as an argument
    pest = MultipleExperimentsEstimator(datasets)
    
    nfe = 60
    ncp = 3

    sigmas1 = {'device':1e-10,'A':1e-10,'B':1e-10,'C':1e-10}
    sigmas2 = {'device':1e-10,'A2':1e-10,'B':1e-10,'C':1e-10}
    
    variances = {'Exp1':sigmas1, 'Exp2':sigmas2}

    # Different methods for initialization:
    #######################################
    #1)Initialize from previous solution:
    # resultestY1 = read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestY1.csv'))
    # resultestC1 = read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestC1.csv'))
    # resultestZ1 = read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestZ1.csv'))
    # resultestdZdt1 = read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestdZdt1.csv'))
    # resultestX1 = read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestX1.csv'))
    # resultestY2 = read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestY2.csv'))
    # resultestC2 = read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestC2.csv'))
    # resultestZ2 = read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestZ2.csv'))
    # resultestdZdt2 = read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestdZdt2.csv'))
    # resultestX2 = read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestX2.csv'))
    #
    # # resultestY1 = add_noise_to_signal(read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestY1.csv')),0.0001)
    # # resultestC1 = add_noise_to_signal(read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestC1.csv')),0.0001)
    # # resultestZ1 = add_noise_to_signal(read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestZ1.csv')),0.0001)
    # # resultestdZdt1 = add_noise_to_signal(read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestdZdt1.csv')),0.0001)
    # # resultestX1 = add_noise_to_signal(read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestX1.csv')),0.0001)
    # # resultestY2 = add_noise_to_signal(read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestY2.csv')),0.0001)
    # # resultestC2 = add_noise_to_signal(read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestC2.csv')),0.0001)
    # # resultestZ2 = add_noise_to_signal(read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestZ2.csv')),0.0001)
    # # resultestdZdt2 = add_noise_to_signal(read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestdZdt2.csv')),0.0001)
    # # resultestX2 = add_noise_to_signal(read_concentration_data_from_csv(os.path.join(dataDirectory, 'resultestX2.csv')),0.0001)
    #
    # resultestY = dict()
    # resultestY = {'Exp1': resultestY1, 'Exp2': resultestY2}
    # resultestZ = dict()
    # resultestZ = {'Exp1': resultestZ1, 'Exp2': resultestZ2}
    # resultestC = dict()
    # resultestC = {'Exp1': resultestC1, 'Exp2': resultestC2}
    # resultestX = dict()
    # resultestX = {'Exp1': resultestX1, 'Exp2': resultestX2}
    # resultestdZdt = dict()
    # resultestdZdt = {'Exp1': resultestdZdt1, 'Exp2': resultestdZdt2}
    #
    # #Have to be added before parameter estimation is called!
    # initextra = {'A'}  # to define to be estimated initial values for extra states
    # builder.add_init_extra('A', init=1e-3, bounds=(0.0, 0.1))
    # builder.set_estinit_extra_species(modelexp1, initextra)  # to define to be estimated initial values for extra states
    #
    # initextra2 = {'A2'}
    # builder22.add_init_extra('A2', init=1e-3, bounds=(0.0, 0.1))
    # builder22.set_estinit_extra_species(modelexp2, initextra2)  # to define to be estimated initial values for extra states
    #
    #
    # #Define our start and end times for the experiment, as we had done in the past, now related to
    # #the dataset names
    # initextra ={'Exp1':'A', 'Exp2':'A2'}
    # start_time = {'Exp1':start_time1, 'Exp2':start_time2}
    # end_time = {'Exp1':end_time1, 'Exp2':end_time2}
    # builder_dict = {'Exp1':builder, 'Exp2':builder22}
    #
    # results_pest = pest.run_parameter_estimation(builder=builder_dict,
    #                                              solver='ipopt_sens',
    #                                              tee=False,
    #                                              nfe=nfe,
    #                                              ncp=ncp,
    #                                              spectra_problem=False,
    #                                              init_files=True,
    #                                              resultY=resultestY,
    #                                              resultZ=resultestZ,
    #                                              resultC=resultestC,
    #                                              resultdZdt=resultestdZdt,
    #                                              resultX=resultestX,
    #                                              sigma_sq=variances,
    #                                              solver_opts=options,
    #                                              covariance=True,
    #                                              start_time=start_time,
    #                                              end_time=end_time)
    ##################################################

    #2)Initialize from simulation with different options: Using PyomoSimulator, FESimulator or both.
    results_sim = pest.run_simulation(builder = builder_dict,
                                                         solver = 'ipopt',
                                                         tee=False,
                                                         nfe=nfe,
                                                         ncp=ncp,
                                                         sigma_sq = variances,
                                                         FEsim=True,#This is an additional option. By default PyomoSimulator is called.
                                                         solver_opts = options,
                                                         start_time=start_time,
                                                         end_time=end_time)
    #############################################################################
    #These additional ones have to be declared after the initial simulations and added to the TemplateBuilder before the parameter estimation!!!
    initextra = {'A'}  # to define to be estimated initial values for extra states
    builder.add_init_extra('A', init=1e-3, bounds=(0.0, 0.1))
    builder.set_estinit_extra_species(modelexp1, initextra)  # to define to be estimated initial values for extra states

    initextra2 = {'A2'}
    builder22.add_init_extra('A2', init=1e-3, bounds=(0.0, 0.1))
    builder22.set_estinit_extra_species(modelexp2, initextra2)  # to define to be estimated initial values for extra states


    #Define our start and end times for the experiment, as we had done in the past, now related to
    #the dataset names
    initextra ={'Exp1':'A', 'Exp2':'A2'}
    start_time = {'Exp1':start_time1, 'Exp2':start_time2}
    end_time = {'Exp1':end_time1, 'Exp2':end_time2}
    builder_dict = {'Exp1':builder, 'Exp2':builder22}

    # Finally we run the parameter estimation. This solves each dataset separately first and then
    # links the models and solves it simultaneously
    results_pest = pest.run_parameter_estimation(solver = 'k_aug', #or ipopt_sens #for k_aug new version of k_aug needs to be installed
                                                        tee=True,
                                                         nfe=nfe,
                                                         ncp=ncp,
                                                         covariance = True,
                                                         solver_opts = options,
                                                         start_time=start_time,
                                                         end_time=end_time,
                                                         spectra_problem = False,
                                                         sigma_sq=variances,
                                                         builder = builder_dict)
    
    # Note here, that with the multiple datasets, we are returning a dictionary containing the
    # results for each block. Since we know that all parameters are shared, we only need to print
    # the parameters from one experiment, however for the plots, they could differ between experiments

    #To save results:
    # for k,v in results_pest.items():
    #     if k=='Exp1':
    #         write_concentration_data_to_csv(os.path.join(dataDirectory, 'resultestC1res.csv'), results_pest[k].C)
    #         write_concentration_data_to_csv(os.path.join(dataDirectory, 'resultestX1res.csv'), results_pest[k].X)
    #         write_concentration_data_to_csv(os.path.join(dataDirectory, 'resultestY1res.csv'), results_pest[k].Y)
    #         write_concentration_data_to_csv(os.path.join(dataDirectory, 'resultestZ1res.csv'), results_pest[k].Z)
    #         write_concentration_data_to_csv(os.path.join(dataDirectory, 'resultestdZdt1res.csv'), results_pest[k].dZdt)
    #     elif k=='Exp2':
    #         write_concentration_data_to_csv(os.path.join(dataDirectory, 'resultestC2res.csv'), results_pest[k].C)
    #         write_concentration_data_to_csv(os.path.join(dataDirectory, 'resultestX2res.csv'), results_pest[k].X)
    #         write_concentration_data_to_csv(os.path.join(dataDirectory, 'resultestY2res.csv'), results_pest[k].Y)
    #         write_concentration_data_to_csv(os.path.join(dataDirectory, 'resultestZ2res.csv'), results_pest[k].Z)
    #         write_concentration_data_to_csv(os.path.join(dataDirectory, 'resultestdZdt2res.csv'), results_pest[k].dZdt)

    print("The estimated parameters are:")

    for k,v in results_pest.items():
        print(results_pest[k].P)

    for k,v in results_pest.items():
        print(results_pest[k].Pinit)
    
    if with_plots:
        for k,v in results_pest.items():

            results_pest[k].X.plot.line(legend=True)
            plt.xlabel("time (s)")
            plt.ylabel("Concentration (mol/L)")
            plt.title("Concentration Profile")
    
            results_pest[k].Z.plot.line(legend=True)
            plt.xlabel("time (s)")
            plt.ylabel("Concentration (mol/L)")
            plt.title("Concentration Profile")
        
            plt.show()
            
