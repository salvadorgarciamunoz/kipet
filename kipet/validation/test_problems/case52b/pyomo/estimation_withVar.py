#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem with concentration 
# example from WF paper simulation of ODE system using pyomo discretization and IPOPT
#
#		\frac{dZ_a}{dt} = -k_1*Z_a*Z_b	            Z_a(0) = 1
#		\frac{dZ_b}{dt} = -k_1*Z_a*Z_b               Z_b(0) = 1
#       \frac{dZ_c}{dt} = k_1*Z_a*Z_b	                Z_c(0) = 0
#        C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#        D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 


from kipet.library.TemplateBuilder import * 
from kipet.library.PyomoSimulator import *
from kipet.library.ParameterEstimator import *
from kipet.library.VarianceEstimator import *
import matplotlib.pyplot as plt

from kipet.library.data_tools import *
import inspect
import sys
import six
import os


if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False

   
            
    # read 200*431 spectra matrix D_{i,j}
    # this defines the measurement points t_i and l_j as well
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), '..','..','data_sets'))
    filename =  os.path.join(dataDirectory,'Dij_case52b.txt')
    D_frame = read_spectral_data_from_txt(filename)


    #D = D_frame.drop(D_frame.index[[0]])
    #D_array = np.array(D)
    #columns = D.columns
    #old_index = D.index
    #new_index = [i-old_index[0] for i in old_index]
    #new_D = pd.DataFrame(data=D_array,
    #                     columns=columns,
    #                     index=new_index)
 
    #write_spectral_data_to_txt('new_D.txt',new_D)
    #sys.exit()

 
    ######################################
    builder = TemplateBuilder()    
    components = {'A':217.324e-3,'B':167.35e-3,'C':2.452e-3}
    builder.add_mixture_component(components)

    # note the parameter is not fixed
    builder.add_parameter('k1',bounds=(0.0,1.0))
    builder.add_spectral_data(D_frame)

    # define explicit system of ODEs 
    def rule_odes2(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']*m.Z[t,'B']
        exprs['B'] = -m.P['k1']*m.Z[t,'A']*m.Z[t,'B']
        exprs['C'] = m.P['k1']*m.Z[t,'A']*m.Z[t,'B']
        return exprs
    
    builder.set_odes_rule(rule_odes2)

    pyomo_model2 = builder.create_pyomo_model(0.0,1077.9)

    #pyomo_model2.P['k1'].value = 0.006655
    #pyomo_model2.P['k1'].fixed = True
 
    v_estimator = VarianceEstimator(pyomo_model2)

    v_estimator.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')

    # Provide good initial guess
    p_guess = {'k1':0.006655}
    raw_results = v_estimator.run_lsq_given_P('ipopt',p_guess,tee=False)

    v_estimator.initialize_from_trajectory('Z',raw_results.Z)
    v_estimator.initialize_from_trajectory('S',raw_results.S)
    v_estimator.initialize_from_trajectory('dZdt',raw_results.dZdt)
    v_estimator.initialize_from_trajectory('C',raw_results.C)
    
    options = dict()
    A_set = [l for i,l in enumerate(pyomo_model2.meas_lambdas) if (i % 4 == 0)]
    results_variances = v_estimator.run_opt('ipopt',
                                            tee=True,
                                            solver_options=options,
                                            tolerance=1e-4,
                                            max_iter=40,
                                            subset_lambdas=A_set)

    print("\nThe estimated variances are:\n")
    for k, v in six.iteritems(results_variances.sigma_sq):
        print(k, v)

    print("The estimated parameters are:")
    for k,v in six.iteritems(pyomo_model2.P):
        print(k, v.value)
        
    sigmas = results_variances.sigma_sq
    optimizer = ParameterEstimator(pyomo_model2)

    optimizer.apply_discretization('dae.collocation',nfe=60,ncp=3,scheme='LAGRANGE-RADAU')

    # Provide good initial guess
    #p_guess = {'k1':0.006655}
    #raw_results = optimizer.run_lsq_given_P('ipopt',p_guess,tee=False)
    
    optimizer.initialize_from_trajectory('Z',results_variances.Z)
    optimizer.initialize_from_trajectory('S',results_variances.S)
    optimizer.initialize_from_trajectory('C',results_variances.C)

    
    #CheckInstanceFeasibility(pyomo_model2, 1e-3)
    # dont push bounds i am giving you a good guess
    solver_options = dict()
    #solver_options['bound_relax_factor'] = 0.0
    #solver_options['mu_init'] =  1e-6
    #solver_options['bound_push'] = 1e-5
    #solver_options['mu_strategy'] = 'adaptive'
    # fixes the standard deaviations for now
    
    results_pyomo = optimizer.run_opt('ipopt',
                                      tee=True,
                                      solver_opts = solver_options,
                                      variances=sigmas,
                                      with_d_vars=True)

    print("The estimated parameters are:")
    for k,v in six.iteritems(results_pyomo.P):
        print(k, v)

    #tol = 2e-1
    #assert(abs(results_pyomo.P['k1']-2.0)<tol)
        
    # display results
    if with_plots:
        results_pyomo.C.plot.line(legend=True)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")


        results_pyomo.S.plot.line(legend=True)
        plt.xlabel("Wavelength (cm)")
        plt.ylabel("Absorbance (L/(mol cm))")
        plt.title("Absorbance  Profile")

        plt.show()


