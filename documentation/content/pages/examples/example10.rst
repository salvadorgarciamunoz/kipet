Parameter Estimability Analysis
-------------------------------
:Files:
    | `Ex_8_estimability.py <https://github.com/kwmcbride/kipet_examples/blob/master/examples/example_8/Ex_8_estimability.py>`_
    | `Ex_9_estimability_with_problem_gen.py  <https://github.com/kwmcbride/kipet_examples/blob/master/examples/example_9/Ex_9_estimability_with_problem_gen.py>`_

The EstimabilityAnalyzer module is used for all algorithms and tools pertaining to estimability. Thus far, estimability analysis tools are only provided for cases where concentration data is available. The methods rely on k_aug to obtain sensitivities, so will only work if k_aug is installed and added to path.

::

    import kipet

    r1 = kipet.ReactionModel('reaction-1')

    # Add the model parameters
    k1 = r1.parameter('k1', bounds=(0.1,2))
    k2 = r1.parameter('k2', bounds=(0.0,2))
    k3 = r1.parameter('k3', bounds=(0.0,2))
    k4 = r1.parameter('k4', bounds=(0.0,2))
    
    # Declare the components and give the initial values
    A = r1.component('A', value=0.3)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0)
    D = r1.component('D', value=0.01)
    E = r1.component('E', value=0.0)
    
    filename = 'data/new_estim_problem_conc.csv'
    r1.add_data('C_frame', file=filename) 
    
    r1.add_ode('A', -k1*A - k4*A )
    r1.add_ode('B',  k1*A - k2*B - k3*B )
    r1.add_ode('C',  k2*B - k4*C )
    r1.add_ode('D',  k4*A - k3*D )
    r1.add_ode('E',  k3*B )
    
    r1.set_times(0, 20)

    param_uncertainties = {'k1':0.09,'k2':0.01,'k3':0.02,'k4':0.5}
    # sigmas, as before, represent the variances in regard to component
    sigmas = {'A':1e-10,'B':1e-10,'C':1e-11, 'D':1e-11,'E':1e-11,'device':3e-9}
    # measurement scaling
    meas_uncertainty = 0.05
    
    params_fit, params_fix = r1.analyze_parameters(method='yao',
                                 parameter_uncertainties=param_uncertainties,
                                 meas_uncertainty=meas_uncertainty,
                                 sigmas=sigmas)


The algorithm for parameter ranking requires the definition by the user of the confidences in the parameter initial guesses, as well as measurement device error in order to scale the sensitivities obtained. In order to run the full optimization problem, the variances for the model are also still required, as in previous examples.
:: 
   
    param_uncertainties = {'k1':0.09,'k2':0.01,'k3':0.02,'k4':0.5}
    sigmas = {'A':1e-10,'B':1e-10,'C':1e-11, 'D':1e-11,'E':1e-11,'device':3e-9}
    meas_uncertainty = 0.05
    
The parameter ranking algorithm from Yao, et al. (2003) needs to be applied first in order to supply a list of parameters that are ranked. This algorithm ranks parameters using a sensitivity matrix computed from the model at the initial parameter values (in the middle of the bounds automatically, or at the initial guess provided the user explicitly).  This function is only applicable to the case where you are providing concentration data, and returns a list of parameters ranked from most estimable to least estimable. Once these scalings are defined we can call the ranking function:
    
This function returns the parameters in order from most estimable to least estimable. Finally we can use these ranked parameters to perform the estimability analysis methodology suggested by Wu, et al. (2011) which uses an algorithm where a set of simplified models are compared to the full model and the model which provides the smallest mean squared error is chosen as the optimal number of parameters to estimate. This is done using:

This will return a list with only the estimable parameters returned. All remaining parameters (non-estimable) should be fixed at their most likely values.

For a larger example with more parameters and which includes the data generation, noising of data, as well as the application of the estimability to a final parameter estimation problem see `this example <https://github.com/kwmcbride/kipet_examples/blob/master/examples/example_9/Ex_9_estimability_with_problem_gen.py>`_.