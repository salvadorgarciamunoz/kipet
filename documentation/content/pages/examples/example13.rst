Using the Alternative Variance Estimation Method
--------------------------------------------------------------
:Files:
    `Ex_13_alternate_method_variances.py <https://github.com/kwmcbride/kipet_examples/blob/master/examples/example_13/Ex_13_alternate_method_variances.py>`_

Since the above method that was used in the other problems, described in the initial paper from Chen et al. (2016), can be problematic for certain problems, new variance estimation procedures have been developed and implemented in KIPET. In these new variance estimation strategies, we solve the maximum likelihood problems directly. The first method, described in the introduction in :ref:`section 3` involves first solving for the overall variance in the problem and then solving iteratively in order to find how much of that variance is found in the model and how much is found in the device.
::

	import kipet

	r1 = kipet.ReactionModel('reaction-1')

    # Add the parameters
    k1 = r1.parameter('k1', value=1.2, bounds=(0.01, 5.0))
	k2 = r1.parameter('k2', value=0.2, bounds=(0.001, 5.0))
    
	# Declare the components and give the initial values
	A = r1.component('A', value=1.0e-3)
	B = r1.component('B', value=0.0)
	C = r1.component('C', value=0.0)
   
	# Define the reaction model
	r1.add_ode('A', -k1 * A )
	r1.add_ode('B', k1 * A - k2 * B )
	r1.add_ode('C', k2 * B )
    
	# Add data (after components)
	r1.add_data(category='spectral', file='data/varest.csv', remove_negatives=True)

	# Settings
	r1.settings.variance_estimator.tolerance = 1e-10
	r1.settings.parameter_estimator.tee = False
	r1.settings.parameter_estimator.solver = 'ipopt_sens'

After setting the problem up in the normal way, we then call the variance estimation routine with a number of new options that help to inform this new technique. 
::
    
    r1.settings.variance_estimator.method = 'alternate'
    r1.settings.variance_estimator.secant_point = 5e-4
    r1.settings.variance_estimator.initial_sigmas = 5e-5
    
The new options include the method, which in this case is ‘alternate’, initial_sigmas, which is our initial value for the sigmas that we wish to start searching from, and the secant_point, which provides a second point for the secant method to start from. The final new option is the individual_species option. When this is set to False, we will obtain only the overall model variance, and not the specific species. Since the problem is unbounded when solving for this objective function, if you wish to obtain the individual species’ variances, this can be set to True, however this should be used with caution as this is most likely not the real optimum, as the device variance that is used will not be the true value, as the objective functions are different.

::
    
	# Perform parameter fitting
	r1.run_opt()
    
	# Display the results
	r1.results.show_parameters
	r1.plot()

.. figure:: ../../images/ex_13_C.svg
   :width: 600px
   :align: center

   Concentration profiles for the alternate variance method

.. figure:: ../../images/ex_13_S.svg
   :width: 600px
   :align: center

   Single species absorbance profiles for the alternate variance method


Included in this tutorial problem is the ability to compare solutions with the standard Chen approach as well as to compare the solutions to the generated data. One can see that both approaches do give differing solutions. And that, in this case, the new variance estimator gives superior solutions.