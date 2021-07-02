Parameter Estimation Using Multiple Datasets
--------------------------------------------
:Files:
    | `Ex_11_multiple_experiments_spectral.py <https://github.com/kwmcbride/kipet_examples/blob/master/examples/example_11/Ex_11_multiple_experiments_spectral.py>`_
    | `Ex_12_multiple_experiments_concentration.py <https://github.com/kwmcbride/kipet_examples/blob/master/examples/example_12/Ex_12_multiple_experiments_concentration.py>`_

KIPET allows for the estimation of kinetic parameters with multiple experimental datasets through the MultipleExperimentsEstimator class. This is handled automatically and takes place when the KipetModel instance contains more than one ReactionModel instance in its models attibute. See the example code below for an overview.

Internally, this procedure is performed by running the VarianceEstimator (optionally) over each dataset, followed by ParameterEstimator on individual models. After the local parameter estimation has been performed, the code blocks are used to initialize the full parameter estimation problem. The algorithm automatically detects whether parameters are shared across experiments based on their names within each model. Note that this procedure can be fairly time-consuming. In addition, it may be necessary to spend considerable time tuning the solver parameters in these problems, as the system involves the solution of large, dense linear systems in a block structure linked via equality constraints (parameters). It is advised to try different linear solver combinations with various IPOPT solver options if difficulty is found solving these. The problems may also require large amounts of RAM, depending on the size.

The `example considered here <https://github.com/kwmcbride/kipet_examples/blob/master/examples/example_12/Ex_12_multiple_experiments_concentration.py>`_ involves two concentration datasets for the same reaction. The second reaction model uses only a sampling on points taken from the first dataset but with added noise.

::

    import kipet
 	
	lab = kipet.ReactionSet()
    
    r1 = lab.new_reaction(name='reaction-1')
    
    # Add the parameters
    k1 = r1.parameter('k1', value=1.0, bounds=(0.0, 10.0))
    k2 = r1.parameter('k2', value=0.224, bounds=(0.0, 10.0))
   
    # Declare the components and give the initial values
    A = r1.component('A', value=1.0e-3)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0)
    
    # define explicit system of ODEs
    rA = k1*A
    rB = k2*B
    
    # Define the reaction model
    r1.add_ode('A', -rA )
    r1.add_ode('B', rA - rB )
    r1.add_ode('C', rB )
   
    # Add the dataset for the first model
    r1.add_data('C_data', file='data/Ex_1_C_data.txt')
    
    # Add the known variances
    r1.variances = {'A':1e-10,'B':1e-10,'C':1e-10}
    
    # Declare the second model (based on first model)
    r2 = lab.new_reaction(name='reaction-2', model=r1)
   
    # Add the dataset for the first model
    noised_data = kipet.add_noise_to_data(r1.datasets['C_data'].data, 0.0001) 
    r2.add_data('C_data', data=noised_data[::10])
    
    # Add the known variances
    r2.components.update('variance', {'A':1e-4,'B':1e-4,'C':1e-4})
   
    # Create the MultipleExperimentsEstimator and perform the parameter fitting
    lab.run_opt()

    # Plot the results
    lab.show_parameters
    lab.plot()
    
This outputs the following:
::

    The estimated parameters are:
    k1 0.22638509313022112
    k2 1.0031160083691573

.. figure:: ../../images/ex_12_C1.svg

   :width: 600px
   :align: center

   Concentration profiles for the first experiment

.. figure:: ../../images/ex_12_C2.svg

   :width: 600px
   :align: center

   Concentration profiles for the second experiment

There are a number of other examples showing how to implement the multiple experiments across different models with shared global and local parameters as well as how to obtain confidence intervals for the problems.
It should be noted that obtaining confidence intervals can only be done when declaring a global model, as opposed to different models in each block. This is due to the construction of the covariance matrices. When obtaining confidence intervals for multiple experimental datasets it is very important to ensure that the solution obtained does not include irrationally large absorbances (from species with low or no concentration) and that the solution of the parameters is not at very close to a bound. This will cause the sensitivity calculations to be aborted, or may result in incorrect confidence intervals.
All the additional problems demonstrating various ways to obtain kinetic parameters from different experimental set-ups are shown in the example table and included in the folder with tutorial examples.