Additional Functions
====================

Along with the functions explained in the examples, KIPET also provides users with a host of other functions that can be used. In this section some of the additional functions provided in KIPET are shown and detailed, along with some the other subtleties that may be useful to the user in implementing and improving their models.

Data manipulation tools
-----------------------

KIPET provides a variety of data manipulation tools and the way in which data is inputted is extremely important in order to correctly pass the results of the user’s experiments onto KIPET. This section provides a further clarification on the types of files that KIPET is able to utilize, how the data should be arranged, as well as how to use KIPET to generate data.
 
Input matrices
^^^^^^^^^^^^^^

Loading data (D) matrices
~~~~~~~~~~~~~~~~~~~~~~~~~

When using KIPET for parameter estimation it is important that the data matrix, D, be inputted correctly. KIPET can read both text files (.txt) or Comma Separated Value files (.csv) from software such as Microsoft Excel or OpenOffice. Examples of data sets and how best to format them are included in the folder “Examples/data_sets”. Text files are best formulated with unlabeled columns with column 1 being time, column 2 being the wavelength, and column 3 being the absorbance associated with the specific wavelength. The method read_spectral_data_from_txt(filename.txt) automatically sorts the data and compiles it into a Pandas DataFrame for further manipulation by KIPET. Even though the order of the data is not important (except that rows need to be consistent), it is important to input the data as floating values and to separate the columns with a space.
When inputting a CSV file, it is necessary to label the columns with headings of the wavelength values and with row labels for the measuring times. The matrix will then be in the correct form with the absorption values being the entries. The read_spectral_data_from_csv(filename.csv) function is the function to call in this case. Additionally, if you have data directly outputted from an instrument as a CSV, there are some tools in this function that automatically turn timestamped data into seconds and also manipulate the matrix into KIPET’s preferred format. This is done through the use of an additional argument “instrument = True”. If the D matrix contains negative values it is also possible to automatically set these to zero using “negatives_to_zero = True”. This is not advised as it is better to either keep these negative values, or to remove them using a baseline shift or other pre-processing tool.

Loading pure component absorbance data (S)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to input S matrices. These can also be inputted in the form of CSV or txt files. In the example provided, “Ad_4_sdae_sim.py” a model is simulated given pure component absorption data. For CSVs the components label the columns and the wavelengths label the rows. The relevant absorption values fill the matrix. For text files the rows can be unordered, but must be column 1: wavelength, column 2: component name, column 3: absorption. To input this into KIPET we use a similar format as previously described:
::

    S_frame = read_absorption_data_from_txt(filename)

After this is formatted into the Pandas DataFrame using the above code, we will need to add the data to our model using:
::

    builder.add_absorption_data(S_frame)

If we plan to use this data to simulate a specific system and perhaps generate a data file (spectra) we will also need to add measurement times to our new model. In the example this is chosen as:
::

    builder.add_measurement_times([i*0.0333 for i in range(300)])

Loading concentration data (C)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If we wish to do parameter estimation for a problem where we have concentrations that are measured directly by laboratory instruments, this is possible within KIPET, as described in section 4.8 of this document. We can read data in either CSV or txt format using either:
::

    C_frame = read_concentration_data_from_txt(filename)

or if you have a csv file:
::

    C_frame = read_concentration_data_from_csv(filename)

Generating input matrices
^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to generate matrices using some of the built-in functions in KIPET.  One such function is able to generate pure-component absorbance data based on Lorentzian parameters:
::

    S_frame = generate_absorbance_data(wl_span,S_parameters)

Where wl_span is the wavelength span that you wish to generate the data for in the form of a vector where the first entry is the starting wavelength, second is the ending wavelength and the third is the step-length. S_parameters is a dictionary of parameters for the Lorentzian distribution function, ‘alphas’, ‘betas’, and ‘gammas’. This function will generate an absorbance profile DataFrame. 
It is also possible to generate a random absorbance data with the function:
::

    generate_random_absorbance_data(wl_span,component_peaks, component_widths = None, seed=None)

Where the wl_span is the same as above, component_peaks is the maximum height of the absorbance peak, component_widths is the maximum Lorentzian parameter for gamma, and seed is the possible seed for the random number generator.

Writing matrices to files
^^^^^^^^^^^^^^^^^^^^^^^^^

KIPET also provides functions to write generated matrices to a file with the following functions:
::

    write_absorption_data_to_csv(filename,dataframe)
    write_absorption_data_to_txt(filename,dataframe)

Where the user can define the filename and which DataFrame to input.
It is also possible that a user might wish to generate a spectral D-matrix from a file and then write this to a file. This can be done using:
::

    write_spectral_data_to_txt(filename,dataframe) 	
    write_spectral_data_to_csv(filename,dataframe)

Which takes in the same inputs as the other function, filename in the form of a string to be used as the output file name and dataframe which is the pandas DataFrame that is to be written.
The same data_tools exist for C-matrices:
::

    write_concentration_data_to_txt(filename,dataframe) 	
    write_concentration_data_to_csv(filename,dataframe)

Plot spectral data
^^^^^^^^^^^^^^^^^^

It is possible to directly plot spectral data using:
::

    plot_spectral_data(dataFrame,dimension='2D')

Where the inputs dataFrame is the spectral data matrix that you wish to plot and the dimension is the dimension of the graph. For the D-matrix, it is more appropriate to change the dimension to ‘3D’ in order to plot the spectra with time as well as wavelength and absorbance.

Multiplicative Scatter Correction (MSC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the experimental measurement data obtained suffers from the scaling or offset effects commonly experienced in spectroscopic measurements, then Multiplicative Scatter Correction (MSC) can be used to pre-process the data using the following function.
::

    D_frame = read_spectral_data_from_txt(filename)
    mD_frame = msc(dataFrame = D_frame)

Automatically the reference spectra is assumed to be the average of each spectrum at each time period. If the user wishes to use a different reference spectrum it can be inputted as a pandas dataframe using the argument reference_spectra=dataframe. An example where MSC is used prior to a Savitzky-Golay filter is provided in Ex_2_estimation_filter_msc.py.

Standard Normal Variate (SNV)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the experimental measurement data obtained suffers from the scatter effects commonly experienced in spectroscopic measurements, then Standard Normal Variate (SNV) can be used to pre-process the data with:
::

    D_frame = read_spectral_data_from_txt(filename)
    sD_frame = snv(dataFrame = D_frame)

SNV is a weighted normalization method that can be sensitive to very noisy entries in the spectra, so it is possible that SNV increases nonlinear behaviour between the S and C matrices, especially as it is not a linear transformation. An additional user-provided offset can be applied to avoid over-normalization in samples that have near-zero standard deviation. The default value is zero, however this could be improved through applying an offset of close to the expected noise level value through the argument offset= noise. The example  Ex_2_estimation_filter_snv.py shows this technique applied to an example prior to Savitzky-Golay filtering.
	
Savitzky-Golay filter
^^^^^^^^^^^^^^^^^^^^^

The Savitzky-Golay (SG) filter is used for smoothing noise from data, with the option to also differentiate the data. It does this by creating a least-squares polynomial fit within successive time windows. In order to implement this smoothing pre=processing step in KIPET the following function is called: 
::

    fD_frame = savitzky_golay(dataFrame = sD_frame, window_size = 15,orderPoly = 2)

Where the user needs to provide the Pandas DataFrame to be smoothed, the number of points over which to apply each smoothing function (window_size) as well as the order of the polynomial to be fitted. Low order polynomials can aggressively smooth data. SNV is commonly employed prior to smoothing to remove scatter effects and an example of this is found in  Ex_2_estimation_filter_snv.py.
A further optional option is to differentiate the data as well, using the orderDeriv argument. This option results in the entire KIPET formulation changing to allow for negative values in the D and S matrices. This option may result in longer solve times and strange-looking solutions as allowing for non-negativity constraints to be relaxed, the rotational ambiguity is increased. An example of this is demonstrated in Ex_2_estimation_filter_deriv.py.

Baseline Shift
^^^^^^^^^^^^^^

If the data matrix contains negative values or has a known shift, then we can implement a baseline shift (adding or subtracting a value from all the data):
::

    D_frame = read_spectral_data_from_txt(filename)
    baseline_shift(dataFrame, shift=None)

If shift is not inputted then the function automatically detects the lowest value in the data and shifts the matrix up or down so that this value is zero. This automatically removes negative values from the dataset. If a specific shift is inputted then the data is shifted by that numerical value.

Adding normally distributed noise
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some simulation cases it may be necessary to add noise to simulated data in order to use this to test estimability or parameter estimation functions. It is possible to use the data_tools function in the following way:
::

    data = add_noise_to_signal(data, size)

This function ensures that Gaussian noise is added to the data (a dataframe) of size (int). The function ensures that no negative values are included by rounding up any negative numbers to 0.
 
Pyomo Simulator
---------------

While tutorial 1 already explained how to use the simulator class, for completion this section provides the full array of options available to the user for the run_sim function:

::

    run_sim(solver,**kwds):
        """ Runs simulation by solving a nonlinear system with ipopt

        Arguments:
            solver (str, required): name of the nonlinear solver to used

            solver_opts (dict, optional): Options passed to the nonlinear 			solver.
            variances (dict, optional): Map of component name to noise 			variance. The map also contains the device noise variance.
            
          	tee (bool,optional): flag to tell the simulator whether to stream 		output to the terminal or not
                    
        Returns:            None

Optimizer Class
---------------

Since tutorial 2 already explains how to make use of the functions in this section, for completion the user is provided with the full array of options available to the user for the run_lsq_given_P function which can be used to initialize any of the optimization functions to obtain parameters or variances:
::

    run_lsq_given_P(self,solver,parameters,**kwds):
        	"""Gives a raw estimate of S given kinetic parameters based on a 	difference of least-squares analysis
      Arguments:
          solver (str, required): name of the nonlinear solver to used

          solver_opts (dict, optional): options passed to the nonlinear solver

          variances (dict, optional): map of component name to noise variance. 							The map also contains the device noise 							variance
	
           tee (bool,optional): flag to tell the optimizer whether to stream 						output to the terminal or not

      initialization (bool, optional): flag indicating whether result should be 							loaded to the pyomo model or not 

       Returns:  Results object with loaded results

VarianceEstimator
-----------------

Since tutorial 2 already explains how to make use of the functions in this section, for completion the user is provided with the full array of options available to the user for the run_opt function for the VarianceEstimator class:

::

    def run_opt(self, solver, **kwds):

        """Solves variance estimation problem following the procedure shown in 	Figure 4 of the documentation. 
	This method solved a sequence of optimization problems to determine variances and also   automatically 		
	sets initialization for the parameter estimation for the variables.

        Args:
            solver_opts (dict, optional): options passed to the nonlinear solver
        
         	tee (bool,optional): flag to tell the optimizer whether to stream output to the terminal or not.
        
           norm (optional): norm for checking convergence. The default value is the infinity norm (np.inf), it uses same options as scipy.linalg.norm

		report_time (optional, bool): True if we want to report the time taken to run the variance estimation

        	max_iter (int,optional): maximum number of iterations for the iterative procedure. Default 400.
           tolerance (float,optional): Tolerance for termination by the change Z. Default 5.0e-5
           subset_lambdas (array_like,optional): Subset of wavelengths to used for the initialization problem,as described in Chen, et al. 									(2016). Default all wavelengths.

        	lsq_ipopt (bool,optional): Determines whether to use ipopt for solving the least squares problems inthe Chen, et al. (2016) procedure. Default False. The default uses scipy.least_squares.

            init_C (DataFrame,optional): Dataframe with concentration data used to start the iterative procedure.
            fixed_device_variance (float, optional): if the device variance is 	known ahead of time and you would not like to 						estimate it, set the variance here.
       	 Returns:  None

Note here that the standard method is to use Scipy least squares, which is actually a slower method for the estimation. Additionally, if device variance is known ahead of time from the manufacturer, we are able to input it directly here.


Parameter Estimator
-------------------

Since tutorial 2 already explains how to make use of the function in this section, for completion the user is provided with the full array of options available to the user for the run_opt function for the ParameterEstimator class:
::

    def run_opt(self, solver, **kwds):
        """ Solves parameter estimation problem.
        Arguments:
           solver (str): name of the nonlinear solver to used

          solver_opts (dict, optional): options passed to the nonlinear solver
        
          variances (dict, optional): map of component name to noise variance. The map also contains the device noise variance.            
          tee (bool,optional): flag to tell the optimizer whether to stream output to the terminal or not.
            
          with_d_vars (bool,optional): flag to the optimizer whether to add 
            				variables and constraints for D_bar(i,j), which is included when we have a problem with noise
		report_time (optional, bool): True if we want to report the time taken to run the parameter estimation
	    covariance(bool, optional): if this is selected, the confidence intervals will be calculated for the estimated parameters. If this is selected then the solver to be used should be ‘ipopt_sens’ or ‘k_aug’ or else an error will be encountered.
        Returns:  Results object with loaded results

Troubleshooting and advanced strategies for difficult problems
--------------------------------------------------------------

Since the problems that KIPET is solving are often highly non-linear and non-convex NLPs, it is often important to provide the solver (IPOPT) with good initial points. This section will briefly describe some of the additional initialization and solver strategies that can be applied in KIPET in order to solve larger and more difficult problems. This section assumes that the user has read the tutorial problems above.
Since the VarianceEstimator needs to solve the full optimization problem, it may be useful to initialize it. It is possible to do this by fixing the unknown parameters to some value (hopefully fairly close to the real values) and then running a least squares optimization in order to get decent initial values for the variables, Z, S, dZ/dt, and C. eg. KIPET provides the ability to do this through an easy to implement function:
::

    p_guess = {'k1':4.0,'k2':2.0}
    raw_results = v_estimator.run_lsq_given_P('ipopt',p_guess,tee=False)
    v_estimator.initialize_from_trajectory('Z',raw_results.Z)    
    v_estimator.initialize_from_trajectory('S',raw_results.S)
    v_estimator.initialize_from_trajectory('dZdt',raw_results.dZdt)
    v_estimator.initialize_from_trajectory('C',raw_results.C)

This will allow the user to initialize the VarianceEstimator using the same methods and functions described in the tutorial sections. Note that it is possible to use the run_lsq_given_P() to initialize the ParameterEstimator method as well if the variances are known or there is no need to compute variances. An example of this is shown in Ad_1_estimation.py.
When running the ParameterEstimator it is possible to improve solution times or to assist IPOPT in converging to a solution by not only providing initializations (either through a least squares with fixed parameters or using the results from the VarianceEstimator) as shown above but also by scaling the NLP. KIPET provides a tool to provide automatic scaling based on the VarianceEstimator’s solution with the following function.
::

    p_estimator.scale_variables_from_trajectory('Z',results_variances.Z)
    p_estimator.scale_variables_from_trajectory('S',results_variances.S)
    p_estimator.scale_variables_from_trajectory('C',results_variances.C)

and this can then be given to the solver as an option in the following way:
::
    
    options = dict()
    options['nlp_scaling_method'] = 'user-scaling'
    results_pyomo = p_estimator.run_opt('ipopt',  tee=True, solver_opts = 							options,variances=sigmas, with_d_vars=True)

If convergences are extremely slow it is also possible to provide the solver with an additional option that changes the barrier update strategy. This option may not necessarily be required, but can help with some problems, especially with noisy data. This is added to the solver options with this:
::

  solver_options['mu_strategy'] = 'adaptive'

Another useful solver option that has not yet been mentioned in this guide and which might help to improve the chances of obtaining a solution is the:
::

    options['bound_push'] =1e-6

Which is the desired minimum distance from the initial point to bound. By keeping this value small it is possible to determine how much the initial point might have to be modified in order to be sufficiently within the bounds.
More information on the IPOPT solver and the available options can be found here:
https://www.coin-or.org/Ipopt/documentation/node2.html
In some cases it can be useful to give initial values for the parameters solving the parameter estimation problems. This can be done providing an additional argument named init, e.g.
::
 
    builder.add_parameter('k1',init=1.0,bounds=(0.0,10.0))

An example can be found in Ex_2_estimationfefactoryTempV.py. 
Furthermore, it might be useful to provide nonnegative bounds for algebraic variables for example for rate laws. To achieve this, add the ones, here r1, with bounds to the TemplateBuilder in the following way
::

    builder.add_algebraic_variable('r1', bounds=(0.0, None))

instead of adding them as a set. This might be useful in some cases but it also restricts the optimization algorithm in a higher manner, such that it can be more difficult to find a solution.   
Another particularly useful feature of KIPET is that we can set certain profiles to have specific features or bounds. And example of this is if we know that some peak exists on one of the pure components’ absorbance or if we know that a certain species’ concentration never exceeds a certain number. To implement bounds such as these, we can use the function:
::

    builder.bound_profile(var = 'S', comp = 'A', bounds = (50,65), profile_range = (1650,1800))

Here the var is which of the profiles we want to bound, comp is the component/species, bounds are the bounds that we wish to impose and profile_range is the specific area we wish to impose the bound. In this case, species A’s absorbance is bounded to between 50 and 65 in the wavelength range of 1650 to 1800. More examples of this are included in the example Ex_2_estimation_bound_prof_fixed_variance.py.
With problems that are difficult to solve it can also be useful to not just initialize the primal variables but also the dual variables from a previous solution. For this the following options should be provided:
::

    options['warm_start_init_point']='yes'
    options['warm_start_bound_push'] = 1e-9
    options['warm_start_mult_bound_push'] = 1e-9
    options['mu_strategy']='adaptive'

and the warmstart argument should be set to true:
::

    results_pyomo = p_estimator.run_opt('ipopt',
                                    tee=True,
                                    solver_opts=options,
                                    variances=sigmas,
                                    with_d_vars=True,
                                    warmstart=True)

An example is provided in Ad_2_estimation_warmstart.py, where we just estimate one parameter first and then initialize the estimation of both parameters with that solution. 
In some cases it can be useful to provide expected optimal parameter values and ensure that the estimated parameters stay close to these values. For that purpose, it is possible to add optional L2-penalty terms to the objective and define the expected parameter values and corresponding penalty weights, e.g.
::

	ppenalty_dict=dict()
	ppenalty_dict={'k1':1.2,  'k2':2.3}

	ppenalty_weights = dict()
   	ppenalty_weights = {'k1': 10., 'k2': 1.}
	
where in ppenalty_dict you define the expected optimal values and in ppenalty_weights you define the corresponding weights.
These dictionaries should then be handed to the ParameterEstimator setting the penaltyparam option to True as well, i.e. 
::
	results_pyomo = p_estimator.run_opt('ipopt',
					    tee=True,
					    solver_opts=options,
					    variances=sigmas,
					    with_d_vars=True,
					    penaltyparam=True,
					    ppenalty_dict=ppenalty_dict,
					    ppenalty_weights=ppenalty_weights)
					    
In case one wants to check the eigenvalues of the reduced Hessian to check whether the estimates have large variances, set the option eigredhess2file option to True, i.e.
::
	eigredhess2file=True
	
handing it to the ParameterEstimator. Note that to use this option you have to solve the problem with sensitivities, i.e. the solver 'ipopt_sens' or 'k_aug' has to be called. 
