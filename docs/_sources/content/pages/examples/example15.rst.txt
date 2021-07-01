Simultaneous Parameter Selection and Estimation Using the Reduced Hessian
-------------------------------------------------------------------------
:Files:
    `Ex_16_reduced_hessian_parameter_selection.py <https://github.com/kwmcbride/kipet_examples/blob/master/examples/example_16/Ex_16_reduced_hessian_parameter_selection.py>`_

The complex models used in reaction kinetic models require accurate parameter estimates.
However, it may be difficult to make accurate estimates for all of the parameters.
To this end, various techniques have been developed to identify parameter subsets that can best be estimated while the remaining parameters are fixed to some initial value.
The selection of this subset is still a challenge.

One such method for parameter subset selection was recently developed by Chen and Biegler (2020).
This method uses a reduced hessian approach to select parameters and estimate their values simultaneously using a collocation approach.
Parameter estimabilty is based on the ratio of their standard deviation to estimated value, and a Gauss-Jordan elimination method strategy is used to rank parameter estimability.
This has been shown to be less computationally demanding than previous methods based on eigenvalues.
For more details about how the algorithm works, the user is recommended to read the article "Reduced Hessian Based Parameter Selection and Estimation with Simultaneous Collocation Approach" by Weifeng Chen and Lorenz T. Biegler, AIChE 2020.

In Kipet, this method is implemented using the EstimationPotential module. It is currently separate from the EstimabilityAnalyzer module used otherwise for estimability (see Tutorial 12).
Kipet can now handle complementary state data, such as temperature and pressure, in its analysis. This should improve the user experience and lead to more robust results.

This module is used in a slightly different manner than other modules in Kipet. The EstimationPotential class requires
the TemplateBuilder instance of the model as the first argument (the models are declared internally). This is followed by the experimental data. Yes, this form of
estimability analysis requires experimental data because the analysis depends on the outputs. For illustration purposes,
the example CSTR problem in this example includes simulated data at the "true" parameter values. Optional arguments include
simulation_data, which takes a Results instance as input. This is recommended for complex systems that require good initilizations.
If no simulation data is provided, the user can use the argument simulate_start to select whether a simulation should be performed internally; performance may vary here, so it is usually better to provide your own simulated data as above.

This tutorial has two examples based on the CSTR example from the paper by Chen and Biegler (2020).

The code for the entire problem is below:

::

    from pyomo.environ import exp
    import kipet
    
    r1 = kipet.ReactionModel('cstr')
    r1.unit_base.time = 'hr'
    r1.unit_base.volume = 'm**3'
   
    # Perturb the initial parameter values by some factor
    factor = 1.2
    
    # Add the model parameters
    Tf = r1.parameter('Tf', value=293.15*factor, bounds=(250, 350), units='K')
    Cfa = r1.parameter('Cfa', value=2500*factor, bounds=(100, 5000), units='mol/m**3')
    rho = r1.parameter('rho', value=1025*factor, bounds=(800, 1100), units='kg/m**3')
    delH = r1.parameter('delH', value=160*factor, bounds=(10, 400), units='kJ/mol')
    ER = r1.parameter('ER', value=255*factor, bounds=(10, 500), units='K')
    k = r1.parameter('k', value=2.5*factor, bounds=(0.1, 10), units='1/hour')
    Tfc = r1.parameter('Tfc', value=283.15*factor, bounds=(250, 350), units='K')#, fixed=True)
    rhoc = r1.parameter('rhoc', value=1000*factor, bounds=(800, 2000), units='kg/m**3')#, fixed=True)
    h = r1.parameter('h', value=1000*factor, bounds=(10, 5000), units='W/m**2/K')#, fixed=True)
    
    # Declare the components and give the valueial values
    A = r1.component('A', value=1000, variance=0.001, units='mol/m**3')
    T = r1.state('T', value=293.15, variance=0.0625,  units='K')
    Tc = r1.state('Tc', value=293.15, variance=0.001, units='K')
   
    # Change this to a clearner method
    full_data = kipet_model.read_data_file('data/all_data.csv')
    
    F = r1.constant('F', value=0.1, units='m**3/hour')
    Fc = r1.constant('Fc', value=0.15, units='m**3/hour')
    Ca0 = r1.constant('Ca0', value=1000, units='mol/m**3')
    V = r1.constant('V', value=0.2, units='m**3')
    Vc = r1.constant('Vc', value=0.055, units='m**3')
    Ar = r1.constant('Area', value=4.5, units='m**2')
    Cpc = r1.constant('Cpc', value=1.2, units='kJ/kg/K')
    Cp = r1.constant('Cp', value=1.55, units='kJ/kg/K')
    
    r1.add_data('T_data', data=full_data[['T']], time_scale='hour')
    #r1.add_data('A_data', data=full_data[['A']].loc[[3.9, 2.6, 1.115505]], time_scale='hour')
    
    # Not really necessary, but useful for tracking
    rA = r1.add_reaction('rA', k*exp(-ER/T)*A, description='Reaction A' )
    
    r1.add_ode('A', F/V*(Cfa - A) - rA )
    r1.add_ode('T', F/V *(Tf - T) + delH/rho/Cp*rA - h*Ar/rho/Cp/V*(T -Tc) )
    r1.add_ode('Tc', Fc/Vc *(Tfc - Tc) + h*Ar/rhoc/Cpc/Vc*(T -Tc) )
    
    # Convert the units
    r1.check_model_units(display=True)
    
    r1.settings.collocation.ncp = 1
    r1.settings.collocation.nfe = 150

To start the simultaneous parameter selection and estimation routine, simply use the estimate method. 
::

    rh_method = 'fixed'
    results = r1.rhps_method(method='k_aug',
                             calc_method=rh_method)


    
.. figure:: ../../images/ex16result3.png
   :width: 600px
   :align: center

   Concentration profiles for the tutorial example 16. Noitce the addition of the three "experimental" points.

.. figure:: ../../images/ex16result4.png
   :width: 600px
   :align: center

   Complementary state (here temperature) profiles for the tutorial example 15