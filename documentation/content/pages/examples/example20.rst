Mixed Units in a Batch Reactor
------------------------------
:Files:
    `Ex_20_fed_batch_step.py <https://github.com/kwmcbride/kipet_examples/blob/master/examples/example_20/Ex_20_fed_batch_step.py>`_

In this example, we again look at the simple series reaction we have considered in many of the examples. The difference here is that we are modeling a batch reactor with a component feed that lasts only a portion of the reaction.

This example also shows how KIPET can automatically check the units provided and ensure that the resulting model uses consistent units.

.. note::
    Since the ODEs are explicitly entered, the volume changes are not automatically added. In this example you can see that we did this manually, although this is not necessary.

::

    import kipet

	r1 = kipet.ReactionModel('fed_batch')
    
	# Set the base time unit (match data)
    r1.unit_base.time = 'min'
    r1.unit_base.volume = 'L'
    
    # Reaction rate constant (parameter to fit)
    k1 = r1.parameter('k1', value = 0.05, units='ft**3/mol/min')

    # Components
    A = r1.component('A', value=2.0, units='mol/L')
    B = r1.component('B', value=0.0, units='mol/L')
    C = r1.component('C', value=0.0, units='mol/L')
    
    # Reactor volume
    V = r1.volume(value = 0.264172, units='gal')
    
    # Step function for B feed - steps can be added
    s_Qin_B = r1.step('s_Qin_B', coeff=1, time=15, switch='off')
    
    # Volumetric flow rate of the feed
    Qin_B = r1.constant('Qin_B', value=6, units='L/hour')
    
    # Concentration of B in feed
    Cin_B = r1.constant('Cin_B', value=2.0, units='mol/L')
    
    # Add the data
    filename = 'data/abc_fedbatch.csv'
    r1.add_data('C_data', file=filename, remove_negatives=True, time_scale='min')
    
    # Convert your model components to a common base
    # KIPET assumes that the provided data has the same units and will be
    # converted as well - be careful!
    #r1.check_component_units()
    
    Qin = Qin_B * s_Qin_B
    
    R1 = k1*A*B
    QV = Qin/V
    
    r1.add_ode('A', -A*QV - R1 )
    r1.add_ode('B', (Cin_B - B)*QV - R1 )
    r1.add_ode('C', -C*QV + R1)
    r1.add_ode('V', Qin )
    
    # Check for consistant units in the model equations
    r1.check_model_units(display=True)
    
    r1.run_opt()
    r1.plot()
    
	
.. figure:: ../../images/ex_20_C.svg
   :width: 600px
   :align: center

   Concentration profile results

.. figure:: ../../images/ex_20_V.svg
   :width: 600px
   :align: center

   Volume change during the reaction