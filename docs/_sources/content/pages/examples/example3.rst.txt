Simulating Advanced Reaction Systems with Additional States
-----------------------------------------------------------
:Files:
    `Ex_3_complementary.py <https://github.com/kwmcbride/kipet_examples/blob/master/examples/example_3/Ex_3_complementary.py>`_

It is also possible to combine additional complementary states, equations and variables into a KIPET model. In this example a problem is solved that includes a temperature and volume change. In this example the model is defined in the same way as was shown before, however this time the complementary state variable temperature is added as a component using the **state** method.

The system of equations is:

.. math::

	\begin{align}
	k1 &= 1.25 e^{\frac{9500}{1.987}(\frac{1}{320} - \frac{1}{T})}\\
	k2 &= 0.08 e^{\frac{7000}{1.987}(\frac{1}{290} - \frac{1}{T})}\\
	r_A &= -k_1A\\
	r_B &= 0.5k_1A - k_2B\\
	r_C &= 3k_2B\\
	C_{A0} &= 4.0\\
	V_0 &= 240\\
	T_1 &= 35000(298 - T)\\
	T_2 &= 4\cdot 240\cdot 30(T - 305)\\
	T_3 &= V(6500k_1A - 8000k_2B)\\
	D_{en} &= (30A + 60B + 20C)V + 3500\\
	\end{align}
	
Using these expressions, the ODEs for this example reaction are:
	
.. math::

    \begin{align}
    \dot{A} &= r_A + (C_{A0} - A)/V\\
    \dot{B} &= r_B - BV_0/V\\
    \dot{C} &= r_C - CV_0/V\\
    \dot{T} &= (T_1 + T_2 + T_3)/D_{en}\\
    \dot{V} &= V_0
	\end{align}
	
At this time, modeling using certain expressions (like 'exp' in the following expressions) requires importing the functions from pyomo.core.
::
    
    import kipet
	
    # This is needed for the construction of the ODEs
    from pyomo.core import exp
    
    r1 = kipet.ReactionModel('reaction-1')
     
    # Declare the components and give the initial values
    A = r1.component('A', value=1.0)
    B = r1.component('B', value=0.0)
    C = r1.component('C', value=0.0)

    # Declare the complementary states and their initial values
    T = r1.state('T', value=290, description='Temperature')
    V = r1.state('V', value=100, description='Volumne')
    
Similar to components, each complementary state will require an ODE to accompany it. In the case of this tutorial example, the following ODEs are defined:
::
    
    # Define the expressions - note that expression method is not used!
    k1 = 1.25*exp((9500/1.987)*(1/320.0 - 1/T))
    k2 = 0.08*exp((7000/1.987)*(1/290.0 - 1/T))
    
    ra = -k1*A
    rb = 0.5*k1*A - k2*B
    rc = 3*k2*B
    
    cao = 4.0
    vo = 240
    T1 = 35000*(298 - T)
    T2 = 4*240*30.0*(T-305.0)
    T3 = V*(6500.0*k1*A - 8000.0*k2*B)
    Den = (30*A + 60*B + 20*C)*V + 3500.0
    
    # Add ODEs
    r1.add_ode('A', ra + (cao - A)/V )
    r1.add_ode('B', rb - B*vo/V )
    r1.add_ode('C', rc - C*vo/V )
    r1.add_ode('T', (T1 + T2 + T3)/Den )
    r1.add_ode('V', vo )
    
    # Simulation requires a time span
    r1.set_time(2.0)
    
    # Change some of the default settings
    r1.settings.collocation.nfe = 20
    r1.settings.collocation.ncp = 1

    # Simulation
    r1.simulate()  

    # Create plots
    r1.plot()


We can then simulate the model (or use experimental data if available and estimate the parameters) in the same way as described in the previous examples. Please follow the rest of the code and run the examples to obtain the output.

.. figure:: ../../images/ex_3_C.svg
   :width: 600px
   :align: center

   Concentration profiles from Tutorial 3

.. figure:: ../../images/ex_3_T.svg
   :width: 600px
   :align: center

   Temperature profile from Tutorial 3