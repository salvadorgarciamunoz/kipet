Overview
======================================
Parameter estimation of reaction kinetics from spectroscopic data remains an important and challenging problem. Kipet is a package that uses a unified framework to address this challenge. The presented framework is based on maximum likelihood principles, nonlinear optimization techniques and the use of collocation methods to solve the differential equations involved. To solve the overall parameter estimation problem, kipet provides functionality to solve an iterative optimization-based procedure in order to estimate the variances of the noise in system variables (e.g. concentrations) and spectral measurements. Having estimated the system variances, one can use kipet to determine the concentration profiles and kinetic parameters simultaneously. From the properties of the nonlinear programming (NLP) solver and solution sensitivity, kipet also provides tools to determine the covariance matrix and standard deviations for the estimated kinetic parameters.

.. image:: figures/portada.png

.. image:: figures/portada1.png
   :width: 48%
.. image:: figures/portada2.png
   :width: 48%
The Kinetic Parameter Estimation Toolkit (KIPET) is a python 
package designed to estimate kinetic parameters following an optimization approach.
The API is flexible and allows for simulation and optimization of of reactive systems, 
along with kinetic parameter estimation and confidence intervals of the estimated values. 
The software includes capability to:

* Simulate a reactive system described with DAEs  

  * Integrate DAE system with a sundial integrator  
  * Solve the DAE system with collocation methods
  * Stochastic integration of the DAE

* Estimate variances of Noise
* Estimate kinetic parameters
* Estimate confidence intervals of the estimated parameters
