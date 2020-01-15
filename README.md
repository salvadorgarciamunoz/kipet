KIPET: Kinetic Parameter Estimation Toolkit
===========================================

KIPET is a Python package designed to simulate, and estimate parameters from 
chemical reaction systems through the use of maximum likelihood principles,
large-scale nonlinear programming and discretization methods. The software 
has the following functionality:

* Simulate a reactive system described with DAEs
* Solve the DAE system with collocation methods
* Pre-process data
* Estimate variances of noise from the model and measurements
* Estimate kinetic parameters from spectra or concentration data across 1 or 
  multiple experiments with different conditions
* Estimate confidence intervals of the estimated parameters
* Able to estimate variances and parameters for problems where there is dosing / inputs into the system
* Provide a set of tools for estimability analysis
* Allows for wavelength selection of most informative wavelengths from a dataset
* Visualize results

For more information and detailed tutorials go to our readthedocs website:

https://kipet.readthedocs.io


License
------------

GPL-3

Organization
------------

Directories
  * kipet - The root directory for Kipet source code
  * documentation - user manual
  * kipet/examples - tutorial examples and data files
  * kipet/library - all libraries and functions
  * kipet/validation - validation/test scripts

Authors
----------

   * Christina Schenk Carnegie Mellon University
   * Michael Short Carnegie Mellon University
   * Jose Santiago Rodriguez Purdue University
   * David M. Thierry Carnegie Mellon University
   * Salvador Garcia-Munoz Eli Lilly
   * Lorenz T. Biegler Carnegie Mellon University

Please cite
------------
C. Schenk, M. Short, J.S. Rodriguez, D. Thierry, L.T. Biegler, S. García-Muñoz, W. Chen (2020)
Introducing KIPET: A novel open-source software package for kinetic parameter estimation from experimental datasets including spectra, Computers & Chemical Engineering, Volume 134, 106716.

Michael Short, Christina Schenk, David Thierry, Jose Santiago Rodriguez, Lorenz T Biegler, Salvador Garcia-Muñoz (2019), KIPET–An Open-Source Kinetic Parameter Estimation Toolkit, Computer Aided Chemical Engineering, 47, 299-304.






