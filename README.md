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

   * Kevin McBride - Carnegie Mellon University
   * Kuan-Han Lin - Carnegie Mellon University
   * Christina Schenk - Basque Center for Applied Mathematics
   * Michael Short - University of Surrey
   * Jose Santiago Rodriguez - Purdue University
   * David M. Thierry - Carnegie Mellon University
   * Salvador García-Muñoz - Eli Lilly
   * Lorenz T. Biegler - Carnegie Mellon University

Please cite
------------
C. Schenk, M. Short, J.S. Rodriguez, D. Thierry, L.T. Biegler, S. García-Muñoz, W. Chen (2020)
Introducing KIPET: A novel open-source software package for kinetic parameter estimation from experimental datasets including spectra, Computers & Chemical Engineering, 134, 106716. https://doi.org/10.1016/j.compchemeng.2019.106716

M. Short, L.T. Biegler, S. García-Muñoz, W. Chen (2020)
Estimating variances and kinetic parameters from spectra across multiple datasets using KIPET, Chemometrics and Intelligent Laboratory Systems, https://doi.org/10.1016/j.chemolab.2020.104012

M. Short, C. Schenk, D. Thierry, J.S. Rodriguez, L.T. Biegler, S. García-Muñoz (2019)
KIPET–An Open-Source Kinetic Parameter Estimation Toolkit, Computer Aided Chemical Engineering, 47, 299-304.






