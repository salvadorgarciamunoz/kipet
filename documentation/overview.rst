Overview
======================================
A new toolkit for kinetic parameter estimation from spectral/concentration data has been developed within python. The package has been developed in colaboration between the pharmaseutical company Eli Lilly and the chemical engineering departments of Purdue and Carnegie Mellon Universities. The package is mainly a reasearch tool for the estimation of kinetic parameters following a unified framework that is based on maximum likelihood principles, robust discretization methods, and large-scale nonlinear optimization. In this document the capabilities of the developed python package are described.

.. image:: figures/portada.png


The Kinetic Parameter Estimation Toolkit (KIPET) is mainly for solving dynamic parameter estimation problems that come from chemical reaction systems. The toolkit provides functionality to solve an iterative optimization-based procedure in order to estimate the variances of the noise in system variables (e.g. concentrations) and spectral measurements. Having estimated the system variances, one can use KIPET to determine the concentration profiles and kinetic parameters simultaneously. The idea is to provide flexible tools that will allow researches to estimate not only kinetic parameters but also the correspoing concentration and absoroptions profiles (Figures 2,3) from multiwavelength spectroscopic data. (Figure 1)  

.. image:: figures/portada1.png
   :width: 48%
.. image:: figures/portada2.png
   :width: 48%


In addition, for a better understanding of chemical reaction systems, we provide within KIPET different functionality to analyse and study kinetic systems. The API is flexible and allows for simulation and optimization of reactive systems described by algebraic-differential equations, along with kinetic parameter estimation and confidence intervals of the estimated values.The following are some of the things that can be done with KIPET:


* Simulate a reactive system described with DAEs  

  * Integrate DAE system with a sundial integrator  
  * Solve the DAE system with collocation methods
  * Stochastic integration of the DAE

* Estimate variances of Noise
* Estimate kinetic parameters
* Estimate confidence intervals of the estimated parameters


In the following sections of this document we give some guidelines on how to install and test the package. We also give a brief description of the third party software that is required to run KIPET. We later show some basic examples where we show how to use the tools. Finally we conclude with some references that provide more detail on the theory behind the numerical techniques implemented in KIPET.
