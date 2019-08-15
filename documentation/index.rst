.. KIPET documentation master file, created by
   sphinx-quickstart on Tue Aug 13 14:34:34 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to KIPET's documentation!
=================================

KIPET is the one-stop shop for kinetic parameter estimation from batch and fed-batch reactor systems using spectroscopic or concentration data. KIPET is a Python-based package using maximum-likelihood statistics, large-scale nonlinear programming optimization, and finite element discretization in a unified framework to solve a variety of parameter estimation problems. Use KIPET to:

* Simulate reactive system described with DAEs  
* Solve DAE systems with collocation methods
* Pre-process data
* Perform estimability analysis
* Estimate data variances
* Estimate kinetic parameters
* Estimate confidence intervals of the estimated parameters
* Estimate parameters from multiple datasets with different experimental conditions
* Obtain the most informative wavelength set to obtain minimal lack-of-fit
* Analyze your system (SVD, PCA, lack of fit, etc.)
* Visualize results

Table of Contents
=================
.. toctree::
   :maxdepth: 2

   content/installation
   content/introduction
   content/citing
   content/background
   content/tutorials
   content/additional
   content/references



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

KIPET Resources
===============
KIPET development is hosted on GitHub and we welcome feedback and questions there:

https://github.com/salvadorgarciamunoz/KIPET

KIPET makes use of Pyomo as the algebraic modelling language and much of the syntax can be found here:

https://pyomo.readthedocs.io/en/stable/
