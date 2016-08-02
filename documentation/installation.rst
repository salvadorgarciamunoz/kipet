Installation
======================================
	
Kipet can be built from source using an SSH or HTTPS protocol (**NOT COMPLETE** https://lilly/git/kipet.git)::

	git clone https://lilly/git/kipet.git kipet 
	cd kipet
	python setup.py install

These commands will checkout the main source code of the python package and install it in the local python directory (or python environment if any is active). Developers can instead create a symbolic link by running the following commands::

	git clone https://lilly/git/kipet.git kipet 
	cd kipet
	python setup.py develop

Before running any tests or examples users need to make sure all the required third party packages are install. Those packages are described below.
	
Requirements
-------------
Requirements for kipet include Python 2.7 along with several Python packages. 

Python
^^^^^^^
Information on installing and using python can be found at 
https://www.python.org/.  Python distributions can also be used to manage 
the Python interface.  Python distributions include Python(x,y) (for Windows) 
and Anaconda (for Windows and Linux). These distributions include most of the 
Python packages needed for kipet, including Numpy, Scipy, Pandas, 
Matplotlib, and CasADi. 

Python(x,y) can be downloaded from http://python-xy.github.io/.  

Anaconda can be downloaded from https://store.continuum.io/cshop/anaconda/. (RECOMENDED)

Python packages
^^^^^^^^^^^^^^^^^
The following python packages are required for kipet:

* Numpy [vanderWalt2011]_: used to support large, multi-dimensional arrays and matrices, 
  http://www.numpy.org/
* Scipy [vanderWalt2011]_: used to support efficient routines for numerical integration, 
  http://www.scipy.org/
* Pandas [McKinney2013]_: used to analyze and store time series data, 
  http://pandas.pydata.org/
* Matplotlib [Hunter2007]_: used to produce figures, 
  http://matplotlib.org/
* Pyomo [Hart2012]_: used for formulating the optimization problems
  http://pyomo.org/
Packages can be installed using pip or the conda command if available.

Nonlinear solvers
^^^^^^^^^^^^^^^^^

* IPOPT: used for solving the the NLP problems.

Currently IPOPT is the only NLP solver tested with the package. For the installation instructions please refer to

http://www.coin-or.org/Ipopt/documentation/node10.html

It is recommended to compile IPOPT with the HSL linear solvers. The examples and test problems have not been tested yet with other linear solvers besides HSL. The HSL software can be found at

http://hsl.rl.ac.uk/ipopt

Additional information on how to install sIpopt can be found in

https://projects.coin-or.org/Ipopt/wiki/sIpopt

sIpopt is necessary for sensitivity analysis and for the computation of confidence intervals of the parameters estimated with kipet. 

Optional dependencies
-------------------------

The following python packages are optional for KIPET:

* CasADi: used for interfacing with sundials integrators

CasADi can be install by simply downloading the tarballs from

https://github.com/casadi/casadi/wiki/InstallationInstructions

Please download the binaries and copy them in the python path
