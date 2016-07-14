Installation
======================================
	
KIPET can be build from source using an SSH or HTTPS protocol (**NOT COMPLETE** Repository should eventually be hosted at https://lilly/git/kipet.git)::

	git clone https://lilly/git/kipet.git KIPET 
	cd KIPET
	python setup.py install

	
Requirements
-------------
Requirements for KIPET include Python 2.7 along with several Python packages. 

Python
^^^^^^^
Information on installing and using python can be found at 
https://www.python.org/.  Python distributions can also be used to manage 
the Python interface.  Python distributions include Python(x,y) (for Windows) 
and Anaconda (for Windows and Linux). These distributions include most of the 
Python packages needed for KIPET, including Numpy, Scipy, Pandas, 
Matplotlib, and CasADi. 

Python(x,y) can be downloaded from http://python-xy.github.io/.  

RECOMENDED: Anaconda can be downloaded from https://store.continuum.io/cshop/anaconda/.

Python packages
^^^^^^^^^^^^^^^^^
The following python packages are required for WNTR:

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
Packages can be installed using pip.

Optional dependencies
-------------------------

The following python packages are optional for WNTR:

* CasADi: used for interfacing with sundials integrators  
