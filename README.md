# <img alt="KIPET" src="branding/kipetlogo_full.svg" height="60">

[![](https://img.shields.io/github/license/salvadorgarciamunoz/kipet)](https://github.com/salvadorgarciamunoz/kipet/blob/master/LICENSE)
[![](https://img.shields.io/github/last-commit/salvadorgarciamunoz/kipet)](https://github.com/salvadorgarciamunoz/kipet/)
[![](https://img.shields.io/pypi/wheel/kipet)](https://pypi.org/manage/project/kipet/release/0.1.1/)
<br>

[![](https://img.shields.io/badge/Install%20with-pip-green)]()
[![](https://img.shields.io/pypi/v/kipet.svg?style=flat)](https://pypi.org/pypi/kipet/)
<br>

[![](https://anaconda.org/kwmcbride/kipet/badges/installer/conda.svg)]()
[![Anaconda-Server Badge](https://anaconda.org/kwmcbride/kipet/badges/version.svg)](https://anaconda.org/kwmcbride/kipet)
[![](https://anaconda.org/kwmcbride/kipet/badges/latest_release_date.svg)]()
[![](https://anaconda.org/kwmcbride/kipet/badges/platforms.svg)]()


KIPET is a Python package designed to simulate, and estimate parameters from 
chemical reaction systems through the use of maximum likelihood principles,
large-scale nonlinear programming and discretization methods. 

- **Documentation:** - https://kipet.readthedocs.io
- **Examples and Tutorials** - https://github.com/kwmcbride/kipet_examples
- **Source code:** - https://github.com/salvadorgarciamunoz/kipet
- **Bug reports:** - https://github.com/salvadorgarciamunoz/kipet/issues

It has the following functionality:

 - Simulate a reactive system described with DAEs
 - Solve the DAE system with collocation methods
 - Pre-process data
 - Estimate variances of noise from the model and measurements
 - Estimate kinetic parameters from spectra or concentration data across 1 or 
  multiple experiments with different conditions
 - Estimate confidence intervals of the estimated parameters
 - Able to estimate variances and parameters for problems where there is dosing / inputs into the system
 - Provide a set of tools for estimability analysis
 - Allows for wavelength selection of most informative wavelengths from a dataset
 - Visualize results


Installation
------------

A packaged version of KIPET can be installed using:

    pip install kipet

If you run into errors when installing KIPET using pip, try installing the following packages beforehand:

    pip install Cython numpy six
    pip install kipet

You may also install KIPET with poetry (this method is recommended):

    poetry add kipet

Finally, if you are using Anaconda, KIPET can be installed using:

    conda install -c kwmcbride kipet

Additionally, KIPET may be installed directly from the repository (if you want the latest version, simply install the desired branch (#branch)):

    poetry add git+http://github.com/salvadorgarciamunoz/kipet#master

Naturally you can simply clone or download the repository.

License
------------

GPL-3


Authors
----------

    - Kevin McBride - Carnegie Mellon University
    - Kuan-Han Lin - Carnegie Mellon University
    - Christina Schenk - Basque Center for Applied Mathematics
    - Michael Short - University of Surrey
    - Jose Santiago Rodriguez - Purdue University
    - David M. Thierry - Carnegie Mellon University
    - Salvador García-Muñoz - Eli Lilly
    - Lorenz T. Biegler - Carnegie Mellon University

Please cite
------------
 - C. Schenk, M. Short, J.S. Rodriguez, D. Thierry, L.T. Biegler, S. García-Muñoz, W. Chen (2020)
Introducing KIPET: A novel open-source software package for kinetic parameter estimation from experimental datasets including spectra, Computers & Chemical Engineering, 134, 106716. https://doi.org/10.1016/j.compchemeng.2019.106716

 - M. Short, L.T. Biegler, S. García-Muñoz, W. Chen (2020)
Estimating variances and kinetic parameters from spectra across multiple datasets using KIPET, Chemometrics and Intelligent Laboratory Systems, https://doi.org/10.1016/j.chemolab.2020.104012

 - M. Short, C. Schenk, D. Thierry, J.S. Rodriguez, L.T. Biegler, S. García-Muñoz (2019)
KIPET–An Open-Source Kinetic Parameter Estimation Toolkit, Computer Aided Chemical Engineering, 47, 299-304.






