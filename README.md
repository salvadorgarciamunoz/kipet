# <img alt="KIPET" src="branding/kipetlogo_full.svg" height="60">

[![](https://img.shields.io/github/license/salvadorgarciamunoz/kipet)](https://github.com/salvadorgarciamunoz/kipet/blob/master/LICENSE)
[![](https://img.shields.io/github/last-commit/salvadorgarciamunoz/kipet)](https://github.com/salvadorgarciamunoz/kipet/)
[![](https://img.shields.io/pypi/wheel/kipet)](https://pypi.org/manage/project/kipet/release/0.1.1/)


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


<br>

## Installation

There are many options for installing KIPET.

### PyPi
[![](https://img.shields.io/badge/Install%20with-pip-green)]()
[![](https://img.shields.io/pypi/v/kipet.svg?style=flat)](https://pypi.org/pypi/kipet/)
<br>

A packaged version of KIPET can be installed using:

    pip install kipet

If you run into errors when installing KIPET using pip (such as when installing into a clean virtual environment), try installing the following packages beforehand:

    pip install Cython numpy six
    pip install kipet

### Anaconda 
[![](https://anaconda.org/kwmcbride/kipet/badges/installer/conda.svg)]()
[![Anaconda-Server Badge](https://img.shields.io/conda/vn/kwmcbride/kipet)](https://anaconda.org/kwmcbride/kipet)
[![](https://img.shields.io/conda/pn/kwmcbride/kipet?color=orange)]()


If you are using Anaconda, KIPET can be installed using:

    conda install -c kwmcbride kipet

The anaconda packages have the benefit of including pynumero ready to go, which is needed for some of the methods included in KIPET. You will need to compile these on your own if you choose to install KIPET using a different method. See the [pynumero readme](https://github.com/Pyomo/pyomo/tree/master/pyomo/contrib/pynumero) for more information. Otherwise, you can also use [k_aug](https://github.com/dthierry/k_aug) for these methods as well. 

### Poetry

You may also install KIPET with poetry:

    poetry add kipet


### GitHub

Additionally, KIPET may be installed directly from the repository (for example, if using poetry, simply install the desired branch (#branch) in the following manner):

    poetry add git+https://github.com/salvadorgarciamunoz/kipet#master

Naturally you can simply clone or download the repository if you wish.

    cd <installation directory>
    git clone https://github.com/salvadorgarciamunoz/kipet.git
    cd kipet
    python setup.py install

### Ipopt and k_aug

To use KIPET to its full potential, you should install Ipopt and k_aug. Ipopt is 
a popular solver for non-linear programs and k_aug is a new method to calculate sensitivities
from the KKT matrix. The latter is required if covariances are to be calculated.

TODO: add links to the installation of each here.

To help ease the installation of these software tools, there are two scripts written for
Linux OS (Debian) for installing Ipopt and k_aug. These are available here https://github.com/kwmcbride/Linux-Setup-Scripts

### Examples and Tutorials

All of the example problems can be easily downloaded from the examples repository:

    cd <example directory>
    git clone https://github.com/kwmcbride/kipet_examples.git


To validate your installation, you can now run the test script included with the examples:

    cd <example directory>/kipet_examples
    python run_examples.py

<br>

## License

GPL-3


## Authors

    - Kevin McBride - Carnegie Mellon University
    - Kuan-Han Lin - Carnegie Mellon University
    - Christina Schenk - Basque Center for Applied Mathematics
    - Michael Short - University of Surrey
    - Jose Santiago Rodriguez - Purdue University
    - David M. Thierry - Carnegie Mellon University
    - Salvador García-Muñoz - Eli Lilly
    - Lorenz T. Biegler - Carnegie Mellon University

## Please cite
<br>

 - C. Schenk, M. Short, J.S. Rodriguez, D. Thierry, L.T. Biegler, S. García-Muñoz, W. Chen (2020)
Introducing KIPET: A novel open-source software package for kinetic parameter estimation from experimental datasets including spectra, Computers & Chemical Engineering, 134, 106716. https://doi.org/10.1016/j.compchemeng.2019.106716

 - M. Short, L.T. Biegler, S. García-Muñoz, W. Chen (2020)
Estimating variances and kinetic parameters from spectra across multiple datasets using KIPET, Chemometrics and Intelligent Laboratory Systems, https://doi.org/10.1016/j.chemolab.2020.104012

 - M. Short, C. Schenk, D. Thierry, J.S. Rodriguez, L.T. Biegler, S. García-Muñoz (2019)
KIPET–An Open-Source Kinetic Parameter Estimation Toolkit, Computer Aided Chemical Engineering, 47, 299-304.






