Installation Guide
==================

KIPET is written in the Python programming language and requires at least version 3.8 for best performance. Thus, to use KIPET, Python needs to be installed on your workstation.

.. note::
  
    Support for Python 2.7 has been dropped in the most recent versions of KIPET!

This documentation does not include a detailed decription of how to install Python. There are enough sources on the internet where you can find detailed instructions on how to install it, such as this detailed guide from `Real Python <https://realpython.com/installing-python/>`_ that covers all the major operating systems.

Installation
------------

The latest versions of KIPET are conveniently provided as Python packages which are hosted in the usual locations. You can use any resource you wish to manage the installation, including virtualenv, poetry, conda, etc.

PyPi Package
^^^^^^^^^^^^
KIPET can be installed with the standard packaging manager, pip:
::

    pip install kipet

If you prefer to use poetry to manage your packages, you can also use
::

    poetry add kipet


Anaconda Package 
^^^^^^^^^^^^^^^^
If you prefer to use Anaconda, KIPET can be installed using:
::

    conda install -c kwmcbride kipet

.. note::

    The anaconda packages have the benefit of already including pynumero, which is needed (but not required) for some of the methods included in  KIPET. You will need to compile these on your own if you choose to install KIPET using a different method. See the `pynumero readme <https://github.com/Pyomo/pyomo/tree/master/pyomo/contrib/pynumero>`_ for more information. Otherwise, you can also use `k_aug <https://github.com/dthierry/k_aug>`_ for these methods as well. 


GitHub
^^^^^^

Additionally, KIPET may be installed directly from the repository (for example, if using poetry, simply install the desired branch (#branch) in the following manner):
::

    poetry add git+https://github.com/salvadorgarciamunoz/kipet#master

Naturally you can simply clone or download the repository if you wish. If you desire to further develop KIPET for your own needs, this method is recommended.
::

    cd <installation directory>
    git clone https://github.com/salvadorgarciamunoz/kipet.git
    cd kipet
    python setup.py install


If you would like to contribute to KIPET, this is the recommended installation route.


Installing IPOPT
----------------

To use KIPET for parameter fitting, you need to have a solver installed that can solver NLPs. Currently the only nonlinear solver implemented and tested in KIPET is IPOPT (Wächter and Biegler, 2006). 

This document only provides basic instructions on the easiest method to install the solvers. For a detailed installation guide please refer to the `COIN-OR project website <https://coin-or.github.io/Ipopt/INSTALL.html>`_. If you have purchased or obtained access to the HSL solver library for additional linear solvers, the instructions for this compilation are also found on the COIN-OR website.

The installation methods only show how to install Ipopt. There are several third party linear solvers that Ipopt requires and these also need to be installed. See the COIN-OR link above for more information.

Linux/MacOS Installation
^^^^^^^^^^^^^^^^^^^^^^^^

Download the IPOPT tarball and then issue the following commands in the relevant directory:
::

   gunzip Ipopt-x.y.z.tgz
   tar xvf Ipopt-x.y.z.tar

Where the version number is x.y.z. Rename the directory that was just extracted:
::

   mv Ipopt-x.y.z CoinIpopt

Then go into the directory we just created:
::

   cd CoinIpopt

and we create a directory to move the compiled version of IPOPT to, e.g.:
::

   mkdir build

and enter this directory:
::

   cd build

Then we run the configure script:
::

   ../configure

make the code
::

   make

and then we test to verify that the compilation was successfully completed by entering:
::

   make test

Finally we install IPOPT:
::

   make install

Microsoft Windows
^^^^^^^^^^^^^^^^^
The simplest installation for Microsoft windows is to download the pre-compiled binaries for IPOPT from COIN-OR. After downloading the file and unzipping it you can place this folder into the Pyomo solver location:
::

   C:\Users\USERNAME\Anaconda3\Lib\site-packages\pyomo\solvers\plugins\solvers

Run an example (explained in the next section) to test if it works. This method should also include a functioning version of sIpopt and so the next step is not necessary unless another method of installation is used.
If trouble is experienced using this approach other methods can be used and they are detailed in the Introduction to IPOPT document.
 
Another simple way to install IPOPT for use in the Anaconda environment is to use the following within the Anaconda Prompt:
::

   conda install -c conda-forge ipopt

.. note::

    Note that this version of IPOPT is not necessarily the most up-to-date and will not have access to the more advanced linear solvers that are available through the HSL library, and so it is rather advised to compile the solver for your own use.
	
Installing k_aug
----------------

If the user would like to utilize k_aug to perform confidence intervals or to compute sensitivities, k_aug needs to be installed and added to the system path. A complete installation guide can be found within the same folder as this documentation on the Github page or on David M. Thierry’s `Github page <https://github.com/dthierry/k_aug>`_.

.. note::

    If you are using a Linux OS, you can try a `script <https://github.com/kwmcbride/Linux-Setup-Scripts/blob/master/install_k_aug.sh>`_ that automatically installs k_aug for you.


Examples and Tutorials
----------------------

All of the example problems can be easily downloaded from the examples repository:
::

    cd <example directory>
    git clone https://github.com/kwmcbride/kipet_examples.git


To validate your installation, you can now run the test script included with the examples:
::

    cd <example directory>/kipet_examples
    python run_examples.py


Validation of the Package
-------------------------

If you want to validate your installation of KIPET, you can download the example problems found in the `examples repository <https://github.com/kwmcbride/kipet_examples.git>`_. You can then run all examples with the following:
::

   python kipet_examples/run_examples.py

Note that if sIpopt or k_aug are not installed, certain test problems will fail to run. If this is the case and you do not intend to use the sensitivity calculations, you can simply ignore these failures.

Updating KIPET
--------------

New versions of KIPET can be updated using the respective package manager you used to install KIPET.


Troubeshooting
--------------

Installation via Anaconda Fails
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some users may not be able to install KIPET using Anaconda. The issue being raised is usually in regard to the version of Python not being correct. This is being worked on, but you can still install KIPET using the Ananconda Navigator GUI or by simply installing using pip within your conda environment.


Windows PATH Management
^^^^^^^^^^^^^^^^^^^^^^^

If there are issues found with running examples it may be necessary in Windows to add Python to the PATH environment variable. This can be done through your IDE, Spyder, in the case of this document by following these steps.  Navigate to to Tools>PYTHONPATH Manager in Spyder and add the folder C:\Users\Username\Anaconda3 to the PATH.
If the user would like to use Python commands from the command prompt, as opposed to the Anaconda prompt, then Python can be added to the PATH Environmental Variable by going into the Start Menu, right-clicking on My Computer and clicking on Advanced Settings in Properties. In this window one can find “Environment Variables”. Click Edit and add Python to the PATH variable by adding the location of where Python is installed on your system.
You should now be ready to use KIPET!

Issues
^^^^^^

If you encounter any issues, please report them on our `Github page <https://github.com/salvadorgarciamunoz/kipet/issues>`_.




