Installation Guide
==================

This section will explain how to install KIPET on different operating systems. Since this guide is meant to provide an introduction to all kinds of users, it shall start with the basics. For those with experience in Python, proceed to the later sections in this chapter. Firstly Python will need to be installed.

Python installation
-------------------
It should be noted that this guide is intended for beginner users and those that already have Python and an IDE installed can move onto the next section that describes the required packages, their versions, and the KIPET installation instructions.
Check to see whether Python is already installed by going into the command line window and typing “python”. Do this by searching “terminal” in Linux and MacOS, and “cmd” in Windows. If there is a response from a Python interpreter it should include a version number. KIPET should work with any version of Python from 2.7 up until 3.7. If you use Python 3.7 the corresponding tkinter module python3.7-tk needs to be installed as well. 
Information on downloading and installing Python for windows is found here. If the user is new to Python and/or uncomfortable with command line interfacing, it is recommended that the user use Anaconda as the Integrated Development Environment (IDE). Anaconda can be downloaded for free and can be installed to include Python, associated packages, as well as a code editor.
Firstly go to the Anaconda download page and select the appropriate option for your operating system and computer architecture.

Microsoft Windows installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For Windows, double-click the .exe downloaded from the site and follow the instructions. During installation the user will be asked to “Add Anaconda to my PATH environment variable” and “Register Anaconda as my default Python X.X”. It is recommended to include both of these options.
After installation, launch Anaconda Navigator by searching for it in your start menu or by launching Anaconda Prompt and typing “anaconda-navigator”.

If you want to use a virtual environment inside Anaconda (useful if you are using python with different package versions for other purposes), launch the Anaconda Prompt and create one using
::

   conda create -n yourenvname python=x.x anaconda

where yourenvname is the name you choose for your environment and also choose the python version that you got with Anaconda (latest one is 3.7). Then switch to new environment via:
::

   source activate yourenvname

Then install numpy, scipy, pandas, matplotlib via 
::

   conda install packagename

More info regarding virtual environments in Anaconda, can be found e.g. here.

MacOS installation
^^^^^^^^^^^^^^^^^^

For MacOs double-click on the downloaded .pkg installer and follow the prompts. During installation the user will be asked to “Add Anaconda to my PATH environment variable” and “Register Anaconda as my default Python X.X”. It is recommended to include both of these options.

Open Launchpad and click on the Terminal icon. In the Terminal window type “anaconda-navigator”.

Linux installation
^^^^^^^^^^^^^^^^^^

After downloading Anaconda enter the following into the terminal to install Anaconda:
::

   bash ~/Downloads/Anaconda3-5.1.0-Linux-x86_64.sh

Follow the instructions and be sure to enter Yes when the installer prompts “Do you wish the installer to prepend the Anaconda 3 install to PATH in your /home/<user>/.bashrc ?”.
Close the terminal in order for the installation to take effect.
To verify the installation, open the terminal and type “anaconda-navigator”. The Anaconda Navigator should open, meaning that you have successfully installed Anaconda.

Installing Packages and Dependencies
------------------------------------
In order to ensure that the dependencies are installed correctly an installer should be used.
	
Installer
^^^^^^^^^
It is recommended to use pip to install all packages and dependencies. If the user already has Python 2.7.9 and up or Python 3.4 and up installed, pip is already included and can be updated using (from the command line or terminal):
::

   python -m pip install -U pip

This command needs to be performed from the directory where Python is installed. E.g. If you have Anaconda installed you can open navigate to the directory where it is installed and then enter the appropriate commands.
Alternatively you can enter the following in Anaconda Prompt:
::

   pip install -U pip

For Linux or MacOS. If the user is making use of Anaconda it can be updated using:
::

   conda update anaconda

Note that pip is included in Anaconda and should be up to date if the above is used. If more help is required to install pip, instructions are included here.

Installing Required Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now that pip is installed and updated, we can install the other dependencies and third-party packages required for KIPET.  Some of the major dependencies include:

* Numpy (van der Walt, 2011): used to support large, multi-dimensional arrays and matrices,
  http://www.numpy.org/
* Scipy (van der Walt, 2011): used to support efficient routines for numerical integration,
  http://www.scipy.org/
* Pandas (McKinney, 2013): used to analyze and store time series data,
  http://pandas.pydata.org/
* Matplotlib (Hunter, 2007): used to produce figures,
  http://matplotlib.org/
* Pyomo (Hart, 2012): used for formulating the optimization problems
  http://pyomo.org/

A complete list of all the dependencies is included in Table :numref:`dependency-table`.

If using Anaconda or another scientific Python distribution such as Python(x,y), most of the packages are already installed as part of the distribution. These include Numpy, Scipy, Pandas, and Matplotlib. If using another Python distribution, then each package can be installed individually using pip by the following commands in the MacOS and Linux terminal and the cmd in Windows:
::

   python -m pip install -U numpy

or via the Anaconda Prompt:
::

   pip install -U numpy

Where “numpy” is just one example of the packages that will need to be installed.
Please note here, that if you install packages directly, that  pyomo needs to be installed as follows:
::

   pip install -U pyomo==5.6.1

In order to install packages when using Anaconda we can also use the following commands:
::

   conda install -c conda-forge pyomo==5.6.1

In fact, using Anaconda, Pyomo should be the only additional package to install, as all others should be included in the original environment. It is recommended that the user installs pyomo using the above command in Anaconda prompt in order to ensure pyomo is installed in the correct folder. If any trouble is encountered during installation of any of the dependencies please go to the relevant package websites and follow the more detailed instructions found there.
There is also the possibility of installing all the required dependencies with the adequate versions using:
::

   cd kipet 
   pip install -r requirements.txt

When doing this, it is advised to install KIPET first before installing the dependencies. Be aware of that if you need other versions of the required packages for another python-based software, you should rather install KIPET in a virtual environment and run it in this virtual environment. Another thing that should be noted is that if the user is using Windows 7, it is advised to use Python 2.7, rather than Python 3.x and also that there are some known issues with matplotlib in this case. In particular it will be required to install pypng and freetype-py before installing matplotlib. This may therefore cause the requirements.txt to not function correctly.

.. _dependency-table:
.. table:: List of dependencies for KIPET

   +-------------------------------+-----------+
   | Package                       | Version   | 
   +===============================+===========+
   | appdirs                       | 1.4.3     |
   +-------------------------------+-----------+
   | backports.functools-lru-cache | 1.5       |
   +-------------------------------+-----------+
   | casadi                        | 3.4.0     |
   +-------------------------------+-----------+
   | coverage                      | 4.5.1     |
   +-------------------------------+-----------+
   | cycler                        | 0.10.0    |
   +-------------------------------+-----------+
   | decorator                     | 4.2.1     |
   +-------------------------------+-----------+
   | kiwisolver                    | 1.0.1     |
   +-------------------------------+-----------+
   | matplotlib                    | 2.2.0     |
   +-------------------------------+-----------+
   | networkx                      | 2.1       |
   +-------------------------------+-----------+
   | nose                          | 1.3.7     |
   +-------------------------------+-----------+
   | numpy                         | 1.14.2    |
   +-------------------------------+-----------+
   | pandas                        | 4.5.1     |
   +-------------------------------+-----------+
   | ply                           | 3.11      |
   +-------------------------------+-----------+
   | pyomo                         | 5.6.1     |
   +-------------------------------+-----------+
   | pyparsing                     | 2.2.0     |
   +-------------------------------+-----------+
   | Python-dateutil               | 2.7.0     |
   +-------------------------------+-----------+
   | pytz                          | 2018.3    |
   +-------------------------------+-----------+
   | scipy                         | 1.0.0     |
   +-------------------------------+-----------+
   | PyUtilib                      | 5.6.5     |
   +-------------------------------+-----------+
   | six                           | 1.11.0    |
   +-------------------------------+-----------+

Installing KIPET
----------------

Firstly, KIPET’s source code can be downloaded from https://github.com/salvadorgarciamunoz/kipet.git or through the following command in Linux if git is installed:
::

   git clone https://github.com/salvadorgarciamunoz/kipet.git

Linux and MacOs
^^^^^^^^^^^^^^^
To install KIPET on Linux or MacOS we simply find the directory in the command prompt with the following command:
::
	
   cd kipet

and then install using:
::

   python setup.py install

Microsoft Windows
^^^^^^^^^^^^^^^^^
On Microsoft Windows we can install KIPET by finding either your command prompt or Anaconda Prompt and going into the KIPET folder using:
::

   cd kipet

And then using:
::

   python setup.py install


Installing solver / IPOPT
-------------------------

Currently the only nonlinear solver implemented and tested in KIPET is IPOPT (Wächter and Biegler, 2006). This document only provides basic instructions on the easiest method to install the solvers. For a detailed installation guide please refer to the COIN-OR project website. If you have purchased or obtained access to the HSL solver library for additional linear solvers, the instructions for this compilation are also found on the COIN-OR website.

Linux/MacOS installation
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

Note that this version of IPOPT is not necessarily the most up-to-date and will not have access to the more advanced linear solvers that are available through the HSL library, and so it is rather advised to compile the solver for your own use.
	
Installing k_aug
----------------
If the user would like to utilize k_aug to perform confidence intervals or to compute sensitivities, k_aug needs to be installed and added to the system path. A complete guide can be found within the same folder as this documentation on the Github page, or can be found in David M. Thierry’s Github page https://github/davidmthierry/k_aug . David has also kindly produced a Youtube video that shows how to install k_aug on Windows.
k_aug is a necessary component if the user would like to make use of the estimability analyses offered within KIPET.

sIPOPT installation
-------------------
We have decided to remove sIPOPT from most of the KIPET package as the software has not been maintained for over 10 years and because k_aug has more flexibility. If, however, for whatever reason, you would like to use sIPOPT, it can still be used sensitivity analysis.
Additional information on how to install sIpopt can be found in:
https://projects.coin-or.org/Ipopt/wiki/sIpopt
It is important to notice here that the instructions for Windows, if the solver was installed as shown above, will not work with sIpopt as no binaries for sIpopt are available. Because of this you will need to follow the Cygwin installation instructions provided by the IDAES group’s Akula Paul, included in the same folder as this documentation with the filename of: “Ipopt_sIpopt_Installation_on_Windows_cygwin.pdf”.

Windows PATH Management
-----------------------

If there are issues found with running examples it may be necessary in Windows to add Python to the PATH environmental variable. This can be done through your IDE, Spyder, in the case of this document by following these steps.  Navigate to to Tools>PYTHONPATH Manager in Spyder and add the folder C:\Users\Username\Anaconda3 to the PATH.
If the user would like to use Python commands from the command prompt, as opposed to the Anaconda prompt, then Python can be added to the PATH Environmental Variable by going into the Start Menu, right-clicking on My Computer and clicking on Advanced Settings in Properties. In this window one can find “Environment Variables”. Click Edit and add Python to the PATH variable by adding the location of where Python is installed on your system.
You should now be ready to use KIPET!

Validation of the Package
-------------------------

To test that the package works, there is a test script provided that checks all of the functions within KIPET through the running of multiple examples in series. The examples can take a while to run. If some of the tests do fail then it is possible something is wrong with the installation and some debugging may need to take place. To run the validation script go into the KIPET folder, enter the validation folder and run ‘validate_installation.py’.
:: 

   python validate_installation.py

Note that if sIpopt or k_aug are not installed, certain test problems will fail to run. If this is the case and you do not intend to use the sensitivity calculations, then ignore these failures.

Updating KIPET
--------------
Repeat steps 2.2., 2.3 and 2.7 with the new version downloaded from github. This is now your new work directory.
