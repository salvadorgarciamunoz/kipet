from setuptools import setup, find_packages
from distutils.core import Extension

DISTNAME = 'kipet'
VERSION = '1.1.0'
PACKAGES = find_packages()
EXTENSIONS = []
DESCRIPTION = 'Package for kinetic parameter estimation based on spectral or concentration data'
LONG_DESCRIPTION = '' #open('README.md').read()
AUTHOR = 'Christina Schenk, Michael Short, Jose Santiago Rodriguez, David M. Thierry, Salvador Garcia-Munoz, Lorenz T. Biegler'
MAINTAINER_EMAIL = 'shortm@andrew.cmu.edu'
LICENSE = 'GPL-3'
URL = 'https://github.com/salvadorgarciamunoz/kipet'

setuptools_kwargs = {
    'zip_safe': False,
    'install_requires': ['six',
                         'pyomo>=5.5',
                         'numpy',
                         'scipy',
                         'pandas',
                         'matplotlib'],
    'scripts': [],
    'include_package_data': True
}

setup(name=DISTNAME,
      version=VERSION,
      packages=PACKAGES,
      ext_modules=EXTENSIONS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      **setuptools_kwargs)
