
from setuptools import setup, find_packages
from distutils.core import Extension

DISTNAME = 'kipet'
VERSION = '1.0.0'
PACKAGES = find_packages()
EXTENSIONS = []
DESCRIPTION = 'Package for kinetic estimation based on spectral or concentration data'
LONG_DESCRIPTION = '' #open('README.md').read()
AUTHOR = 'Jose-Santiago-Rodriguez, Salvador Garcia-Munoz, Lorentz T. Biegler, David M. Thierry, Christina Schenk, Michael Short '
MAINTAINER_EMAIL = 'TODO'
LICENSE = 'GPL-3'
URL = 'https://github.com/salvadorgarciamunoz/KIPET'

setuptools_kwargs = {
    'zip_safe': False,
    'install_requires': ['six',
                         'pyomo',
                         'coverage',
                         'numpy',
                         'scipy',
                         'pandas'],
    'scripts': [],
    'include_package_data': True
}

setup(name=DISTNAME,
      version=VERSION,
      packages=PACKAGES,
      ext_modules=EXTENSIONS,
      description='',
      long_description='',
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      **setuptools_kwargs)