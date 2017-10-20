
from setuptools import setup, find_packages
from distutils.core import Extension

DISTNAME = 'kipet'
VERSION = '0.0.0'
PACKAGES = find_packages()
EXTENSIONS = []
DESCRIPTION = 'Package for kinetic estimation based on spectral or concentration data'
LONG_DESCRIPTION = '' #open('README.md').read()
AUTHOR = 'Jose-Santiago-Rodriguez '
MAINTAINER_EMAIL = 'TODO'
LICENSE = 'TODO'
URL = 'TODO'

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