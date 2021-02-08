from distutils.core import Extension
import fileinput
import pathlib
from setuptools import setup, find_packages

DISTNAME = 'kipet'
VERSION = '2.1.0'
PACKAGES = find_packages()
EXTENSIONS = []
DESCRIPTION = 'Package for kinetic parameter estimation based on spectral or concentration data'
LONG_DESCRIPTION = '' #open('README.md').read()
AUTHOR = 'Kevin McBride, Christina Schenk, Michael Short, Jose Santiago Rodriguez, David M. Thierry, Salvador Garcia-Munoz, Lorenz T. Biegler'
MAINTAINER_EMAIL = 'kmcbride@andrew.cmu.edu'
LICENSE = 'GPL-3'
URL = 'https://github.com/salvadorgarciamunoz/kipet'

setuptools_kwargs = {
    'zip_safe': False,
    'install_requires': ['six',
                         'pyomo>=5.5',
                         'numpy',
                         'scipy',
                         'pandas',
                         'plotly',
                         'matplotlib'],
    'scripts': [],
    'include_package_data': True,
    'data_files': [('kipet', ['kipet/settings.txt'])],
}


"""
This sets up the default data directory based on where KIPET is installed
"""
print('Updating data and chart paths')

DEFAULT_DATA_DIRECTORY = pathlib.Path('data')
DEFAULT_CHART_DIRECTORY = pathlib.Path('charts')

def replace_in_file(file_path, search_text, new_text):
    with fileinput.input(file_path, inplace=True) as f:
        for line in f:
            new_line = line.replace(search_text, new_text)
            print(new_line, end='')

# Get the installation directory and the expected location of the settings file
install_dir = pathlib.Path(__file__).parent.absolute()
print(f'Parent directory: {install_dir}')
setting_file_path = install_dir.joinpath('kipet', 'settings.txt')

# Replace the data directory based on the installation directory
data_dir_label = 'DATA_DIRECTORY='
data_dir_new = data_dir_label.rstrip('\n') + install_dir.joinpath('kipet', DEFAULT_DATA_DIRECTORY).as_posix()
replace_in_file(setting_file_path, data_dir_label, data_dir_new)
print(f'New data directory: {data_dir_new}')
# Replace the chart directory based on the installation directory
chart_dir_label = 'CHART_DIRECTORY='
chart_dir_path = install_dir.joinpath('kipet', DEFAULT_CHART_DIRECTORY)
chart_dir_new = pathlib.Path(chart_dir_label.rstrip('\n') + chart_dir_path.as_posix())
replace_in_file(setting_file_path, chart_dir_label, chart_dir_new.as_posix())
chart_dir_path.mkdir(exist_ok=True)
print(f'New chart directory: {chart_dir_path}')

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

