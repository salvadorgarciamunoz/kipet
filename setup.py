# -*- coding: utf-8 -*-

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup

import os.path

from setuptools import dist
dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10', 'six>=1.15'])

readme = ''
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'rb') as stream:
        readme = stream.read().decode('utf8')

setup(
    long_description=readme,
    name='kipet',
    version='0.1.6',
    description='An all-in-one tool for fitting kinetic models using spectral and other state data',
    python_requires='==3.*,>=3.8.0',
    project_urls={
        "repository": "https://github.com/salvadorgarciamunoz/kipet"},
    author='Kevin McBride, Christina Schenk, Michael Short, Jose Santiago Rodriguez, David M. Thierry, Salvador Garcia-Munoz, Lorenz T. Biegler',
    author_email='kevin.w.mcbride.86@gmail.com',
    maintainer='Kevin McBride',
    license='GPL-3.0-or-later',
    keywords='optimization scientific parameter reaction spectral',
    packages=[
        'kipet', 'kipet.common', 'kipet.core_methods', 'kipet.dev_tools',
        'kipet.mixins', 'kipet.model_components', 'kipet.nsd_funs',
        'kipet.post_model_build', 'kipet.top_level', 'kipet.variance_methods',
        'kipet.visuals'
    ],
    package_dir={"": "."},
    package_data={
        "kipet": ["*.yml"],
    },
    install_requires=[
        'attrs==20.*,>=20.3.0', 'matplotlib==3.*,>=3.3.4',
        'numpy==1.*,>=1.20.1', 'pandas==1.*,>=1.2.2', 'pint==0.*,>=0.16.1',
        'plotly==4.*,>=4.14.3', 'pyomo==5.*,>=5.7.3', 'pyyaml==5.*,>=5.4.1',
        'scipy==1.*,>=1.6.0'
    ],
    extras_require={"dev": ["pytest==5.*,>=5.2.0", "spyder==4.*,>=4.2.2"]},

)
