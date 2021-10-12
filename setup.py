# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------------
# Load modules
# --------------------------------------------------------------------------------

import os, sys
from setuptools import setup, find_packages



# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

pkg_name = 'annin_dofu'



# --------------------------------------------------------------------------------
# read files
# --------------------------------------------------------------------------------

# get current working directory
# current_path = os.path.abspath(os.path.dirname(__file__))

# read files

# requirements.txt
# requirements_path = os.path.join(current_path, 'requirements.txt')
requirements_path = './requirements.txt'
with open(file=requirements_path, mode='r', encoding='utf-8') as f:
    requirements_list = f.readlines()

# README.rst
# readme_path = os.path.join(current_path, 'README.rst')
readme_path = './README.rst'
with open(file=readme_path, mode='r', encoding='utf-8') as f:
    readme_txt = f.read()

# LICENSE
# license_path = os.path.join(current_path, 'LICENSE')
license_path = './LICENSE'
with open(file=license_path, mode='r', encoding='utf-8') as f:
    license_txt = f.read()

# get version(__version__)
exec(open('{}/_version.py'.format(pkg_name)).read())



# --------------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------------

setup(
    name=pkg_name,
    version=__version__,
    description='{} description'.format(pkg_name),
    long_description=readme_txt,
    author='laplaciannin102(Kosuke Asada)',
    author_email='laplaciannin102@gmail.com',
    install_requires=requirements_list,
    url='https://github.com/laplaciannin102/annin_dofu',
    license=license_txt,
    # packages=find_packages(exclude=('tests', 'docs')),
    packages=[
        pkg_name,
        f'{pkg_name}/response_surface_methodology',
        f'{pkg_name}/stocking_quantity_optimization'
    ],
    package_dir={
        'annin_dofu': 'annin_dofu'
    },
    test_suite='tests'
)

