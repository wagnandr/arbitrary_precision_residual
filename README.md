# Description

Small library to evaluate a residual in arbitrary precision.

# Dependencies 

sudo apt-get install libmpfr-dev


# Python

For installation execute

PETSC_DIR=<REPLACE_WITH_PETSC_DIRECTORY> python3 -m pip install .

in the directory containing the setup.py file. Test if the installation was successful with

python3 -c "import flows1d0d3d"

For developers: Note that the providing the -e, --editable parameter to pip allows you to edit and test the python source code without having to reinstall the package.