#!bin/bash

# This script creates a conda env to test a kipet install and validation

################################################################################
# SetUp                                                                        #
################################################################################

env_name="test_env"
install_path="directory to where you want to install it"
git_url="git@github.com:salvadorgarciamunoz/kipet.git"
branch="master"
python_version="3.8.2"

################################################################################
# Install                                                                      #
################################################################################
Install()
{
    
    mkdir $install_path
    cd $install_path

    source ~/scratch/anaconda3/etc/profile.d/conda.sh
    echo y | conda create --name $env_name python=$python_version 
    conda activate $env_name

    git clone $git_url
    cd kipet
    git checkout $branch

    python setup.py install

    echo y | conda install cyipopt
    echo y | conda install pint

    python kipet/validation/validate_tut_problems.py 

}

################################################################################
# Validate                                                                     #
################################################################################
Validate()
{
    cd $install_path

    source ~/scratch/anaconda3/etc/profile.d/conda.sh
    conda activate $env_name
    
    cd kipet
    git checkout $branch
    
    python kipet/validation/validate_tut_problems.py 

}


################################################################################
# Help                                                                         #
################################################################################
Help()
{
   # Display Help
   echo "KIPET Test Install Help."
   echo
   echo "Syntax: kipet [-h|i|c]"
   echo "options:"
   echo "h     Print this Help."
   echo "i     Install KIPET version and run validation" 
   echo "c     Delete env and all files"
   echo
}

################################################################################
# Clean                                                                        #
################################################################################
Clean()
{
    source ~/scratch/anaconda3/etc/profile.d/conda.sh
    echo "Deleting Test Install"
    conda deactivate
    conda env remove --name $env_name
    rm -rf $install_path

}

################################################################################
# Run                                                                          #
################################################################################

while getopts ":hic" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      i) # Install and run validation
         Install
	     exit;;
	  v) # Install and run validation
         Validate
	     exit;;
      c) # Delete
         Clean
         exit;;
      \?) # incorrect option
         echo "Error: Invalid option"
         exit;;

   esac
done

# Default Install
Install
