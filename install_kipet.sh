#!bin/bash

# This script creates a conda env to test a kipet install and validation

################################################################################
# SetUp                                                                        #
################################################################################

env_name="test_env"
install_path="/home/user/directory_for_kipet"
                  
git_url="git@github.com:salvadorgarciamunoz/kipet.git"

branch="master"
python_version="3.8.2"

################################################################################
# Display                                                                      #
################################################################################
Display()
{
   # Display Help
   echo "Default KIPET Install Variables"
   echo 
   echo "Installation directory:"
   echo $install_path
   echo 
   echo "Git URL:"
   echo $git_url
   echo
   echo "Branch:"
   echo $branch
   echo 
   echo "Conda environment name:"
   echo $env_name
   echo
}

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
   echo "Syntax: kipet [-h|i|v|c]"
   echo "options:"
   echo "  -h     Print this Help."
   echo "  -i     Install KIPET version"
   echo "  -v     Run validation checks" 
   echo "  -c     Delete env and all files"
   echo "  -d     Display installation information"
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

while [ ! -z "$1" ]; do
  case "$1" in
      --install|-i) # Install and run validation
         Install
	     exit;;
	  -validate|-v) # Validate the install
	     Validate
	     exit;;
      -clean|-c) # Delete
         Clean
         exit;;
     --dir) # Set the  install directory
         shift
         install_path=$1
         echo "KIPET installation directory: $install_path"
         ;;
     --url|-u)
         shift
         git_url="git@github.com:"$1
         echo "GitHub URL: $git_url"
         ;;
     --branch|-b)
        shift
        branch=$1
        echo "Branch: $branch"
         ;;
     --env|-e)
        shift
        env_name=$1
        echo "Conda environment: $env_name"
         ;;
     --display|-d)
         Display
         ;;
      *)
        Help
        ;;
  esac
shift
done

