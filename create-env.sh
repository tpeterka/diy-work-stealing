#!/bin/bash

export SPACKENV=diy-work-stealing-env
export YAML=$PWD/env.yaml

# create spack environment
echo "creating spack environment $SPACKENV"
spack env deactivate > /dev/null 2>&1
spack env remove -y $SPACKENV > /dev/null 2>&1
spack env create $SPACKENV $YAML

# activate environment
echo "activating spack environment"
spack env activate $SPACKENV

spack add mpich@4
spack add diy@master
spack add fmt
spack add spdlog

# following is for optional debugging; comment out if not needed
# spack add gdb
# spack add cgdb
# spack add tmux

# install everything in environment
echo "installing dependencies in environment"
spack install

# deactivate environment
spack env deactivate


