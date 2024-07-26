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

# install everything in environment
echo "installing dependencies in environment"
spack install

# deactivate environment
spack env deactivate


