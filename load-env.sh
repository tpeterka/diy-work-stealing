#!/bin/bash

# activate the environment
export SPACKENV=diy-work-stealing-env
spack env deactivate > /dev/null 2>&1
spack env activate $SPACKENV
echo "activated spack environment $SPACKENV"

echo "setting flags for building moab-example"
export DIY_PATH=`spack location -i diy`
export FMT_PATH=`spack location -i fmt`


