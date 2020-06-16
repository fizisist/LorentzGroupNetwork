#!/bin/bash
export HOME=$PWD
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
export ALRB_localConfigDir=$HOME/localConfig
echo 'Running setupATLAS.'
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh --quiet # this technically sets up some version of Python2

# Note: Due to apparent limitations with TPythia8 support, we must use LCG_95 x86_64-centos7-gcc7-opt instead of LCG_96python3 x86_64-slc6-gcc8-opt.
#       This means that we will have to run with Python2 instead of Python3 (yuck!) on the condor workers. We will also have to install numba
#       via pip, because LCG_95 x86_64-centos7-gcc7-opt doesn't offer numba (that would be too convenient and useful...). This is why I hate clusters...

echo 'Running lsetup, with LCG_95 x86_64-centos7-gcc7-opt.'
# Note: Must set up numpy as below, not with pip (otherwise import fails!)
lsetup "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt Python" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt numpy" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt numba" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt pythia8 244" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt ROOT"  # set up python, numpy, numba, pythia8 v. 244, ROOT 6.18.00

echo 'Setup is complete.'
# Run the generation script.
python generator.py
