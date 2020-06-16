#!/bin/bash
export HOME=$PWD
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
export ALRB_localConfigDir=$HOME/localConfig
echo 'Running setupATLAS.'
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh --quiet # this technically sets up Python2
echo 'Running lsetup, with LCG_96python3 x86_64-slc6-gcc8-opt.'
# Note: Must set up numpy as below, not with pip (otherwise import fails!)
lsetup "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt Python" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt numpy" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt h5py" # sets up Python3, numpy, h5py


# do the conversion!
echo 'Concatenating ' $1 ' and ' $2 ' into ' $3 ' .'
python concat.py $1 $2 $3 $4 # this will use Python3 as set up above
