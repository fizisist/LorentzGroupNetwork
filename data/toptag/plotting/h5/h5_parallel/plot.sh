#!/bin/bash
export HOME=$PWD
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
export ALRB_localConfigDir=$HOME/localConfig
echo 'Running setupATLAS.'
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh --quiet # this technically sets up Python2
echo 'Running lsetup, with LCG_96python3 x86_64-slc6-gcc8-opt.'
lsetup "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt Python" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt pip" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt numpy" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt numba" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt h5py" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt ROOT"

# do the conversion!
echo 'Converting files.'
python plotting_h5.py $1
