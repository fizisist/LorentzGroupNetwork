#!/bin/bash
export HOME=$PWD
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
export ALRB_localConfigDir=$HOME/localConfig
echo 'Running setupATLAS.'
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh --quiet # this technically sets up Python2
echo 'Running lsetup, with LCG_96python3 x86_64-slc6-gcc8-opt.'
# Note: Must set up numpy as below, not with pip (otherwise import fails!)
lsetup "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt Python" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt pip" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt numpy" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt h5py" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt pandas" "lcgenv -p LCG_96python3 x86_64-slc6-gcc8-opt numba" # sets up Python3, pip, numpy, h5py, pandas, numba

echo 'Using pip to install pyTables.' # using pip for the above packages (e.g. numpy, h5py) causes problems with imports, when using Python3 from lsetup
python -m pip install tables --user # PyTables -- why on earth did they shorten it to "tables"? Just to confuse people?

# do the conversion!
echo 'Converting ' $1 ' .'
python raw2h5.py $1 $2 $3 $4
