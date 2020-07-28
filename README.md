# Lorentz Group Network

Neural network architecture that is fully equivariant with respect to transformations under the Lorentz group, a fundamental symmetry of space and time in physics.

## Overview

This repository holds the software and technical information for a new neural network architecture design based on Lorentz Group Equivariance. The usage and performance of this network is deployed and demonstrated in context of high energy hadronic jet physics. 


## Dependencies

The code in this repository can be broken down in three categories, corresponding with different tasks: dataset conversion, network training, and plotting kinematics & network results. Below we list the code dependencies (global, and task-specific).

##### Global
- [Python3](https://www.python.org)

##### Dataset conversion
- Python3 libraries: [h5py](https://www.h5py.org), [numba](https://numba.pydata.org), [numpy](https://numpy.org), [pandas](https://pandas.pydata.org)
- [HTCondor](https://research.cs.wisc.edu/htcondor/) 
   - optional, highly recommended (for parallelizing conversion tasks)
- [ROOT](https://root.cern.ch) (& [PyROOT](https://root.cern.ch/pyroot))
    - optional

##### Network training
- Python3 libraries: [cudatoolkit](https://anaconda.org/anaconda/cudatoolkit), [h5py](https://www.h5py.org), [PyTorch](https://pytorch.org), [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) 
    - easy setup via conda (see further down)

##### Plotting
- Python3 libraries: [h5py](https://www.h5py.org), [matplotlib](https://matplotlib.org), [numpy](https://numpy.org)
- [ROOT](https://root.cern.ch) (& [PyROOT](https://root.cern.ch/pyroot))




## Installation

The easiest way is to install in top of a conda environment, via pip.
LGN requires Python 3, PyTorch 1.2, CUDA 10, and a few more small packages.
All these should be installed automatically when you run setup.py

### Using pip

LGN is installable using pip.  You can currently install it from
source by going to the directory with setup.py::

    pip install lgn .

If you would like to modify the source code directly, note that LGN
can also be installed in "development mode" using the command::

    pip install lgn -e .



## Running the Code: Top-Tagging
In order to explain how to train the network, we will focus on the example of performing top-tagging using the reference dataset [here](https://zenodo.org/record/2603256) -- this is the dataset used in the summary paper ["The Machine Learning Landscape of Top Taggers"](https://arxiv.org/abs/1902.09914) by G. Kasieczka et. al.

### Basic Training

##### 1) Converting the dataset
First, we need to convert the dataset files into a format that our network's data-loading utilities (and our plotting utilities) understand. This is especially important since we may want to use different datasets from different sources, each with their own formats & organizations -- converting each to a single format avoids the need for multiple PyTorch `Dataset` classes and duplicate plotting scripts. 

In the case of the top-tagging dataset, this conversion is very lightweight: The dataset, like the format our network uses, stores jet constituents as lists of momentum 4-vectors in Cartesian format `(E, px, py, pz)`, where the z-axis corresponds with the beamline. All we need to do is copy the data from a pandas `DataFrame` (saved in an HDF5 file) to a new HDF5 file written using h5py.

To accomplish this, we will use the script at `/DataSamples/toptag/conversion/raw2h5/convert.py`. This script makes use of HTCondor to parallelize the conversion process and speed things up -- it can also be run without HTCondor, in which case the conversion process will not be parallelized. The script can be run as follows:
```
python3 convert.py /path/to/dir/with/data/files njobs
```
Here, the first argument is the path to the directory containing the *unconverted* top-tagging files. The second argument (`njobs`) is the number of jobs to submit to HTCondor. *If not using Condor*, this should be set to `-1`.

##### 2) Training the network

With the data files ready to be read into the network, it's time for training! The process of setting up training is outlined in `/NetworkDesign/Lambda_Instructions.rst`. Some of steps are specific to logging into one of our clusters, `Lambda`, but the instructions are general to all machines capable of setting up conda & using CUDA. For completeness, here is an outline of the instructions:

- Clone the git repository to the machine where the network will be trained.
    ```
    cd [project folder]
    git clone git@github.com:fizisist/LorentzGroupNetwork.git
    ```
- Create a conda environment for this project, and install pytorch, torchvision and cudatoolkit:
    ```
    conda create -n pt python=3.7 anaconda
    conda activate pt
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    ```
    The version of cudatoolkit may depend on the GPU's being used.
- Install LGN
    ```
    cd [project folder]/LorentzGroupNetwork/NetworkDesign/
    pip install -e .
    ```
- Check which GPU's are available, and select one to use for training.
    ```
    nvidia-smi
    export CUDA_VISIBLE_DEVICES=[device_id]
    ```
    Training is currently *not* parallelized across GPU's.
- Train!
    ```
    python3 scripts/train_lgn.py
    ```
    This last script can be passed a wide range of arguments, corresponding with hyperparameters & network configurations. For example, one may consider the following:
    ```
    scripts/train_lormorant.py --datadir=/path/to/dir/with/converted/data/files --maxdim=3 --max-zf=1 --num-channels 2 4 4 2 --num-epoch=10 --batch-size=8 --num-cg-levels=3 --lr-init=0.001 --lr-final=0.00001 --mlp=True --pmu-in=True --nobj=126 --prefix=set8run-n128 --verbose=0
    ```
    - `datadir`: Directory containing the *converted* top-tagging files.
    - `maxdim`: Maximum dimensionality of tensors produced in the network.
    - `max-zf`: Maximum degree of zonal functions used in tensor decompositions.
    - `num-channels`: Number of channels per layer.
    - `num-epoch`: Number of training epochs.
    - `batch-size`: Mini-batch size.
    - `num-cg-levels`: Number of Clebsch-Gordan layers. If this is smaller than `num-channels`, the extra layers at the end will be standard multi-layer perceptrons (MLP's) acting on any Lorentz-invariants produced.
    - `lr-init`: Initial learning rate.
    - `lr-final`: Final learning rate.
    - `mlp`: Whether or not to insert MLP's acting on Lorentz-invariant scalars within the CG layers.
    - `pmu-in`: Whether or not to feed in 4-momenta themselves to the first CG layer, in addition to scalars.
    - `nobj`: Max number of jet constituents to use for entry. Constituents are ordered by decreasing `pT`, so the network uses the `nobj` leading constituents.
    
    For a full list of possible arguments for `train_lgn.py`, see `/NetworkDesign/src/lgn/engine/args.py`.

##### 3) Plotting Network Diagnostics & Dataset Kinematics

We can plot network diagnostics:
- accuracy
- area under the ROC curve
- loss
- signal efficiency at 30% background rejection

using the script at `/Figures/scripts/perf_plot.py`.



##### References

[1] A. Bogatskiy, B. Anderson, J. T. Offermann, M. Roussi, D. W. Miller, R. Kondor, _Lorentz Group Equivariant Neural Network for Particle Physics_, ICML 2020 (accepted).

