========
Overview
========

This is the home of the LGN network for relativistic particle classification.


Getting started
===============

Installation
------------

The easiest way is to install in top of a conda environment, via pip.
LGN requires Python 3, PyTorch 1.2, CUDA 10, and a few more small packages.
All these should be installed automatically when you run setup.py

Using pip
``````````

LGN is installable using pip.  You can currently install it from
source by going to the directory with setup.py::

    pip install lgn .

If you would like to modify the source code directly, note that LGN
can also be installed in "development mode" using the command::

    pip install lgn -e .


Training examples
----------------

The example training script is in NetworkDesign/scripts/train_lgn.py:.

For one of the small datasets (no extra copying of data required)

    python3 scripts/train_lgn.py --datadir=./data/sample_dataset/ --cpu

The --cpu flag makes the script ignore CUDA. To make the dataset even smaller,
include `--num-train=100 --num-valid=100 --num-test=100` (just 100 datapoints)

The dataset used in the paper is available at https://zenodo.org/record/2603256. If you put it in /data/v0, then you can train on it using the command

    python3 scripts/train_lgn.py --datadir=./data/v0/



Output
----------------

When you start a training session, invariance and covariance tests will be 
first run automatically on the first minibatch (unless you set --test=False).
In this output you will see Boost and Rotation tests for a range of Euler angles,
followed by the average values of components of all tensors in the network (to make sure they are all order 1).
Then the relative errors for the invariance of the outputs and covariance of all activations are shown.
The covariance errors are improved very significantly by turning out double precision with the --double flag.

After these tests, the training starts. Each minibatch generates a line in the log, looking like this:

    prefix E: 1/6, B: 10/151000, L: 0.6997, ACC:   0.9200, AUC:   0.9600  dt:  0.51    5.22    0.01  1.00E-03

"prefix" is the name of the training session, set by the option --prefix=<name>.
"E: 1/6" is the number of the epoch.
"B: 1/151000" is the number of the minibatch within the current epoch.
"L: 0.6997" is the value of the loss that is being minimized (CrossEntropy). It is log(2)~0.69 for a random classifier.
"ACC: 0.9200" is the accuracy at this minibatch.
"AUC: 0.9600" is the Area Under Curve score of this minibatch.
"dt: 0.51   5.22   0.01" is the total time spent on forward and backward passes of this minibatch, 
    total wallclock time since the start of the epoch, and the time spent on collating.
"1.00E-03" is the current learning rate (by default it starts at --lr-init=1E-03 and goes down to --lr-final=1E-05 following a Cosine curve)

NB! By default the live outputs of L, ACC and AUC are exponentially "smoothed" over the history since the start of the epoch. 
This prevents the metrics from jumping around too much between batches and gives a better idea of the average performance,
while still updating them on the fly. The averaging is controlled by the parameter --alpha (default: 100). 
If it's 0, there is no averaging, and the higher it goes the higher is the averaging. For large alpha, the average is weighted in 
such a way that the weight of the (current-alpha)'th minibatch is a factor of e~2.71 smaller than the weight of the current minibatch. 

At the end of an epoch you will see an output like this:

        Epoch 1 Complete! Current Training Loss:     0.6488     0.6700     0.4571     2.8571 @0.3250
        [[0.5  0.1 ]
        [0.23 0.17]]

        ROC saved to file ./predict/prefix.epoch1.train_ROC.csv

        Saving predictions to file: ./prefix/nosave.final.train.pt
        Saving model to checkpoint file: ./model/prefix.pt
        Starting testing on valid set: 
        Done! (Time: 1.502796s)
        Epoch 1 Complete! Current Validation Loss:     0.5079     0.7500     0.5761     23.5000 @0.2453
        [[0.44 0.03]
        [0.22 0.31]]

        ROC saved to file ./predict/prefix.epoch1.valid_ROC.csv

        Saving predictions to file: ./predict/prefix.final.valid.pt
        Lowest loss achieved! Saving best model to file: ./model/prefix_best.pt
        Epoch 1 complete!

The lines starting with "Epoch 1 Complete!" list the total loss, accuracy and AUC values for the epoch,
as well as the background rejection evaluated at the signal efficiency of 30% (or whichever point is closest to 30% on the ROC curve).
The output for this last metric has the form 1/eB @ eS.
Right after that line you will see the confusion matrix, and then confirmations that the ROC, 
the list of predictions, and the model were saved to files.

The full log can be found in ./log/prefix.log. The predictions are saved to ./predict, and the model is saved to ./model.
The names of all files written by a specific training session start with its prefix.

Loading
----------------

After finishing or terminating a training session, it is possible to keep training the same model by using the --load flag.
Note that you cannot change most of the parameters in the command if you intend to load a model. 
You can always check the original command used to train the model at the beginning of its log.
Usually upon loading a model you change only the --num-epoch and perhaps --lr-init and --lr-final.
Keep in mind that the learning rate is updated on a Cosine rule, so if you're continuing a training session that has 
already reached the minimum learning rate, the learning rate will start *rising* along the same Cosine curve.
To avoid this and fully reset the scheduler upon loading the model, uncomment lines 108-109 in ./engine/engine.py.



================
Architecture
================

A more detailed description of the LGN architecture is available in:
`the LGN paper <https://arxiv.org/abs/??????????>`_
