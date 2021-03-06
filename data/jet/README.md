# Data generation with `jet`

This subdirectory contains the software package needed to generate your own training data for `CLARIANT`. Specifically, it can generate HDF5 data files, in which each entry (defined by the 1st index in the HDF5 file's datasets) corresponds with a jet produced in some kind of proton-proton collision -- this package allows for a large amount of configurability, so that the user can specify the types of events, how the jets are clustered, and how exactly they are selected per event & saved.

### Requirements:
- [Python3](https://www.python.org) *
    - Packages: [h5py](https://www.h5py.org), [numba](https://numba.pydata.org), [numpy](https://numpy.org)
- [ROOT](https://root.cern.ch)
    - Must be set up with [PyROOT](https://root.cern.ch/pyroot), linked to Python3
- [Pythia8](http://home.thep.lu.se/~torbjorn/Pythia.html)
- [HTCondor](https://research.cs.wisc.edu/htcondor/)
    - This is necessary for parallelization. Data generation can be run locally *without* HTCondor, but will not be parallelized.
    - 
*Note: Python3 must be set up to be aliased with the command `python`. Alternatively, a few calls to `subprocess.check_call()` will need to be modified throughout.

This software package makes use of the ROOT [TPythia8](https://root.cern.ch/doc/master/classTPythia8.html) wrapper class -- however ROOT does *not* need to be built with Pythia8 support, as TPythia8 is locally re-implemented as a custom class (to be automatically compiled & loaded if needed).

### Usage:

Events can be generated by running
```
python generate.py
```
All basic settings can be configured in `config/config.txt`. This includes:
- which processes to run*
- pT bin edges & number of events per bin
- how many constituents to save per jet (ordered by pT)
- how many truth-level particles to save per event**

*The processes are defined in `config/samples`. To add new processes, simply create a new subdirectory and place a template Pythia8 configuration file in it. (see the existing files for details -- they must contain variables `$MINIMUM_PT` & `$MAXIMUM_PT`)
**See below

Under-the-hood, the `generate.py` script effectively runs these three processes:
- `/generation/utils/condor/generator.py`: Generate full events in Pythia8, perform jet clustering, and save to a ROOT file.
- `/root2h5/reduction.py`: Take the ROOT files containing full events, and output new ROOT files that contain one jet per event. (This involves filtering -- see the "Jet selection" below).
- `/root2h5/root2h5.py`: Take the ROOT files with one jet per event, and convert them into HDF5 files (separate files for training, testing & validation).

### Jet selection
In order to create a dataset where each entry corresponds with a single jet, we need a rule (or algorithm) to decide which jet to save from each event (we only save one jet per event).
This is implemented by a set of rules defined via functions in `root2h5/jet_selector.py`. These functions are called inside `root2h5/reduction.py`, which selects one jet from each full event record. This design makes it relatively simple to create new rules (by defining new functions in `jet_selector.py`).

### Truth-level particle selection
You may also want to save truth-level particles from each event, together with a jet. For example, these might correspond with the jet's "mother particle", or some particles from somewhere in the middle of that particle's decay & hadronization. This is done during event generation -- and much like jet selection, is handled by a set of rules defined in `generation/utils/condor/truth_selector.py`. Similarly, it is relatively easy to define new rules for truth-level particle selection.

### Output Data Format
TODO
