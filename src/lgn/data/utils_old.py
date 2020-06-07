import torch
import numpy as np

import logging, os, h5py, pandas

from torch.utils.data import DataLoader
from . import JetDataset # "jet" dataset for jet identification (top vs. light quark)
from . import FeynmanDataset # "feynman" dataset for Feynman diagram identification (ttbar vs QCD dijet)
from . import TwoVThreeDataset # "2v3body" dataset for identification of events based on 2 leading jet 4-momenta
from . import TwoVThreeComplexDataset # "2v3bodycomplex", like "2v3body" but using 2 lists of jet constituents (more complex)
from . import TopTagDataset # "toptag", like "jet" but at detector level (from top-tagging reference dataset)

def initialize_datasets(args, datadir='../../../data/samples_h5', num_pts=None):
    """
    Initialize datasets.
    """
    ### ------ 1: Get the file names ------ ###
    # datadir should be the directory in which the 3 HDF5 files (out_test.h5, out_train.h5, out_valid.h5) reside
    splits = ["train", "valid", "test"]
    datafiles = [(datadir[:-1] if datadir[-1] is '/' else datadir) + '/out_' + entry + '.h5' for entry in splits]

    ### ------ 2: Set the number of data points ------ ###
    if num_pts is None:
        num_pts={'train': args.num_train, 'test': args.num_test, 'valid': args.num_valid}

    ### ------ 3: Determine the kind of dataset, based on location ------ ###
    datatype = 'jet' # primitive way of getting dataset type, but this might be handy
    if ('feynman' in datadir): datatype = 'feynman'
    elif ('2v3bodycomplex' in datadir): datatype = '2v3bodycomplex'
    elif ('2v3body' in datadir): datatype = '2v3body'
    elif ('toptag' in datadir): datatype = 'toptag' # top tag reference dataset, HDF5 files look different than our own
            
    ### ------ 4: Load the data ------ ###
    datasets = {}
    # Case 1: datasets saved using OUR own HDF5 file structure
    if(datatype is not 'toptag'):
        for i in range(len(splits)):
            with h5py.File(datafiles[i], 'r') as f:
                datasets[splits[i]] = {key: torch.from_numpy(val[:]) for key, val in f.items()}

    # Case 2: toptag dataset, with its own (different) HDF5 file structure. needs pandas
    else:
        for i in range(len(splits)):
            with pandas.HDFStore(datafiles[i]) as f:
                dataframe = f['table'] # TODO: this seems slow, should change if possible. Seems like pandas does away with the advantages of using HDF5, by loading everything in memory. Why is it so awful?!
                columns = table.columns.values # get the list of column names. Can't figure out how to do this w/out going from HDFStore -> DataFrame via the previous line. HDFStore seems kind of limiting.
                datasets[splits[i]] = {entry: torch.from_numpy(dataframe[entry].values) for entry in columns}

    ### ------ 5: Error checking ------ ###
    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]), 'Datasets must have same set of keys!'

    ### ------ 6: Initialize datasets ------ ###
    # Now initialize datasets based upon loaded data
    datatype = 'jet' # primitive way of getting dataset type, but this might be handy
    if ('feynman' in datadir): datatype = 'feynman'
    elif ('2v3bodycomplex' in datadir): datatype = '2v3bodycomplex'
    elif ('2v3body' in datadir): datatype = '2v3body'

    if(datatype is 'jet'): datasets = {split: JetDataset(data, num_pts=num_pts.get(split, -1)) for split, data in datasets.items()}
    elif(datatype is 'feynman'): datasets = {split: FeynmanDataset(data, num_pts=num_pts.get(split, -1)) for split, data in datasets.items()}
    elif(datatype is '2v3body'): datasets = {split: TwoVThreeDataset(data, num_pts=num_pts.get(split, -1)) for split, data in datasets.items()}
    else: datasets = {split: TwoVThreeComplexDataset(data, num_pts=num_pts.get(split, -1)) for split, data in datasets.items()}

    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts

    return args, datasets
