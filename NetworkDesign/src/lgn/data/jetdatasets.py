import torch
from torch.utils.data import Dataset

import os
from itertools import islice
from math import inf

import logging

class JetDataset(Dataset):
    """
    PyTorch dataset.
    """
    def __init__(self, data, num_pts=-1, shuffle=True):

        self.data = data

        if num_pts < 0:
            self.num_pts = len(data['Nobj'])
        else:
            if num_pts > len(data['Nobj']):
                logging.warn('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['Nobj'])))
                self.num_pts = len(data['Nobj'])
            else:
                self.num_pts = num_pts
        if shuffle:
            self.perm = torch.randperm(len(data['Nobj']))[:self.num_pts]
        else:
            self.perm = None


    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]
        return {key: val[idx] for key, val in self.data.items()}
