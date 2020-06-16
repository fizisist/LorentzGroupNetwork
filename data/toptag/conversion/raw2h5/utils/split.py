#  File: split.py
#  Author: Jan Offermann
#  Date: 09/28/19.

import sys, os, time
import pandas as pd
import h5py as h5
import numpy as np

# splitting raw files
def raw_split(file_to_convert, output_folder_prefix, ndivisions, dataset = 'df'):
    frame = pd.read_hdf(file_to_convert, dataset)
    frame_shape = frame.shape
    nentries = frame_shape[0]
    nentries_per_file = int(nentries / ndivisions)
    start_idx = []
    end_idx = []
    
    for i in range(ndivisions):
        start_idx.append(nentries_per_file * i)
        end_idx.append(nentries_per_file * (i+1))
    end_idx[-1] = nentries
    
    filename = ''
    for i in range(ndivisions):
        filename = file_to_convert.split('/')[-1]
        folder = output_folder_prefix + str(i)
        filename = folder + '/' + filename
        try: os.mkdir(folder)
        except: pass
        df = frame.iloc[start_idx[i]:end_idx[i],:]
        df.to_hdf(filename,key=dataset,complevel=9)

def ttv_split(file_to_split, names, ttv):

    sum = ttv[0] + ttv[1] + ttv[2]
    train_percentage = float(ttv[0])/float(sum)
    test_percentage = float(ttv[1])/float(sum)
    
    file = h5.File(file_to_split, 'r')
    keys = list(file.keys())
    
    nentries = file[keys[0]].shape[0]
    indices = np.random.permutation(nentries)
    
    split1 = int(nentries * train_percentage)
    split2 = split1 + int(nentries * test_percentage)
    ttv_indices = [indices[:split1], indices[split1:split2], indices[split2:]]
    
    for i in range(len(names)):
        f = h5.File(names[i],'w')
        for key in keys:
            dset = f.create_dataset(key, data=(file[key][:])[ttv_indices[i]], compression = 'gzip')
        f.close()
    file.close()
