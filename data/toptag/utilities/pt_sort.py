#
#  pt_sort.py
#
#  Use this to rearrange converted toptag files, to be sorted by decreasing jet pt.
#  Since the dataloader will randomly sample from files, this will have
#  no effect on training, but it is required for splitting the dataset
#  into different jet pT bins.
#
#  Created by Jan Offermann on 3/10/20.
#

import sys, os, glob
import h5py as h5, numpy as np
from numba import jit

def main(args):

    file_dir = str(sys.argv[1])
    files = glob.glob(file_dir + '/*.h5')
    
    for file in files:
        
        print('Starting for file ' + file)
        f = h5.File(file,'r')
        keys = list(f.keys())
        
        # get the number of entries
        nentries = f['Pmu'].shape[0]
        print('\tnentries = ' + str(nentries))
                
        # array of jet pt values
        pt = f['jet_pt'][:]
        
        # get sorted indices based on jet pt
        indices = np.argsort(pt)
           
        # Now we copy things into a new file.
        # Note that h5py allows for indexing like f['Pmu'][[a,b,c,d]],
        # but [a,b,c,d] must be in increasing order or this will crash
        # with a TypeError. This seems like something that should be
        # fixed in h5py, but as a workaround we'll move things into
        # numpy arrays in memory.
                
        f_data = {key: f[key][:] for key in keys}
        
#        g_data = {'Nobj':np.zeros(nentries,np.dtype('i2')), # number of 4-momenta per event
#        'Pmu':np.zeros((nentries,nvectors,4),np.dtype('f8')), # list of 4-momenta for each event
#        'truth_Pmu':np.zeros((nentries,4),np.dtype('f8')), # top 4-momentum for each event (only meaningful for signal)
#        'is_signal':np.zeros(nentries,np.dtype('i2')), # signal/background flag
#        'jet_pt':np.zeros(nentries,np.dtype('f8')) # jet pt -- used for splitting dataset, *not* used by network
#        }
        
        g_data = {key: np.zeros(f_data[key].shape,dtype=f_data[key].dtype) for key in keys}
        
        for key in keys:
            g_data[key][:] = f_data[key][indices]
        
        file2 = file.replace('.h5','_sort.h5')
        g = h5.File(file2,'w')
        [g.create_dataset(key, data=g_data[key],compression='gzip') for key in keys]

        print('Sorted file ', file)
        print('\tResults in ', file2)
        f.close()
        g.close()
    
    


if __name__ == '__main__':
    main(sys.argv)

