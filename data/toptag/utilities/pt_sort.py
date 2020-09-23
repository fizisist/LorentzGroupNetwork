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

def pt_sort(file, debug = 0):
    if(debug != 0): print('Starting for file ' + file)
    f = h5.File(file,'r')
    keys = list(f.keys())
        
    # get the number of entries
    nentries = f['Pmu'].shape[0]
    if(debug != 0): print('\tnentries = ' + str(nentries))
                
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
    g_data = {key: np.zeros(f_data[key].shape,dtype=f_data[key].dtype) for key in keys}
    for key in keys: g_data[key][:] = f_data[key][indices]
    
    file2 = file.replace('.h5','_sort.h5')
    g = h5.File(file2,'w')
    for key in keys: g.create_dataset(key, data=g_data[key],compression='gzip')
    if(debug != 0):
        print('Sorted file ', file)
        print('\tResults in ', file2)
    f.close()
    g.close()
    return file2
    
def main(args):
    file = str(sys.argv[1])
    sorted_file = pt_sort(file)
    
if __name__ == '__main__':
    main(sys.argv)
