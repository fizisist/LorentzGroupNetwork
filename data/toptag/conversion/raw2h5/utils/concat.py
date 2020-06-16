#  File: concat.py
#  Author: Jan Offermann
#  Date: 10/10/19.
#  Goal: Concatenate two HDF5 files, by creating a new file
#        containing the concatenation of all the datasets.
#        Currently, we assume that the 1st dimension of each
#        dataset is its "length", and we concatenate along
#        this dimension. (thus only this dimension can vary
#        between the two files for them to be combinable).

import h5py as h5
import numpy as np
import sys

# concatenate 2 HDF5 files (file1 & file2)
# into a new one (file3)
def concat2(file1, file2, file3, compress = True, comp_level = 5):
    f1 = h5.File(file1,'r')
    f2 = h5.File(file2,'r')
    keys1 = [str(x) for x in f1.keys()]
    keys2 = [str(x) for x in f2.keys()]
    
    # --- Checks --- #
    # check that keys match
    if(sorted(keys1) != sorted(keys2)):
        print('Keys don\'t match for ', file1, ', ', file2, ' .')
        print(file1, ' keys: ', sorted(keys1))
        print(file2, ' keys: ', sorted(keys2))
        raise ValueError('Key mismatch.')
        return
        
    # check that dimensions match (except for lengths) for each dataset
    for key in keys1:
        shape1 = f1[key].shape
        shape2 = f2[key].shape
        # check number of dimensions
        if(len(shape1) != len(shape2)):
            print('# of dimensions don\'t match for dataset ', key, ' .')
            print(file1, '[', key, '] ndim = ', len(shape1))
            print(file2, '[', key, '] ndim = ', len(shape2))
            raise ValueError('Number of dimensions mismatch.')
            return
            
        # check each dimension, except for the first (length allowed to mismatch)
        n_mismatch = 0
        for i in range(len(shape1)-1):
            if (shape1[i+1] != shape2[i+1]):
                print('Dimension', i+1, ' mismatch for dataset ', key, ' .')
                n_mismatch = n_mismatch + 1
        if(n_mismatch > 0):
            dimension_error = ' dimension mismatch'
            if(n_mismatch > 1): dimension_error = ' dimension mismatches.'
            raise ValueError(str(n_mismatch) + dimension_error)
    
    # --- Finished checks --- #
    # get final shapes of each dataset
    shape_dict = {}
    for key in keys1:
        length = f1[key].shape[0] + f2[key].shape[0]
        shape = (length,) + f1[key].shape[1:]
        shape_dict[key] = shape
    f3 = h5.File(file3,'w')
    for key in keys1:
        a = f1[key][:]
        b = f2[key][:]
        c = np.concatenate((a,b),axis=0) # concatenate along length axis
        if(compress): dset = f3.create_dataset(key, data=c, compression='gzip', compression_opts=comp_level)
        else: dset = f3.create_dataset(key, data=c)
    f3.close()
    f1.close()
    f2.close()
   
def main(args):
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    file3 = sys.argv[3]
    concat2(file1, file2, file3)

if __name__ == '__main__':
    main(sys.argv)

