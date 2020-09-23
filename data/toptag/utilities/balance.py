# balance.py
# Created by Jan Offermann on 09/23/20.
#
# Goal: Given a data file, throw out signal
#       or background events so that there are
#       the same number of each (throwing out
#       as few as possible).

import sys
import h5py as h5, numpy as np

def balance(filename):
    file = h5.File(filename,'r')
    keys = list(file.keys())
    
    n = [np.sum(file['is_signal'][:])]
    n.append(file['is_signal'].shape[0] - n[0])
    n = np.array(n,dtype=np.dtype('i8'))
    n_final = np.min(n)
    
    # Determine how many signal and background events to remove.
    n_remove = np.array([np.maximum(x - n_final,0) for x in n],dtype=np.dtype('i8'))
    
    print('Initial # of events (sig,bck):',n)
    print('  Final # of events (sig,bck):',np.array([n_final,n_final],dtype=np.dtype('i8')))
    
    # Get the lists of indices corresponding with signal and background files.
    # These are in increasing order (i.e. ordering is preserved in each).
    idxs = [np.flatnonzero(file['is_signal'][:] == 1),np.flatnonzero(file['is_signal'][:] == 0)]

    # Now randomly remove some indices from each list.
    idxs = [np.setdiff1d(idxs[x],np.random.choice(idxs[x],n_remove[x],replace=False)) for x in range(2)]
    
    # Now join the lists of signal and background indices, and sort to preserve original order.
    idxs = np.sort(np.concatenate((idxs[0],idxs[1])))
    
    # Now copy the input file, using only the index list we've gathered.
    copyname = filename.replace('.h5','_balance.h5')
    copy = h5.File(copyname,'w')
    
    file_data = {key: file[key][:] for key in keys}
    copy_data = {key: file_data[key][idxs] for key in keys}
    
    for key in keys:
        print('Writing data for key ' + str(key) + '.')
        copy.create_dataset(key,data=copy_data[key],compression = 'gzip')
    
    copy.close()
    file.close()
    return copyname

def main(args):
    input_file = sys.argv[1] # file to balance
    balanced_file = balance(input_file)
    return
    
if __name__ == '__main__':
    main(sys.argv)

