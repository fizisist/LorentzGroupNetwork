#
# signal_split.py
#
#  Created by Jan Offermann on 03/24/20.
#

import sys, os, glob, subprocess as sub
import numpy as np, h5py as h5

def main(args):
    
    file_dir = sys.argv[1]
    files = glob.glob(file_dir + '/*.h5')
    
    for file in files:
        
        f = h5.File(file,'r')
        keys = list(f.keys())
        
        signal = f['is_signal'][:]
        nentries = signal.shape[0]
        n_signal = int(np.sum(signal))
        n_bck = nentries - n_signal
        
        # get sorted indices based on signal -- first background (0), then signal (1)
        indices = np.argsort(signal)

        # Now we copy things into a new file.
        # Note that h5py allows for indexing like f['Pmu'][[a,b,c,d]],
        # but [a,b,c,d] must be in increasing order or this will crash
        # with a TypeError. This seems like something that should be
        # fixed in h5py, but as a workaround we'll move things into
        # numpy arrays in memory.
        
        f_data = {key: f[key][:] for key in keys}
        sig_data = {}
        bck_data = {}
        
        for key in keys:
            bck_data[key] = f_data[key][indices[:n_bck]]
            sig_data[key] = f_data[key][indices[n_bck:]]
            
        sig_file = file.replace('.h5','_sig.h5')
        bck_file = file.replace('.h5','_bck.h5')

        sig_f = h5.File(sig_file,'w')
        bck_f = h5.File(bck_file,'w')
        
        [sig_f.create_dataset(key, data=sig_data[key],compression='gzip') for key in keys]
        [bck_f.create_dataset(key, data=bck_data[key],compression='gzip') for key in keys]

        print('Signal/background split for file ', file)
        print('Results in\n\t' + sig_file + '\n\t' + bck_file)
        f.close()
        sig_f.close()
        bck_f.close()
    return
        
    
    
    
    

if __name__ == '__main__':
    main(sys.argv)

