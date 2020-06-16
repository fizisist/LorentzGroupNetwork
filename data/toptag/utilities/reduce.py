#
#  reduce.py
#
#  Created by Jan Offermann on 03/25/20.
#
#  Goal: Reduce a dataset to some fraction of
#        the total number of events, keeping
#        the same signal/background ratio.
#

import sys, os, glob, subprocess as sub
import numpy as np, h5py as h5

def main(args):
    
    file_dir = sys.argv[1]
    percentage = float(sys.argv[2])
    
    if (percentage > 100.): percentage = 100.
    
    files = glob.glob(file_dir + '/*.h5')
    
    for file in files:
        
        f = h5.File(file,'r')
        keys = list(f.keys())
        
        signal = f['is_signal'][:]
        nentries = signal.shape[0]
        n_signal = int(np.sum(signal))
        n_bck = nentries - n_signal
        
        # Let's figure out which events to copy.
        # We'll make sure that the signal/background ratio
        # is preserved, by taking the same percentage of
        # signal and background events from the file.
        
        indices = np.argsort(signal) # sorted indices -- background (0), then signal (1)
        indices = [indices[:n_bck],indices[n_bck:]]
        [np.random.shuffle(x) for x in indices] # modifies elements
        indices = [x[:int(0.01 * percentage * x.shape[0])] for x in indices]
        indices = np.append(indices[0], indices[1]) # signal & background indices together

        print('File = ', file)
        print('nentries: ', nentries, ' -> ', len(indices))
        # Now we copy things into a new file.
        # Note that h5py allows for indexing like f['Pmu'][[a,b,c,d]],
        # but [a,b,c,d] must be in increasing order or this will crash
        # with a TypeError. This seems like something that should be
        # fixed in h5py, but as a workaround we'll move things into
        # numpy arrays in memory.
        
        f_data = {key: f[key][:] for key in keys}
        g_data = {key: f_data[key][indices] for key in keys}
        
        reduced_file = file.replace('.h5','_reduced_' + str(int(percentage)) + '%.h5')
        g = h5.File(reduced_file,'w')
        [g.create_dataset(key, data=g_data[key],compression='gzip') for key in keys]
        print('Reduction complete, results in ' + reduced_file)
        f.close()
        g.close()
    return
        
    
    
    
    

if __name__ == '__main__':
    main(sys.argv)

