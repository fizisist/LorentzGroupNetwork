#  File: split.py
#  Author: Jan Offermann
#  Date: 09/28/19.

import sys, os, time
import h5py as h5
import numpy as np

# splitting h5 files
def split(file_to_convert, output_folder_prefix, ndivisions):
    
    f = h5.File(file_to_convert,'r')
    keys = list(f.keys())
    nentries = f['Nobj'].shape[0]
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
        print('File ', i+1, '/', ndivisions, '\tindices = (',start_idx[i], ', ',end_idx[i], ')')
        with h5.File(filename, 'w') as g:
            dset1 = g.create_dataset('Nobj', data=f['Nobj'][start_idx[i]:end_idx[i]],compression='gzip')
            dset2 = g.create_dataset('Pmu', data=f['Pmu'][start_idx[i]:end_idx[i],:,:],compression='gzip')
            dset3 = g.create_dataset('truth_Pmu', data=f['truth_Pmu'][start_idx[i]:end_idx[i],:],compression='gzip')
            dset4 = g.create_dataset('is_signal', data=f['is_signal'][start_idx[i]:end_idx[i]],compression='gzip')

            

def main(args):
    file_to_convert = str(sys.argv[1])
    folder_prefix = 'run' # hard-coded
    ndivisions = int(sys.argv[2])
    start = time.time()
    split(file_to_convert, folder_prefix, ndivisions)
    end = time.time()
    elapsed = end - start
    print('Time elapsed: ' + str(elapsed) + '.')
     
if __name__ == '__main__':
    main(sys.argv)
