#
# slice.py
#
#  Created by Jan Offermann on 03/11/20.
#

import sys, glob, uuid
import subprocess as sub, h5py as h5, numpy as np
from pt_sort import pt_sort
from pt_slice import pt_slice
from balance import balance
from rescale import rescale

def main(args):
    path_to_files = sys.argv[1]
    min = int(sys.argv[2]) # 550
    max = int(sys.argv[3]) # 650
    step = int(sys.argv[4]) # 10
    if(len(sys.argv) > 5): debug = 1
    else: debug = 0

    # Get the original files -- we do *not* want to modify these.
    original_files = glob.glob(path_to_files + '/*.h5')
    
    # First, pT-sort the files.
    for file in original_files: sorted_file = pt_sort(file,debug)
    
    # Record the new filenames.
    sort_files = glob.glob(path_to_files + '/*.h5')
    sort_files = list(set(sort_files) - set(original_files))

    # Next, perform the pT splicing. Delete the sorted files when no longer needed.
    for file in sort_files:
        for i in range(min,max,step): pt_slice(file,str(i),str(i+step))
        sub.check_call(['rm',file])

    # Get the pT-split files. By definition it's everything in the directory except for the "original files".
    pt_files = glob.glob(path_to_files + '/*.h5')
    pt_files = list(set(pt_files) - set(original_files))
    
    # Now our goal is to remove events from *training* pt slices,
    # such that each has the same number of signal and background events, with a 50/50 split.
    pt_files = [x for x in pt_files if('train' in x)]

    # Now we seek file with the smallest number of signal *or* background events.
    files_h5 = [h5.File(x,'r') for x in pt_files]
    ns = [np.sum(x['is_signal'][:]) for x in files_h5]
    nb = [files_h5[x]['is_signal'].shape[0] - ns[x] for x in range(len(files_h5))]

    for file_h5 in files_h5: file_h5.close() # can safely close the files (will be overwritten later)

    min_s = (np.min(ns),np.argmin(ns))
    min_b = (np.min(nb),np.argmin(nb))
    min_index = -1
    if(min_s[0] < min_b[0]): min_index = min_s[1]
    else: min_index = min_b[1]
    min_file = pt_files[min_index] # filename of file with the smallest set of signal or background events
    
    # Now "balance" this file -- make a copy with 50% signal, 50% background.
    balanced_min_file = balance(min_file)
    
    # Now delete min_file, and rename balanced_min_file -> min_file
    sub.check_call(['rm',min_file])
    sub.check_call(['mv',balanced_min_file,min_file])
    
    # Now min file has 50% signal and 50% background, and all the other files can be rescaled to match
    # its number of signal and background events. (Thus they will be "balanced" too by definition).
    
    for pt_file in pt_files:
        if(pt_file == min_file): continue
        rescaled_file = rescale(pt_file, min_file)
        sub.check_call(['rm',pt_file])
        sub.check_call(['mv',rescaled_file,pt_file])
    
    # Done. Now we do an (optional) sanity check.
    if(debug != 0):
        files_h5 = [h5.File(x,'r') for x in pt_files]
        ns = [np.sum(x['is_signal'][:]) for x in files_h5]
        nb = [files_h5[x]['is_signal'].shape[0] - ns[x] for x in range(len(files_h5))]
        for i in range(len(files_h5)): print(pt_files[i], ns[i], nb[i])
    
    return

if __name__ == '__main__':
    main(sys.argv)

