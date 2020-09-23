#
#  pt_slice.py
#
#  Use this to select jet pT slices of the dataset.
#  For this to work, you *must* be using dataset files
#  that contain the jet_pt column (added March 2020)
#  and have been sorted using utilities/pt_sort.py.
#
#  Created by Jan Offermann on 03/10/20.
#

import sys, os, glob
import h5py as h5

def pt_slice(file, pt_min, pt_max, debug = 0):
    pt_window = [float(pt_min), float(pt_max)]
    if(debug != 0): print('Slicing with jet pT window: ', pt_window)

    file_split = file.replace('.h5',str(pt_min) + '_' + str(pt_max) + '.h5').replace('_sort','')
    f = h5.File(file,'r')
    keys = list(f.keys())
    if('jet_pt' not in keys):
        print('Error: File ', file, ' does not contain key \'jet_pt\', which is needed to split on jet pt. Aborting. (This file may be out of date.)')
        
    if(debug != 0): print('File = ', file)
    nentries = f['Pmu'].shape[0]
    pt = f['jet_pt'][:]
        
    idx_start = 0
    idx_end = nentries-1
    for i in range(nentries):
        if(pt[i] >= pt_window[0]):
            idx_start = i
            break
    for i in range(idx_start, nentries):
        if(pt[i] > pt_window[1]):
            idx_end = i-1
            break
        
    if(debug != 0): print('Indices = [', idx_start , ', ',idx_end, '].')
    g = h5.File(file_split,'w')
    [g.create_dataset(key, data=f[key][idx_start:idx_end+1],compression='gzip') for key in keys]
    f.close()
    g.close()

def main(args):
    file = str(sys.argv[1])
    pt_min = int(sys.argv[2])
    pt_max = int(sys.argv[3])
    debug = 0
    if(len(sys.argv) > 4): debug = int(sys.argv[4])
    pt_slice(file, pt_min, pt_max, debug)
    
if __name__ == '__main__':
    main(sys.argv)
