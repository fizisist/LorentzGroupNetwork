# Goal: Convert the CLARIANT-ready toptag HDF5 files to ROOT format.

import sys, os
import numpy as np, h5py as h5, ROOT as rt

def main(args):
    file_to_convert = sys.argv[1]
    chunk_size = int(sys.argv[2]) # Number of entries to read into memory at a time. Larger -> less I/O calls, more memory usage.
    root_filename = file_to_convert.replace('.h5','.root').replace('.hdf5','.root')
    
    nparticle_max = 200 # from toptag: up to 200 4-vectors per event entry
    print('Converting ' + file_to_convert + ' -> ' + root_filename)

    # Open the HDF5 file & get datasets
    h5_file = h5.File(file_to_convert, 'r')
    keys = list(h5_file.keys())
    keys_expected = ['Nobj', 'Pmu', 'is_signal', 'truth_Pmu']
    if(set(keys) != set(keys_expected)):
        print('Error: Input file format invalid.')
        print('Expected keys: ', keys_expected)
        print('   Found keys: ', keys)
        return
    nentries = h5_file['Nobj'].shape[0]
    
    # Set up the ROOT file & tree
    root_file = rt.TFile(root_filename,'UPDATE')
    tree = rt.TTree('events_reduced','tree with toptag events')
    
    # Set up arrays to hold info to copy to the tree. Must specify dtype for this to work!
    nparticle = np.zeros(1,np.dtype('i2'))
    
    Pmu = np.zeros((nparticle_max,4),np.dtype('f8'))
    sig = np.zeros(1,np.dtype('i2'))

    tree.Branch( 'nparticle', nparticle, 'nparticle/S' )
    
    tree.Branch( 'Pmu', Pmu, 'Pmu[nparticle][4]/D' )

    tree.Branch( 'is_signal', sig, 'is_signal/S' )

    # Add max values to TTree.fUserInfo
    tree.GetUserInfo().Add(rt.TParameter('int')('nparticle_max', nparticle_max))

    # Loop through HDF5 file to fill tree.
    # In each loop, we fill some numpy matrices
    # with a chunk of the data, then loop on that
    # chunk to fill the TTree.
    nchunks = int(np.ceil(nentries / chunk_size))
    data = {'Nobj':np.zeros(chunk_size, np.dtype('i2')),
    'Pmu':np.zeros((chunk_size,nparticle_max,4),np.dtype('f8')),
    'is_signal':np.zeros(chunk_size,np.dtype('i2')),
    }

    for i in range(nchunks):
    
        # considering range [start_idx, end_idx)
        start_idx = i*chunk_size
        end_idx = np.minimum((i+1)*chunk_size,nentries)
        length = end_idx-start_idx
        data['Nobj'][0:length] = h5_file['Nobj'][start_idx:end_idx]
        data['Pmu'][0:length,:,:] = h5_file['Pmu'][start_idx:end_idx,:,:]
        data['is_signal'][0:length] = h5_file['is_signal'][start_idx:end_idx]

        for j in range(length):
            nparticle[0] = data['Nobj'][j]
            sig[0] = data['is_signal'][j]
            Pmu[:nparticle[0],:] = data['Pmu'][j,:nparticle[0],:]
            tree.Fill()
        
    tree.Write('',rt.TObject.kOverwrite)
    root_file.Close()
    h5_file.close() # close the HDF5 file
    return
    
if __name__ == '__main__':
    main(sys.argv)

    

