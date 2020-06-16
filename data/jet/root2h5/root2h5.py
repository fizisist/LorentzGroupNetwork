#  File: root2h5.py
#  Author: Jan Offermann
#  Date: 08/06/19.
#  Goal: Convert our ROOT MC data files to HDF5 format for easy handling by PyTorch.

#import necessary packages
import ROOT as rt
import sys, os
import numpy as np
import h5py as h5



def convert_file(root_filename, h5_filename):

    number_of_constituents = 200
    number_of_truth_pars = 10
    
    print('Converting ' + root_filename + ' -> ' + h5_filename + ' .')
    input_file = rt.TFile.Open(root_filename)
    tree = input_file.Get('events_reduced')
    nentries = tree.GetEntries()
    nentries_per_chunk = int(1e5);
    
    # buffer for the data to be saved -- defines the HDF5 data structures
    data = {
    'Nobj':np.zeros(nentries_per_chunk,dtype=np.dtype('i2')), # number of jet constituents
    'Pmu':np.zeros((nentries_per_chunk,number_of_constituents,4),dtype=np.dtype('f8')), # 4-momenta of jet constituents
    'truth_Nobj':np.zeros(nentries_per_chunk,dtype=np.dtype('i2')), # number of truth-level particles
    'truth_Pdg':np.zeros((nentries_per_chunk,number_of_truth_pars),dtype=np.dtype('i4')), # PDG codes to ID truth particles
    'truth_Pmu':np.zeros((nentries_per_chunk,number_of_truth_pars,4),dtype=np.dtype('f8')), # truth-level particle 4-momenta
    'is_signal':np.zeros(nentries_per_chunk,dtype=np.dtype('i1')) # signal flag (0 = background, 1 = signal)
    }

    # Prepare the HDF5 file (to be filled)
    dsets = {}
    with h5.File(h5_filename, 'w') as f:
        for key, val in data.items():
            shape = list(val.shape)
            shape[0] = nentries
            shape = tuple(shape)
            dsets[key] = f.create_dataset(key, shape, val.dtype,compression='gzip')
   
    # Some indexing preparation
    nchunks = int(np.ceil(nentries / nentries_per_chunk))
    start_idxs = np.zeros(nchunks,dtype = np.dtype('i8'))
    for i in range(1,start_idxs.shape[0]): start_idxs[i] = start_idxs[i-1] + nentries_per_chunk
    stop_idxs = start_idxs + nentries_per_chunk
    stop_idxs[-1] = nentries
    ranges = stop_idxs - start_idxs
   
    print('Writing to HDF5 file in ' + str(nchunks) + ' chunks.')
   
    # For pdg sorting
    truth_pdgs = np.zeros(number_of_truth_pars,dtype =np.dtype('i8'))
   
    for i in range(nchunks):
        # clear the buffer (for safety)
        for key in data.keys(): data[key][:] = 0
    
        for j in range(ranges[i]):
            tree.GetEntry(start_idxs[i] + j)
            
            n_const = tree.n_const
            data['Nobj'][j] = n_const
            # TODO: Not sure how TTree handles broadcasting, will do C-style loop
            for k in range(n_const):
                data['Pmu'][j,k,0] = tree.E[k]
                data['Pmu'][j,k,1] = tree.px[k]
                data['Pmu'][j,k,2] = tree.py[k]
                data['Pmu'][j,k,3] = tree.pz[k]
    
            n_truth = tree.n_truth
            data['truth_Nobj'][j] = n_truth
            
            # We will sort the truth particles by the absolute value of the PDG codes.
            # #TODO: (This sorting could be moved to reduction.py)
            truth_pdgs[:] = 0
            for k in range(n_truth):
                truth_pdgs[k] = np.abs(tree.pdg_truth[k])
            
            truth_idxs = np.argsort(truth_pdgs[:n_truth])
            for k in range(n_truth):
                data['truth_Pmu'][j,k,0] = tree.E_truth[int(truth_idxs[k])]
                data['truth_Pmu'][j,k,1] = tree.px_truth[int(truth_idxs[k])]
                data['truth_Pmu'][j,k,2] = tree.py_truth[int(truth_idxs[k])]
                data['truth_Pmu'][j,k,3] = tree.pz_truth[int(truth_idxs[k])]
                data['truth_Pdg'][j,k] = tree.pdg_truth[int(truth_idxs[k])]

            data['is_signal'][j] = tree.is_signal
            
        print('\tWriting chunk ' + str(i) + '.')

        with h5.File(h5_filename, 'a') as f:
            for key in dsets.keys():
                dset = f[key]
                dset[start_idxs[i]:stop_idxs[i]] = data[key][:ranges[i]]
    input_file.Close()
    print('Done writing ' + h5_filename)

def main(args):
    root_filename = sys.argv[1]
    h5_filename = root_filename.replace('.root','.h5')
    convert_file(root_filename,h5_filename)
    return


if __name__ == '__main__':
    main(sys.argv)

