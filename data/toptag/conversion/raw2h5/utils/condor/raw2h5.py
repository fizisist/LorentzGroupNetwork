#  File: raw2h5.py
#  Author: Jan Offermann
#  Date: 09/28/19.
#  Goal: Convert the top-tagging reference dataset to our own HDF5 format.
#        The reference dataset is saved as a pandas DataFrame in an HDF5 file,
#        using the pandas.HDFStore utility. We will instead save things to a
#        "native" (i.e. non-pandas) HDF5 file, which can be opened and read
#        using h5py.
#
#        There are a few different variations on the file format, depending on
#        its use. Besides our default conversion, for use with LGN, we offer
#        the ability to add 2 beam particles to the dataset. In this case,
#        an additional field called "scalars" is also added. For each particle,
#        this contains the mass (which may be helpful), and more importantly,
#        a label that identifies whether or not this is a beam particle.
#
#        For use with the LGN companion network, which is specialized for
#        Lorentz-invariant tasks, we also offer the possibility to pre-compute
#        all 4-momentum dot products p_i p^j. The results are placed in a
#        data field called "dots". In this case, the data field "Pmu" is not
#        saved, nor is "scalars". Instead, the particle labels are also saved
#        in "dots":
#        For N particles, "dots" is of shape [N,N,3], where the [i,j,1]’th
#        element is the dot product of p_i and p_j, the [i,j,2]’th element
#        is the label of p_i, and the [i,j,3]'th element is the label of p_j.
#

import sys, os, time
import pandas as pd
import h5py as h5
import numpy as np
from pt import pt # our numba function for calculating transverse momentum
from dot import dots_matrix_single, masses # our numba functions for calculating dot products & invariant mass
from simplify import SimplifyPath
 
def convert(file_to_convert, add_beams = False, dot_products = False, double_precision = True):
    # Setup beam particles to be added, if required.
    # If added, they will go in the last 2 positions
    # of the 4-momentum list data['Pmu'].
    nbeam = 0
    if(add_beams): nbeam = 2
    beam_mass = 0. # mass for beam particles
    beam_pz = 1.
    beam_E = np.sqrt(beam_mass * beam_mass + beam_pz * beam_pz)
    beam_vecs = np.array([beam_E,0.,0.,beam_pz])
    beam_vecs = np.array([beam_vecs,beam_vecs],dtype=np.dtype('f8'))
    beam_vecs[-1,-1] = -1. * beam_vecs[-1,-1]
    
    # Number of 4-momenta per event, and number of columns in the raw file's DataFrame.
    nvectors_original = 200
    nvectors = nvectors_original + nbeam # 200 4-momenta per event in data
    ncolumns = 806 # expected number of columns in toptag dataset
    
    # Get the DataFrame.
    frame = pd.read_hdf(file_to_convert, 'table')
    nentries = frame.shape[0] # number of entries to convert
    if(frame.shape[1] != ncolumns): # Check that number of columns matches what is expected
        print('Warning: Expected ' + str(ncolumns) +' columns in ' + file_to_convert + ', found ' + str(frame.shape[1]) + '.')
        return
    
    precision = 'f8' #64-bit floating-point number (default)
    if(not double_precision): precision = 'f4' #32-bit floating-point number
    
    # Dictionary holding the data.
    data = {'Nobj':np.zeros(nentries,np.dtype('i2')), # number of 4-momenta per event
    'Pmu':np.zeros((nentries,nvectors,4),np.dtype(precision)), # list of 4-momenta for each event
    'truth_Pmu':np.zeros((nentries,4),np.dtype(precision)), # top 4-momentum for each event (only meaningful for signal)
    'is_signal':np.zeros(nentries,np.dtype('i2')), # signal/background flag
    'jet_pt':np.zeros(nentries,np.dtype(precision)), # jet pt -- used for splitting dataset, *not* used by network
    'label':np.zeros((nentries,nvectors),np.dtype('i2')), # Lorentz-inv. labels -- used to identify different types of particles, e.g. beam vs. non-beam
    'mass':np.zeros((nentries,nvectors),np.dtype(precision)) # particle masses
    }
    
    # Add column for dot products, if required.
    if(dot_products):
        # position (i,j,k) gives dot product p_j p^k, for event i
        data['dots'] = np.zeros((nentries,nvectors,nvectors),np.dtype(precision))
    
    # 4-vectors occupy columns 0-799.
    # truth 4-vector occupues columns 800-803.
    # ttv is in column 804 (redundant variable).
    # is_signal in column 805.
    
    # Get the indices in the raw file corresponding to the reco particle momenta, and the truth-level top momentum.
    pmu_idxs = np.array([np.linspace(0,nvectors_original*4,nvectors_original,False) + x for x in range(4)])
    truth_pmu_idxs = np.linspace(4 * nvectors_original,4 * nvectors_original+4,4,False)

    # Loop over entries, fill the numpy arrays in memory
    for i in range(nentries):
        # Fill in 4-momenta of the reco particles
        for j in range(4): data['Pmu'][i,:nvectors_original,j] = frame.iloc[i,pmu_idxs[j]].values[:] # 4-momenta from the file
        
        # Find + fill in number of non-zero reco particles
        nobj = np.nonzero(frame.iloc[i,pmu_idxs[0]].values == 0.)[0]
        if(nobj.shape[0] == 0): nobj = nvectors # no zeroes found -> (E_0...E_199) must all be non-zero
        else: nobj = nobj[0] + nbeam # Must add nbeam here, in case it is non-zero.
        data['Nobj'][i] = nobj
        
        # Signal flag
        data['is_signal'][i] = frame.iloc[i,-1]
        
        # Fill in truth particle
        data['truth_Pmu'][i,:] = frame.iloc[i,truth_pmu_idxs].values[:]
        
        # Jet pT, from reco particles
        data['jet_pt'][i] = pt(np.sum(data['Pmu'][i,:,:],axis=0))
        
        # Now, add beam particles if necessary.
        if(add_beams): data['Pmu'][i,[-2,-1],:] = beam_vecs
        
        # Now, fill in the particle labels.
            #  1: reco particle
            #  0: no particle (empty)
            # -1: beam particle
        data['label'][i,:nobj] = 1 # default entries are zero
        if(add_beams): data['label'][i,-2:]=-1
        
        # Fill in the particle masses.
        data['mass'][i,:] = masses(data['Pmu'][i,:,:])
        
        # Now, fill in the dot products if necessary.
        if(dot_products): data['dots'][i,:,:] = dots_matrix_single(data['Pmu'][i])
            
    # numpy arrays are filled, now we must write to a new file
    # Prepare the output filename
    output_filename = os.path.dirname(os.path.realpath(__file__)) + '/' + file_to_convert.split('/')[-1]
    output_filename = output_filename.replace('.h5','_c.h5')
    
    # mildly paranoid safety to prevent overwrite of raw info
    if((output_filename is file_to_convert) or ('.h5' not in output_filename)):
        import uuid
        output_filename = os.path.dirname(os.path.realpath(__file__)) + '/' + str(uuid.uuid4().hex) + '.h5'

    # Determine which data columns to write
    keys_to_write = list(data.keys()) # TODO: For now, we will always write all columns.
#    if(dot_products): # "Pmu" should not be written in this case, not needed for this use case
#        keys_to_write = [x for x in keys_to_write if x!='Pmu']
    with h5.File(output_filename, 'w') as f:
        [f.create_dataset(key, data=data[key],compression='gzip') for key in keys_to_write]
    print('File saved as ' + output_filename)
    return

def main(args):
    file_to_convert = str(sys.argv[1])
    add_beams = False
    dot_products = False
    double_precision = True
    if(len(sys.argv) > 2): add_beams = (int(sys.argv[2]) > 0)
    if(len(sys.argv) > 3): dot_products = (int(sys.argv[3]) > 0)
    if(len(sys.argv) > 4): double_precision = (int(sys.argv[4]) > 0)
    if(add_beams): print('Adding beam particles.')
    if(dot_products): print('Adding dot products of the 4-momenta to \'dots\' data column.')
    if(not double_precision): print('Using 32-bit floating point precision.')
    start = time.time()
    output_filename = ''
    convert(file_to_convert, add_beams, dot_products, double_precision) # Performs file conversion
    end = time.time()
    elapsed = end - start
    print('Time elapsed: ' + str(elapsed) + '.')
    return
     
if __name__ == '__main__':
    main(sys.argv)
