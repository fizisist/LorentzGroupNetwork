# reduction.py

# Pythonized version of "reduction.C".
# Takes the ROOT files output by generator.py, and strips them down:
# These ROOT files contain full final-state records, but we just want
# information for the leading jet. We save the results to new ROOT files,
# which will then be converted to HDF5 format by raw2h5.py.

import sys
import ROOT as rt
import numpy as np
import jet_selector as js

def reduce(input_file, strategy = 'top_jet'):

    tfile_in = rt.TFile(input_file, 'READ')
    ttree_in = tfile_in.Get('events')
    nentries = ttree_in.GetEntries()
    
    # Access some info from TTree::fUserInfo:
    #   nconst_max: maximum number of jet constituents ID'd per jet
    #   power, radius, pt_min, eta_max: jet-clustering parameters
    user_info_list = ttree_in.GetUserInfo()
    user_info = {}
    for item in user_info_list:
        # remove the "i_"/"f_" at the start of the name,
        # this is mostly useful when performing casts
        # in C++/CINT
        name = item.GetName()
        if(name[1] == '_'): name = name[2:]
        user_info[name] = item.GetVal()
    
    # Find:
    #   - max number of jet constituents
    #   - max number of final-state particles
    #   - max number of truth-level particles
    # from TTree::UserInfo.
    
    njet_constituents = 200 # default
    nparticle_final = 4000 # default TODO: nparticle_final is unused
    nparticle_truth = 10 # default

    if('njet_constituents' not in user_info.keys()):
        print('Error: Cannot determine maximum number of jet constituents. Defaulting to',njet_constituents,'.')
    else: njet_constituents = user_info['njet_constituents']
    
    if('nparticle_final' not in user_info.keys()):
        print('Error: Cannot determine maximum number of final-state particles. Defaulting to',nparticle_final,'.')
    else: nparticle_final = user_info['nparticle_final']
    
    if('nparticle_truth' not in user_info.keys()):
        print('Error: Cannot determine maximum number of truth-level particles. Defaulting to',nparticle_truth,'.')
    else: nparticle_truth = user_info['nparticle_truth']
    
    # Determine what info to save to the output TTree.
    # For constituents, we only save those from the leading jet.
    data = {
    'n_const' : np.zeros(1,dtype=np.dtype('i2')), # number of jet constituents
    'E' : np.zeros(njet_constituents,dtype=np.dtype('f8')),
    'px' : np.zeros(njet_constituents,dtype=np.dtype('f8')),
    'py' : np.zeros(njet_constituents,dtype=np.dtype('f8')),
    'pz' : np.zeros(njet_constituents,dtype=np.dtype('f8')),
    'n_truth' : np.zeros(1,dtype=np.dtype('i2')), # number of truth particles
    'E_truth' : np.zeros(nparticle_truth,dtype=np.dtype('f8')),
    'px_truth' : np.zeros(nparticle_truth,dtype=np.dtype('f8')),
    'py_truth' : np.zeros(nparticle_truth,dtype=np.dtype('f8')),
    'pz_truth' : np.zeros(nparticle_truth,dtype=np.dtype('f8')),
    'pdg_truth' : np.zeros(nparticle_truth,dtype=np.dtype('i4')), # PDG code -- used to ID particle
    'is_signal' : np.zeros(1,dtype=np.dtype('i2')), # signal flag (0=background, 1=signal)
    'ttv': np.zeros(1,dtype=np.dtype('i2')), # ttv split (0 = training, 1 = testing, 2 = validation)
    }
    
    # Define the output TFile & TTree
    output_file = input_file.replace('.root','_reduced.root')
    tfile_out = rt.TFile(output_file, 'RECREATE')
    ttree_out = rt.TTree('events_reduced','events_reduced')
    
    # Set up branches
    branches = {}
    for key, val in data.items():
        descriptor = key
        if(key in ['E','px','py','pz']): descriptor += '[n_const]'
        elif(key in ['E_truth','px_truth','py_truth','pz_truth','pdg_truth']): descriptor += '[n_truth]'
        descriptor += '/'
        if(val.dtype == np.dtype('i2')): descriptor += 'S'
        elif(val.dtype == np.dtype('i4')): descriptor += 'I'
        elif(val.dtype == np.dtype('i8')): descriptor += 'L'
        elif(val.dtype == np.dtype('f4')): descriptor += 'F'
        elif(val.dtype == np.dtype('f8')): descriptor += 'D'
        else:
            print('Warning, setup issue for branch: ', key, '. Aborting.')
            return
        branches[key] = ttree_out.Branch(key,data[key],descriptor)

    # RNG for TTV split
    rando = rt.TRandom2()
    ttv_split = np.array([3,1,1]) # TODO: Make TTV user-configurable
    ttv_sum  = np.sum(ttv_split)
    ttv_thresholds = np.cumsum(ttv_split)
    
    # Loop over input TTree.
    for event in ttree_in:
    
        # Cleanup: We reset the buffer arrays to contain zero's
        for key in data.keys(): data[key][:] = 0
        
        # Variables to be filled
        is_signal = 0
        lead_jet_idx = -1

        # We need to consistently determine *which* jet to pick for each event.
        # The current idea is to use an input to reduce() called "strategy" that determines
        # what rules to apply. This should allow for flexibility & extending this beyond
        # top/QCD jet dataset production.
        
        # top_jet
        if(strategy == 'top_jet'):
            lead_jet_idx = js.top_jet(event)
            if(lead_jet_idx == -1):
                is_signal = 0
                lead_jet_idx = js.lead_jet(event)
            else: is_signal = 1
            
        # no strategy
        else:
            print('Error: Strategy is not top_jet, but no others established.')
            return
        
        # Now get the indices of the jet's constituent particles.
        # This is a 2D array, but PyROOT will read it back as a
        # 1D array (since this is how it is internally stored
        # by ROOT in the TTree). The 2D shape is given by
        # [njet,njet_constituents] -> [i,j] gives the j'th
        # constituent of the i'th jet. The array is padded
        # with negative values (-1).
        start_idx = lead_jet_idx * njet_constituents
        jet_const_idxs = np.zeros(njet_constituents, dtype = np.dtype('i2'))
        for i in range(njet_constituents):
            jet_const_idxs[i] = event.jet_constituent_indices[start_idx + i]
        jet_const_idxs = jet_const_idxs[0:np.argmax(jet_const_idxs<0)] # remove -1 padding

        # Ready to fill info on the leading jet.
        n = jet_const_idxs.shape[0]
        data['n_const'][0] = n
        
        # Get the TClonesArray of final-state TParticle's from the TTree.
        # TODO: Is there a faster/more Pythonic way to handle the TClonesArray?
        fs_pars = event.final_state_particles
        for i in range(n):
            data['E'][i] = fs_pars[int(jet_const_idxs[i])].Energy() # TODO: Why is the int cast necessary? (for both i2 & i8)
            data['px'][i] = fs_pars[int(jet_const_idxs[i])].Px()
            data['py'][i] = fs_pars[int(jet_const_idxs[i])].Py()
            data['pz'][i] = fs_pars[int(jet_const_idxs[i])].Pz()

        # Get the TClonesArray of truth-level TParticle's from the TTree.
        tl_pars = event.truth_particles
        n = tl_pars.GetEntriesFast()
        data['n_truth'][0] = n
        for i in range(n):
            data['E_truth'][i] = tl_pars[i].Energy()
            data['px_truth'][i] = tl_pars[i].Px()
            data['py_truth'][i] = tl_pars[i].Py()
            data['pz_truth'][i] = tl_pars[i].Pz()
            data['pdg_truth'][i] = tl_pars[i].GetPdgCode()

        # TTV split.
        random_number = rando.Uniform(ttv_sum)
        data['ttv'][0] = np.argmax(ttv_thresholds >= random_number)
        
        # Signal flag. #TODO: should rethink how the signal flag is determined (define some functions like in jet selection?)
        data['is_signal'][0] = is_signal
        
        # Fill the tree.
        ttree_out.Fill()
    
    # Done filling the tree.
    nentries_out = ttree_out.GetEntries()
    print('Sucessfully reduced',nentries_out,'entries. ('+str(int(100 * nentries_out/nentries))+'%)')
    
#     # copy the TTree info.
#     # TODO: Copying UserInfo *before* hadd'ing causes an issue, see: https://sft.its.cern.ch/jira/browse/ROOT-10763
#     # For now, we will just get the userinfo after hadd'ing, from one of the individual reduced files.
#    for param in ttree_in.GetUserInfo():
#        ttree_out.GetUserInfo().Add(param)
#
    ttree_out.Write('',rt.TObject.kOverwrite)
    tfile_out.Close()
    tfile_in.Close()
    return
        
def main(args):
    input_file = str(sys.argv[1])
    reduce(input_file)
    return

if __name__ == '__main__':
    main(sys.argv)

        
        
    
    
    
