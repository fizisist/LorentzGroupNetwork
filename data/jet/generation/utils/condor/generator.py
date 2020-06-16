
# generator.py

# Pythonized version of generator.C

import numpy as np, ROOT as rt
import sys, os, glob
from pt2 import Pt2_sort
from config import GetJetParameters, GetNEvents, GetHadronization, GetMPI, GetRngSeed, GetNTruth, GetNFinal
import truth_selector as ts

# Global variables
path_to_this = os.path.dirname(os.path.realpath(__file__)) # directory where this script lives
    
# Generate & save events for a particular config file (corresponding with a particular process)
def Generate(debug = False, truth_selection = ''):

    global path_to_this # not necessary unless modifying this variable
    
    # Configuration files. These will have been copied to the
    # job directory by generate.py. Names are hard-coded.
    configs = {}
    configs['pythia'] = path_to_this + '/pythia_config.txt'
    configs['jet'] = path_to_this + '/jet_parameters.txt'
    configs['general'] = path_to_this + '/config.txt'
    
    # Load the Pythia8 library in ROOT -- the accompanying rootlogon.C takes care
    # of the situation where this doesn't exist, in which case we provide our own
    # local TPythia8.
    rt.gSystem.Load('libEG')
    error_ignore_level = rt.gErrorIgnoreLevel
    rt.gErrorIgnoreLevel = rt.kFatal
    try: rt.gSystem.Load('libEGPythia8')
    except: pass
    rt.gErrorIgnoreLevel = error_ignore_level
    
    # Prepare the output file.
    # This will be handled by condor -> we will not make an output directory,
    # the file will be written to the same location as the script.
    output_filename = 'out.root'
    print('Preparing file: ', output_filename)
    f = rt.TFile(output_filename,'RECREATE')

    # Prepare TPythia
    try: pythia8 = rt.TPythia8(False) # Argument removes banner printout -- thanks, Axel!
    except: pythia8 = rt.TPythia8() # banner removal is a new feature -> might not be ready on Tier3
    pyth = pythia8.Pythia8() # access to underlying Pythia8 object, for full functionality
    pythia8.ReadConfigFile(configs['pythia']) # read in config
    if(not debug): pythia8.ReadString('Print:quiet = on') # avoid excessive printouts
    
    # Set up RNG seed
    rng_seed = GetRngSeed()
    pythia8.ReadString('Random:setSeed = on')
    pythia8.ReadString('Random:seed = ' + str(rng_seed))
    if(debug): print('Random seed =',rng_seed)
    
    # Set up MPI, hadronization
    use_hadronization = GetHadronization()
    use_MPI = GetMPI()
    if(debug):
        print('MPI =',use_MPI)
        print('Hadronization =',use_hadronization)
    if(use_hadronization): pythia8.ReadString('HadronLevel:all = on')
    else: pythia8.ReadString('HadronLevel:all = off')
    if(use_MPI): pythia8.ReadString('PartonLevel:MPI = on')
    else: pythia8.ReadString('PartonLevel:MPI = off')
    
    # Create an array for extracting particles from event listings
    particle_array = rt.TClonesArray('TParticle',4000) # TODO: Should this value be adjusted?
    # initialize Pythia8
    pyth.init()
        
    # Set up Pythia8 jet clustering (using "SlowJet", which despite the name, should be fast now).
    n_sel = 2 # exclude neutrinos and other invisibles from jet study
    #TODO: Determine mass setting
    # 0 = all massless
    # 1 = photons are massless, everything else is given pi+- mass
    # 2 = all given correct masses
    mass_setting = 2
    jet_vars = GetJetParameters(configs['jet'])
    jet_power = jet_vars['power'][0]
    jet_radius = jet_vars['radius'][0]
    jet_pt_min = jet_vars['min_pt'][0]
    jet_eta_max = jet_vars['max_eta'][0]
    slowjet = rt.Pythia8.SlowJet(jet_power, jet_radius, jet_pt_min, jet_eta_max, n_sel, mass_setting)
    
    # Additional jet variables, not used by SlowJet
    njet_min = 1 # minimum number of jets required in an event
    njet_max = 20 # maximum number of jets allowed in an event
    nconst_max = jet_vars['nconst'][0] # max number of constituents to save from each jet (pT-ordered)

    # Additional event-level variables
    nparticle_final = GetNFinal()
    nparticle_truth = GetNTruth()

    # Variables for filling TTree branches
    data = {
    'process_id' : np.zeros(1,dtype=np.dtype('i4')), # id number of process (see Pythia/PDG manual)
    'final_state_particles' : rt.TClonesArray('TParticle',nparticle_final), # array of final-state particles
    'nparticle' : np.zeros(1,dtype=np.dtype('i2')), # number of final-state particles
    'njet' : np.zeros(1,dtype=np.dtype('i2')), # number of jets
    'jet_m' : np.zeros(njet_max, dtype=np.dtype('f8')), # jet mass
    'jet_pt' : np.zeros(njet_max, dtype=np.dtype('f8')), # jet pt
    'jet_eta' : np.zeros(njet_max, dtype=np.dtype('f8')), # jet eta
    'jet_phi' : np.zeros(njet_max, dtype=np.dtype('f8')), # jet phi
    'jet_nconst' : np.zeros(njet_max, dtype=np.dtype('i2')), # jet number of constituents
    'jet_constituent_indices' : np.full((njet_max, nconst_max),-1,dtype=np.dtype('i2')), # element [i][j] is the index (in final_state_particles) of the jth constituent of the ith jet
    'truth_particles' : rt.TClonesArray('TParticle',nparticle_truth), # array of (optional) truth-level particles
    'ntruth' : np.zeros(1,dtype=np.dtype('i2'))
    }

    # Set up the TTree and its branches
    tree = rt.TTree('events','events')
    branches = {}
    
    # Use fancy Python to set up all the basic-type branches
    for key, val in data.items():
        if(type(val) != np.ndarray): continue
        if(val.ndim > 1): continue
        descriptor = key
        if('jet_' in descriptor): descriptor += '[njet]'
        descriptor += '/'
        
        if(val.dtype == np.dtype('i2')): descriptor += 'S'
        elif(val.dtype == np.dtype('i4')): descriptor += 'I'
        elif(val.dtype == np.dtype('i8')): descriptor += 'L'
        elif(val.dtype == np.dtype('f4')): descriptor += 'F'
        elif(val.dtype == np.dtype('f8')): descriptor += 'D'
        else:
            print('Warning, setup issue for branch: ', key, '. Aborting.')
            return
        branches[key] = tree.Branch(key,data[key],descriptor)
        
    # Explicitly take care of the non-basic-type branches, and array branches with ndim >= 2.
    branches['final_state_particles'] = tree.Branch('final_state_particles',data['final_state_particles'], 256 * nparticle_final)
    branches['truth_particles'] = tree.Branch('truth_particles',data['truth_particles'], 256 * nparticle_truth)
    branches['jet_constituent_indices'] = tree.Branch('jet_constituent_indices',data['jet_constituent_indices'],'jet_constituent_indices[njet][' + str(nconst_max) + ']/S')

    # We save a few variables to TTree::fUserInfo, for easy reference.
    # nconst_max: the 2nd dimension of jet_constituent_indices, which is fixed.
    # jet power, radius, pt_min, eta_max
    # nparticle_final: max number of final-state particles saved
    # nparticle_truth: max number of truth-level particles saved
    #
    # The "i_" or "f_" may be used to determine whether to template
    # as float or int upon retrieval.
    user_info = {
    'njet_constituents' : rt.TParameter('int')('i_njet_constituents',nconst_max),
    'jet_power' : rt.TParameter('int')('i_jet_power',jet_power),
    'jet_radius' : rt.TParameter('float')('f_jet_radius',jet_radius),
    'jet_pt_min' : rt.TParameter('float')('f_jet_pt_min',jet_pt_min),
    'jet_eta_max' : rt.TParameter('float')('f_jet_eta_max',jet_eta_max),
    'nparticle_final' : rt.TParameter('int')('i_nparticle_final',nparticle_final),
    'nparticle_truth' : rt.TParameter('int')('i_nparticle_truth',nparticle_truth)
    }
    for key, param in user_info.items(): tree.GetUserInfo().Add(param)

    # Some variables we'll need for the event generation loop.
    # Make a mapping between event listing indices and final-state particle array indices.
    index_mapping = {} # key is the event listing index
    nevents = GetNEvents() # number of events we want to generate
    events_generated = 0 # keep track of successfully-generated events
    percent_print = 0 # for print statements
    percent_print_step = 10 # for print statements - controls frequency
    
    # Event loop
    while(events_generated < nevents):
    
        if(debug): print('Event generation #',events_generated)
        # Clear the index mapping dictionary, and the TClonesArray's
        index_mapping.clear()
        data['final_state_particles'].Clear()
        data['truth_particles'].Clear()
        
        # reset the jet constituent indices, to be safe
        data['jet_constituent_indices'][:,:] = -1
        
        # Generate an event
        pythia8.GenerateEvent()
        pythia8.ImportParticles(particle_array, 'All')
        data['process_id'][0] = pyth.info.code()
        
        # Particle loop
        npar = particle_array.GetEntriesFast() # number of particles in event listing
        final_state_idx = 0
        if(debug): print('Number of particles in the full event listing = ', npar)
        for ip in range(npar):
            particle = particle_array.At(ip)
            # save final-state particles
            ist = particle.GetStatusCode()
            if(debug): print('\tip =',ip,'\t ist =',ist)
            if(ist <= 0): continue
            data['final_state_particles'][final_state_idx] = particle
            index_mapping[ip] = final_state_idx
            final_state_idx += 1
        data['nparticle'][0] = final_state_idx
        if(debug): print('nparticle =',final_state_idx)
        
        # Jet-finding.
        slowjet.analyze(pyth.event)
        njet = slowjet.sizeJet()
        if(njet < njet_min or njet > njet_max): continue
        data['njet'][0] = njet
        
        # Place jets in branches. Ordered by decreasing pT in SlowJet.
        for i in range(njet):
            data['jet_m'][i] = slowjet.p(i).mCalc() # TODO: Check that this is right function call
            data['jet_pt'][i] = slowjet.p(i).pT()
            data['jet_eta'][i] = slowjet.p(i).eta()
            data['jet_phi'][i] = slowjet.p(i).phi()
            data['jet_nconst'][i] = slowjet.multiplicity(i)
            
            # Record jet constituents. SlowJet.Constituents() gives particles indices
            # with respect to the Pythia8 event listing. Note that the event listing
            # uses 1-indexing, since position 0 is occupied by a "system" object.
            # However Pythia8.ImportParticles() gives a 0-indexed list. So we will have
            # to take care of this discrepancy when cross-referencing.
            # Also note that the output of SlowJet.Constituents() is *not* pT-ordered.
            constituent_indices = np.array(slowjet.constituents(i))
            
            # Perform pT-sorting of the constituent indices.
            pt2 = np.array([[pyth.event[int(x)].px(),pyth.event[int(x)].py()] for x in constituent_indices]) # TODO: x is of type np.dtype('i4'), but removing the int() cast causes issues -- why?
            pt_sort = Pt2_sort(pt2)
            constituent_indices = constituent_indices[pt_sort]
            
            # Now save the jet constituent indices.
            for j in range(nconst_max):
                if(j >= data['jet_nconst'][i]): break
                data['jet_constituent_indices'][i,j] = index_mapping[constituent_indices[j] - 1]
              
        # Now we save some truth-level particles.
        # First we get their indices in the Pythia8 event listing
        # (1-indexed, since index 0 gives 'system'),
        # then we convert to 0-indexing for lookup in particle_array
        
        if(truth_selection == 't2Wb'): truth_indices = ts.t2Wb(pyth.event, nparticle_truth) - 1
        else: truth_indices = np.array([-1],dtype=np.dtype('i2'))
        if(debug): print('truth_indices', truth_indices)
        
        # Now we will fill the truth particle TClonesArray using
        # truth_indices, ignoring any negative entries that correspond
        # to a lack of particles (this is *expected*).
        tr_fill_idx = 0
        for tr_idx in truth_indices:
            if(tr_idx < 0): break # break at first negative entry
            data['truth_particles'][tr_fill_idx] = particle_array.At(int(tr_idx))
            tr_fill_idx += 1
        data['ntruth'][0] = tr_fill_idx
        if(debug): print('\tFilled',data['ntruth'][0],'truth particles.')
        
        # Fill the TTree, increase the counter for a successfully-generated event.
        tree.Fill()
        events_generated += 1
        
        # Progress print statement
        percent_done = 100. * events_generated / nevents
        while(percent_done >= percent_print):
            percent_print_string = str(percent_print) + '%.'
            if(percent_print < 10): percent_print_string = ' '+percent_print_string
            print(percent_print_string)
            percent_print += percent_print_step
       
    if(debug): print('Finished the event loop. Writing',tree.GetEntries(), 'entries to the TTree.')
    # Write the TTree to the output file, and close the file.
    tree.Write('',rt.TObject.kOverwrite) # get rid of any extra cycles caused by TTree.AutoSave
    f.Close()
    print('Done. Output is saved to', output_filename)
    return
    
def main(args):
    
    rt.gROOT.SetBatch(True) # not really necessary if not plotting anything
    #TODO: Find out how to get PyROOT to automatically call a local rootlogon.C.
    try: rt.gROOT.Macro('rootlogon.C')
    except:
        print('Was unable to find/launch the local rootlogon.C. Will abort.')
        return
        
    debug = False
    truth_selection = 't2Wb'
    Generate(debug, truth_selection)
    
    # Now clean up some additional files that may have been
    # created if we compiled our local TPythia8.
    files_to_delete = []
    files_to_delete += glob.glob(path_to_this + '/*.pcm')
    files_to_delete += glob.glob(path_to_this + '/*.d')
    for file in files_to_delete:
        os.remove(file)
        
    return
   
if __name__ == '__main__':
    main(sys.argv)
