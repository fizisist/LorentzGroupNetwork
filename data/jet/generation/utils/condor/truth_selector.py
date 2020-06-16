
# truth_selector.py

# Contains functions that define different methods
# by which we determine which truth-level particles
# to save.
# For each function, the input is a Pythia8 event
# (provided via ROOT's TPythia8 wrapper, via
# TPythia8::)

# TODO: Consider writing some GenericSelector function using numba's jit,
#       if the current approach is too slow.

import ROOT as rt
import numpy as np

# Selects particles in decay t -> Wb, if present.
# Only looks in *hardest* process, not in any
# subsequent secondary processes.
# Returns the indices in the event listing.
# (1-indexed, since entry 0 is the 'system' object)
def t2Wb(event, ntruth):
#    names = ['t','tbar','W+','W-','b','bbar'] # names - can alternatively use PDG codes via event[i].id()
    names = ['t','W+','b'] # names - can alternatively use PDG codes via event[i].id()
    npar = event.size() # number of particles in event, incl. system @ 0
    indices = np.full((ntruth),-1,dtype=np.dtype('i2'))
    nfill = 0 # keep track of how many positions have been filled
    for i in range(npar):
        if(nfill >= ntruth): break
        status = event[i].statusAbs() # absolute value of the status
        # we only want to consider particles with |status| \in [20,29]
        # (see Particle Properties @ http://home.thep.lu.se/~torbjorn/pythia81html/Welcome.html)
        if(int(status/10) != 2): continue
        name = event[i].name()
        if(name in names):
            indices[nfill] = i
            nfill += 1
    return indices

        
        
        
        
    
    
    

    

