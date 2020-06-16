# nx_et.py

# Goal: For a given jet, get Nx_E or Nx_ET: the minimum number of particles
#       needed such that their combined (transverse) energy is x% of the jet's
#       (transverse) energy.
#
# Note: Currently, We make the assumption that even in the case of transverse energy,
#       we can do this by ordering constituents by decreasing energy,
#       and taking a running sum of constituents from the top of the list.
#       (This, by definition, works for finding x% of total energy, we
#       empirically find it to work for x% of total E_T as well, with this
#       particular dataset).

import numpy as np, ROOT as rt
from numba import jit

# Get indices to sort jet constituents by decreasing energy
@jit # just-in-time compilation with LLVM -> make this fast
def GetIndices(constituents):
    n = constituents.shape[0]
    distances = np.zeros(n)
    energies = constituents[:,0]
    indices = np.argsort(-energies)
    return indices

# Inputs: List of jet constituents
# Output: List of constituents, ordered by decreasing energy
def MakeList(constituents):
    indices = GetIndices(constituents)
    return constituents[indices]

def GetE(constituents, N, transverse = False):
    n = constituents.shape[0]
    if(N>n): n = N
    ordered_list = MakeList(constituents)
    vec = np.sum(ordered_list[:N],axis=0)
    rvec = rt.TLorentzVector()
    rvec.SetPxPyPzE(vec[1],vec[2],vec[3],vec[0])
    if(transverse): return rvec.Et()
    return rvec.E()

def GetNx(constituents, nconst = -1, x = 90., transverse = False):
    # trim the zero-padding if nconst is provided
    if(nconst != -1): constituents = constituents[:nconst]
    N = constituents.shape[0]
    # get the target energy / et
    jet_vec = np.sum(constituents,axis=0)
    jet = rt.TLorentzVector()
    jet.SetPxPyPzE(jet_vec[1],jet_vec[2],jet_vec[3],jet_vec[0])
    if(transverse): jet_val = jet.Et()
    else: jet_val = jet.E()
    target_val = 0.01 * x * jet_val
    Nx = -1
    for i in range(N):
        val = GetE(constituents, i, transverse)
        if(val >= target_val):
            Nx = i
            break
    return Nx
