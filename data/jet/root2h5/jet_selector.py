# jet_selector.py

# Determine which jet to select for a given event,
# to be passed on to our jet dataset.

import ROOT as rt
import numpy as np

# STRATEGY: top_jet
#
# For events with a truth-level top,
# select the jet closest to the top
# in deltaR. (*ignores anti-tops*)
def top_jet(event):
    
    jet_index = -1
    
    # Get the truth top, if it exists.
    truth_pars = event.truth_particles
    n_truth_pars = truth_pars.GetEntriesFast()
    top_index = -1
    for i in range(n_truth_pars):
        if(truth_pars[i].GetPdgCode() == 6): # top PDG code is 6
            top_index = i
            break
    if(top_index == -1): return jet_index # Top not found. (e.g. background event)

    # TODO: Implementing some new versions of TLorentzVector,
    # which is still widely used but technically a legacy class.
    # These might offer better performance, which is always good news.
    top_eta = truth_pars[top_index].Eta()
    top_phi = truth_pars[top_index].Phi()
    top_vec = rt.Math.PtEtaPhiEVector(0, top_eta, top_phi, 0) # don't need pt, E for deltaR calc

    # Get the jet 4-vectors (eta/phi only), for calculating deltaR^2 's.
    njet = event.njet
    jet_vec = rt.Math.PtEtaPhiEVector(0.,0.,0.,0.)
    dR2 = np.zeros(njet)
    for i in range(njet):
        jet_vec.SetEta(event.jet_eta[i])
        jet_vec.SetPhi(event.jet_phi[i])
        dR2[i] = rt.Math.VectorUtil.DeltaR2(top_vec, jet_vec)
    return dR2.argmin()
        
# STRATEGY: lead_jet
#
# Return the leading jet.
# (i.e. highest-pt)
def lead_jet(event):
    njet = event.njet
    jet_pt = np.zeros(njet,dtype=np.dtype('f8'))
    for i in range(njet): jet_pt[i] = event.jet_pt[i] # TODO: TTree branch indexing doesn't seem to allow for ranges
    return jet_pt.argmax()

